
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# --- Code generation config ---

@dataclass
class CodeConfig:
    code_length: int = 500
    min_val: int = -500
    max_val: int = 1_000

@dataclass
class RuleConfig:
    code_length: int = 50
    min_val: int = 0
    max_val: int = 100

# --- Interpreter-specific config ---

@dataclass
class IconfractranConfig:
    max_step: int = 200

@dataclass
class TreemoConfig:
    max_step: int = 50
    tree_size: int = 100
    # Rule advancement strategy:
    #   pass_mode=True  → a rule fires at most once before the interpreter moves on
    #   pass_mode=False → a rule fires until it no longer matches before moving on
    pass_mode: bool = False # True makes any ouput an inert input
    # Where to move after a rule fires and the interpreter advances:
    #   first_mode=True  → restart from rule 0
    #   first_mode=False → continue to the next rule in sequence
    first_mode: bool = False

@dataclass
class SubleqConfig:
    library_path: str = "./interpreters/subleq/libsubleq.so" # Should this be hardcoded in the subleq code ?
    max_output_length: int = 2_000
    max_iter: int = 20_000

# --- Payoff / parallelism config ---

@dataclass
class PayoffConfig:
    # 1 = sequential, >1 = use ProcessPoolExecutor
    n_workers: int = 1


# --- Selection step configs (ordered pipeline applied each generation) ---

@dataclass
class SkimStepConfig:
    kind: str = "skim"
    # rounds of iterated elimination of strictly dominated strategies
    n_rounds: int = 3
    # fraction of the dominated set actually dropped per round (1.0 = all)
    fraction: float = 1.0
    # stop skimming once the population is below this size (None = never)
    n_accepted: Optional[int] = 2_000


@dataclass
class CapStepConfig:
    # "cap_top" keeps the best payoff-row-sums, "cap_random" downsamples uniformly
    kind: str = "cap_top"
    max_pop: int = 1_000


@dataclass
class NashStepConfig:
    kind: str = "nash"


# --- Genetics / operator config ---

@dataclass
class GeneticsConfig:
    mutation_rate: float = 0.05
    crossover_prob: float = 0.5
    homoiconic_prob: float = 0.3

    # Integer-program operator selection ("uniform" | "creep")
    code_mutation_op: str = "uniform"
    # ("single_point" | "two_point" | "uniform")
    code_crossover_op: str = "two_point"

    # Tree operator selection ("leaf" | "subtree")
    tree_mutation_op: str = "subtree"
    # ("depth1" | "random_depth")
    tree_crossover_op: str = "random_depth"


# --- Experiment config ---

@dataclass
class ExperimentConfig:
    ### Which interpreter / reward to use
    # subleq | iconfractran | treemo | treemo_py
    interpreter: str = "treemo"
    # blind|placeholder|quine_pressure
    reward: str = "blind" 

    ### Sub-configs
    subleq: SubleqConfig = field(default_factory=SubleqConfig)
    iconfractran: IconfractranConfig = field(default_factory=IconfractranConfig)
    treemo: TreemoConfig = field(default_factory=TreemoConfig)

    payoff: PayoffConfig = field(default_factory=PayoffConfig)
    genetics: GeneticsConfig = field(default_factory=GeneticsConfig)

    code: CodeConfig = field(default_factory=CodeConfig)
    # code: RuleConfig = field(default_factory=RuleConfig)


# --- Random baseline experiment config ---

@dataclass
class RandomBaselineConfig:
    n_ref: int = 10**1
    n_tested: int = 10**7
    n_grain: int = 500
    out_path: str = "outputs/random_baseline/" + time.strftime("%Y%m%d_%H%M%S")
    out_name: str = "results.json"
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# --- Unified run config (driven by core.loop.run) ---

@dataclass
class RunConfig:
    # RNG seed; resolved to a concrete value at startup and persisted in config.json
    seed: Optional[int] = 42
    # extra random individuals injected at generation 0 only (fresh starts)
    n_init: int = 0
    # store pop_ids whose individuals seed generation 0 (composite runs)
    seed_populations: list[str] = field(default_factory=list)
    # fresh randoms / genetic offspring injected per generation
    n_random: int = 10**2
    n_offspring: int = 10**2
    n_iter: int = 10**4
    # ordered selection pipeline applied after payoff extension each generation
    selection: list = field(default_factory=lambda: [SkimStepConfig(), CapStepConfig()])
    # persist payoffs/payoff_XXXXXX.npz every N generations + final (0 = never)
    payoff_every: int = 10*3
    # False: persist only newborns that survive their birth generation
    log_all_births: bool = False
    # path to a previous run's out_dir to continue (runs n_iter MORE generations)
    resume_from: Optional[str] = "outputs/main/20260719_173059" # None
    # if set, publish the final population to the store under this label
    publish_label: Optional[str] = 'initial_test'
    store_dir: str = "outputs/store/populations"
    out_dir: str = field(default_factory=lambda: "outputs/run/" + time.strftime("%Y%m%d_%H%M%S"))
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        if self.resume_from:
            self.out_dir = self.resume_from


def _stamped(prefix: str) -> str:
    return prefix + time.strftime("%Y%m%d_%H%M%S")


def preset_main() -> RunConfig:
    # the historical main.py loop: randoms + offspring, skim then cap_top
    return RunConfig(out_dir=_stamped("outputs/main/"))


def preset_evolution() -> RunConfig:
    # the historical sides/evolution.py loop: fixed initial pop, offspring only
    return RunConfig(
        n_init=50,
        n_random=0,
        n_offspring=20,
        n_iter=1000,
        selection=[SkimStepConfig(n_rounds=2, n_accepted=None), CapStepConfig(max_pop=500)],
        out_dir=_stamped("outputs/evolution/"),
    )


def preset_random_skimmed() -> RunConfig:
    # the historical sides/random_skimmed.py loop: random injection only
    return RunConfig(
        n_random=10**3,
        n_offspring=0,
        selection=[SkimStepConfig(), CapStepConfig(kind="cap_random", max_pop=2_000)],
        out_dir=_stamped("outputs/random_skimmed/"),
    )


PRESETS: dict[str, Callable[[], RunConfig]] = {
    "main": preset_main,
    "evolution": preset_evolution,
    "random_skimmed": preset_random_skimmed,
}


# --- Config hashing / reconstruction ---

def _canonical_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


# Which sub-config carries an interpreter's execution semantics.
INTERPRETER_CFG_FIELD = {
    "subleq": "subleq",
    "iconfractran": "iconfractran",
    "treemo": "treemo",
    "treemo_py": "treemo",
}


def compat_key(exp_cfg_dict: dict) -> str:
    """
    Hash of execution semantics only: the interpreter name and its sub-config.
    Two populations can play each other iff their compat_keys match. Creation
    settings (code bounds, genetics) and the reward are excluded: they shape
    how programs were made, not how they run. treemo and treemo_py are kept
    distinct on purpose — the implementations are meant to agree but have not
    been proven equivalent.
    """
    name = exp_cfg_dict["interpreter"]
    payload = {"interpreter": name, "config": exp_cfg_dict[INTERPRETER_CFG_FIELD[name]]}
    return _canonical_hash(payload)


def method_key(run_cfg_dict: dict) -> str:
    """
    Hash identifying a methodology: the full run config minus run identity
    (seed, paths). Same-methodology samples across seeds share this key.
    """
    excluded = ("seed", "out_dir", "resume_from", "publish_label", "store_dir")
    payload = {k: v for k, v in run_cfg_dict.items() if k not in excluded}
    return _canonical_hash(payload)


def exp_cfg_from_dict(d: dict) -> ExperimentConfig:
    """Rebuild an ExperimentConfig from a persisted config dict (fails loudly on drift)."""
    return ExperimentConfig(
        interpreter=d["interpreter"],
        reward=d["reward"],
        subleq=SubleqConfig(**d["subleq"]),
        iconfractran=IconfractranConfig(**d["iconfractran"]),
        treemo=TreemoConfig(**d["treemo"]),
        payoff=PayoffConfig(**d["payoff"]),
        genetics=GeneticsConfig(**d["genetics"]),
        code=CodeConfig(**d["code"]),
    )