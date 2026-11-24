
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# --- Interpreter config ---

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

# --- Payoff config (everything runs on the GPU) ---

@dataclass
class PayoffConfig:
    # Device memory of one Treemo execution launch (states and scan
    # watermarks); larger batches are split into several launches.
    gpu_memory_mb: int = 1024
    # Matchups scored per batch; bigger blocks go in tiles, so memory stays
    # bounded (all outputs of a batch stay on the device until it is scored).
    gpu_batch_matchups: int = 1_000_000


# --- Selection step configs (ordered pipeline applied each generation) ---

@dataclass
class DedupeStepConfig:
    # Keep one individual per distinct genome (exact match, first occurrence).
    # Reads no payoff, so it belongs in RunConfig.pre_selection, where the
    # duplicates it drops never cost a matchup.
    # Selection by payoff alone cannot do this: identical genomes have identical
    # payoff rows, so neither strictly dominates the other (skim) and their
    # row-sums tie (cap_top) — duplicates always survive as a block.
    kind: str = "dedupe"


@dataclass
class MaxLengthStepConfig:
    max_length: int
    kind: str = "max_length"


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
class LexicaseStepConfig:
    kind: str = "lexicase"
    # Number of unique survivors, selected without replacement.
    n_accepted: int = 1_000


@dataclass
class KNNNoveltyStepConfig:
    kind: str = "knn_novelty"
    n_accepted: int = 1_000
    # Rank by mean Euclidean distance to the k nearest other payoff rows.
    k: int = 10


@dataclass
class KNNScoreStepConfig:
    kind: str = "knn_score"
    n_accepted: int = 1_000
    # Keep the largest mean distance to k nearest other programs' total payoffs.
    k: int = 10


@dataclass
class BestWorstRandomStepConfig:
    kind: str = "best_worst_random"
    # Best total payoff scores, then worst remaining, then random remaining.
    n_best: int = 1_000
    n_worst: int = 0
    n_rand: int = 0


@dataclass
class MixtureStepConfig:
    kind: str = "mixture"
    n_best: int = 1_000
    # Pool of next-best payoff row-sums, after excluding n_best.
    n_good: int = 1_000
    # Select from that pool by scalar kNN distance against the full population.
    n_good_novel: int = 0
    # Select from all unselected programs by full-population payoff-row novelty.
    n_novelty: int = 0
    k: int = 10


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
    # Per-original-node event probability (subtree); per-tree probability (leaf).
    mutation_rate: float = 0.05
    crossover_prob: float = 0.8
    homoiconic_prob: float = 0.3

    # "subtree": uniform delete/swap/insert/regenerate events; "leaf": legacy operator.
    tree_mutation_op: str = "subtree"
    # ("depth1" | "random_depth")
    tree_crossover_op: str = "random_depth"


# --- Experiment config ---

@dataclass
class ExperimentConfig:
    # Treemo is the only interpreter; the field stays so compat keys of
    # stored populations (core/config_keys.py) keep their values.
    interpreter: str = "treemo"
    # a key of rewards.base.REWARDS: blind | placeholder | quine_pressure
    reward: str = "blind"

    treemo: TreemoConfig = field(default_factory=TreemoConfig)
    payoff: PayoffConfig = field(default_factory=PayoffConfig)
    genetics: GeneticsConfig = field(default_factory=GeneticsConfig)


# --- Random baseline experiment config ---

@dataclass
class RandomBaselineConfig:
    # seed of the random trees (reference population and every pool)
    seed: int = 0
    n_ref: int = 10**1
    n_tested: int = 10**7
    n_grain: int = 500
    out_path: str = "outputs/random_baseline/" + time.strftime("%Y%m%d_%H%M%S")
    out_name: str = "results.json"
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# --- Online loop (core/loop.py): worker processes, lag, memory ---

@dataclass
class StreamConfig:
    # Worker processes that create and test batches, each on its own CUDA
    # context (Python threads cannot: the GIL serializes their host work).
    # 0 makes every batch in the coordinator, one at a time (tests,
    # debugging); the run is the same either way.
    n_workers: int = 3
    # Batches in flight: batch t is bred from the population after batch
    # t - in_flight was inserted (or at the start of its epoch). At least
    # n_workers, so workers keep testing while one batch is inserted.
    in_flight: int = 6
    # CuPy pool cap of each worker and of the coordinator. Together, with
    # ~0.5 GB of CUDA context per process, they must stay below the GPU's
    # dedicated memory minus what other programs use, or the driver spills
    # into shared system memory. A batch that hits its cap is made again in
    # smaller launches (same results).
    worker_memory_mb: int = 1_500
    coordinator_memory_mb: int = 1_500


# --- Unified run config (driven by core.loop.run) ---

@dataclass
class RunConfig:
    # RNG seed; resolved to a concrete value at startup and persisted in config.json
    seed: Optional[int] = 42
    # random individuals of the initial population (fresh starts), tested and
    # selected before the stream starts
    n_init: int = 0
    # store pop_ids whose individuals join the initial population (composite runs)
    seed_populations: list[str] = field(default_factory=list)
    # one batch of the stream: fresh random codes and children of the population
    n_random: int = 6
    n_offspring: int = 58
    # insertions per epoch: the unit of logging, checkpoints and resume (the
    # "gen" of the run-directory files); the stream drains between epochs
    epoch_batches: int = 100
    # epochs to run
    n_iter: int = 10**3
    stream: StreamConfig = field(default_factory=StreamConfig)
    # selection steps applied BEFORE testing, on genomes alone: new codes they
    # drop never cost a matchup (in the worker, against the batch's snapshot,
    # then again at insertion, against the current population). Only
    # payoff-free steps may go here (selection.base validates).
    pre_selection: list = field(default_factory=lambda: [MaxLengthStepConfig(max_length=10000), DedupeStepConfig()])
    # selection pipeline applied at each insertion, to the square payoff
    # matrix of the population and the batch's new codes
    selection: list = field(default_factory=lambda: [BestWorstRandomStepConfig(n_best=150, n_worst=20, n_rand=30)])
    # persist payoffs/payoff_XXXXXX.npz every N epochs + final (0 = never)
    payoff_every: int = 10*3
    # False: persist only new codes that survive their insertion
    log_all_births: bool = False
    # path to a previous run's out_dir to continue (runs n_iter MORE epochs)
    resume_from: Optional[str] = None
    # Restore the population AFTER this epoch; continue at resume_gen + 1.
    # None resumes the latest checkpoint.
    resume_gen: Optional[int] = None
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
    # randoms + offspring in every batch, best/worst/random survival
    return RunConfig(out_dir=_stamped("outputs/main/"))


def preset_evolution() -> RunConfig:
    # a fixed initial population, offspring only
    return RunConfig(
        n_init=50,
        n_random=0,
        n_offspring=20,
        n_iter=100,
        selection=[SkimStepConfig(n_rounds=2, n_accepted=None), CapStepConfig(max_pop=500)],
        out_dir=_stamped("outputs/evolution/"),
    )


def preset_random_skimmed() -> RunConfig:
    # random injection only
    return RunConfig(
        n_random=64,
        n_offspring=0,
        selection=[SkimStepConfig(), CapStepConfig(kind="cap_random", max_pop=2_000)],
        out_dir=_stamped("outputs/random_skimmed/"),
    )


PRESETS: dict[str, Callable[[], RunConfig]] = {
    "main": preset_main,
    "evolution": preset_evolution,
    "random_skimmed": preset_random_skimmed,
}
