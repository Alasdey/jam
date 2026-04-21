
import time
from dataclasses import dataclass, field

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


# --- Genetics / operator config ---

@dataclass
class GeneticsConfig:
    mutation_rate: float = 0.05
    crossover_prob: float = 0.5
    homoiconic_prob: float = 0.3

    # Integer-program operator selection ("uniform" | "creep")
    code_mutation_op: str = "uniform"
    # ("single_point" | "two_point" | "uniform")
    code_crossover_op: str = "single_point"

    # Tree operator selection ("leaf" | "subtree")
    tree_mutation_op: str = "leaf"
    # ("depth1" | "random_depth")
    tree_crossover_op: str = "depth1"


# --- Experiment config ---

@dataclass
class ExperimentConfig:
    ### Which interpreter / reward to use
    # subleq|iconfractran|treemo
    interpreter: str = "treemo"
    # blind|placeholder|quine_pressure
    reward: str = "quine_pressure" 

    ### Sub-configs
    subleq: SubleqConfig = field(default_factory=SubleqConfig)
    iconfractran: IconfractranConfig = field(default_factory=IconfractranConfig)
    treemo: TreemoConfig = field(default_factory=TreemoConfig)

    payoff: PayoffConfig = field(default_factory=PayoffConfig)
    genetics: GeneticsConfig = field(default_factory=GeneticsConfig)

    # code: CodeConfig = field(default_factory=CodeConfig)
    code: RuleConfig = field(default_factory=RuleConfig)


# --- Random baseline experiment config ---

@dataclass
class RandomBaselineConfig:
    n_ref: int = 10**1
    n_tested: int = 10**7
    n_grain: int = 500
    out_path: str = "outputs/random_baseline/" + time.strftime("%Y%m%d_%H%M%S")
    out_name: str = "results.json"
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)