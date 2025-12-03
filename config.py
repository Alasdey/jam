
import time
from dataclasses import dataclass, field

# --- Code generation config ---

@dataclass
class CodeConfig:
    code_length: int = 500
    min_val: int = -500
    max_val: int = 1000


# --- Interpreter-specific config ---

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


# --- Experiment config ---

@dataclass
class ExperimentConfig:
    # Which interpreter / reward to use (later: "brainfuck", etc.)
    interpreter: str = "subleq"
    reward: str = "blind"

    # Sub-configs
    subleq: SubleqConfig = field(default_factory=SubleqConfig)
    payoff: PayoffConfig = field(default_factory=PayoffConfig)
    code: CodeConfig = field(default_factory=CodeConfig)


# --- Random baseline experiment config ---

@dataclass
class RandomBaselineConfig:
    n_ref: int = 10**3
    n_tested: int = 10**7
    out_path: str = "outputs/random_baseline/" + time.strftime("%Y%m%d_%H%M%S")
    out_name: str = "results.json"
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)