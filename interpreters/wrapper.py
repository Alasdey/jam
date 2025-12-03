# interpreters/wrappers.py

from config import ExperimentConfig, SubleqConfig
from interpreters.subleq.subleq import SubleqInterpreter


def make_subleq_interpreter(cfg: SubleqConfig) -> SubleqInterpreter:
    """
    Build a SubleqInterpreter from its specific config.
    """
    return SubleqInterpreter(
        library_path=cfg.library_path,
    )


def make_interpreter(cfg: ExperimentConfig):
    """
    Top-level interpreter factory.
    Decides which interpreter to build based on cfg.interpreter.
    """
    if cfg.interpreter_name == "subleq":
        return make_subleq_interpreter(cfg.subleq)

    raise ValueError(f"Unknown interpreter: {cfg.interpreter_name}")
