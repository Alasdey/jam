# interpreters/wrappers.py

from config import ExperimentConfig, SubleqConfig, IconfractranConfig
from interpreters.subleq.subleq import SubleqInterpreter
from interpreters.iconfractran import IconfractranInterpreter


def make_subleq_interpreter(cfg: SubleqConfig) -> SubleqInterpreter:
    """
    Build a SubleqInterpreter from its specific config.
    """
    return SubleqInterpreter(
        library_path=cfg.library_path,
        max_output_length=cfg.max_output_length,
        max_iter=cfg.max_iter,
    )

def make_iconfractran_interpreter(cfg: IconfractranConfig):
    """
    Build a IconfractranInterpreter from its specific config.
    """
    return IconfractranInterpreter(max_step=cfg.max_step)

def make_interpreter(cfg: ExperimentConfig):
    """
    Top-level interpreter factory.
    Decides which interpreter to build based on cfg.interpreter.
    """
    if cfg.interpreter == "subleq":
        return make_subleq_interpreter(cfg.subleq)
    if cfg.interpreter == "iconfractran":
        return make_iconfractran_interpreter(cfg.iconfractran)

    raise ValueError(f"Unknown interpreter: {cfg.interpreter_name}")
