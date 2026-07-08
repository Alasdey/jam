from typing import Callable

from config import ExperimentConfig
from core.types import Interpreter

# Builders import lazily so that selecting one interpreter never loads
# another's shared library.


def _build_subleq(cfg: ExperimentConfig) -> Interpreter:
    from interpreters.subleq.subleq import SubleqInterpreter

    return SubleqInterpreter(
        library_path=cfg.subleq.library_path,
        max_output_length=cfg.subleq.max_output_length,
        max_iter=cfg.subleq.max_iter,
    )


def _build_iconfractran(cfg: ExperimentConfig) -> Interpreter:
    from interpreters.iconfractran.iconfractran import IconfractranInterpreter

    return IconfractranInterpreter(max_step=cfg.iconfractran.max_step)


def _build_treemo_c(cfg: ExperimentConfig) -> Interpreter:
    from interpreters.treemo_c.treemo import TreemoInterpreter

    return TreemoInterpreter(
        max_step=cfg.treemo.max_step,
        pass_mode=cfg.treemo.pass_mode,
        first_mode=cfg.treemo.first_mode,
    )


def _build_treemo_py(cfg: ExperimentConfig) -> Interpreter:
    from interpreters.treemo_py.treemo import TreemoInterpreter

    return TreemoInterpreter(
        max_step=cfg.treemo.max_step,
        pass_mode=cfg.treemo.pass_mode,
        first_mode=cfg.treemo.first_mode,
    )


INTERPRETERS: dict[str, Callable[[ExperimentConfig], Interpreter]] = {
    "subleq": _build_subleq,
    "iconfractran": _build_iconfractran,
    "treemo": _build_treemo_c,
    "treemo_py": _build_treemo_py,
}


def build_interpreter(cfg: ExperimentConfig) -> Interpreter:
    return INTERPRETERS[cfg.interpreter](cfg)
