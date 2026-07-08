from typing import Callable

from config import ExperimentConfig
from creation.base import Creator


def _build_treemo(cfg: ExperimentConfig) -> Creator:
    from creation.treemo import TreemoCreator

    return TreemoCreator(cfg)


def _build_subleq(cfg: ExperimentConfig) -> Creator:
    from creation.subleq import SubleqCreator

    return SubleqCreator(cfg)


def _build_iconfractran(cfg: ExperimentConfig) -> Creator:
    from creation.iconfractran import IconfractranCreator

    return IconfractranCreator(cfg)


# Keyed by ExperimentConfig.interpreter: the genome space is tied to the
# interpreter family, so treemo_py shares the treemo creator.
CREATORS: dict[str, Callable[[ExperimentConfig], Creator]] = {
    "treemo": _build_treemo,
    "treemo_py": _build_treemo,
    "subleq": _build_subleq,
    "iconfractran": _build_iconfractran,
}


def build_creator(cfg: ExperimentConfig) -> Creator:
    return CREATORS[cfg.interpreter](cfg)
