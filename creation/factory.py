from config import ExperimentConfig
from creation.base import Creator


def make_creator(cfg: ExperimentConfig) -> Creator:
    if cfg.interpreter == "treemo":
        from creation.treemo import TreemoCreator
        return TreemoCreator(cfg)
    if cfg.interpreter == "subleq":
        from creation.subleq import SubleqCreator
        return SubleqCreator(cfg)
    if cfg.interpreter == "iconfractran":
        from creation.iconfractran import IconfractranCreator
        return IconfractranCreator(cfg)
    raise ValueError(f"Unknown interpreter: {cfg.interpreter!r}")
