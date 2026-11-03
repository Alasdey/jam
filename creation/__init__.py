from config import ExperimentConfig
from creation.treemo import TreemoCreator


def build_creator(cfg: ExperimentConfig, runtime) -> TreemoCreator:
    return TreemoCreator(cfg, runtime)
