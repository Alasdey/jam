"""
Treemo is jam's only interpreter, and it runs on the GPU (interpreters/treemo_gpu).

interpreters/treemo_c (C, via ctypes) and interpreters/treemo_py (the
language spec in Python) are the references the GPU kernels are tested
against; nothing in a run uses them.
"""

from config import ExperimentConfig
from interpreters.treemo_gpu.treemo import TreemoRuntime


def build_runtime(cfg: ExperimentConfig) -> TreemoRuntime:
    if cfg.interpreter != "treemo":
        raise ValueError(f"jam runs Treemo only, on the GPU; got interpreter {cfg.interpreter!r}")
    return TreemoRuntime.from_config(cfg)
