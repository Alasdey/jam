"""
Config-derived identity keys and reconstruction.

Hashes that decide what a persisted config *means*: which populations may
meet in a tournament (compat_key), which samples share a methodology
(method_key), and how to rebuild an ExperimentConfig from a stored dict.
Kept out of config.py so that module stays a declaration of values.
"""

import hashlib
import json

from config import ExperimentConfig, GeneticsConfig, PayoffConfig, TreemoConfig


def _canonical_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]


# Which sub-config carries an interpreter's execution semantics.
INTERPRETER_CFG_FIELD = {
    "subleq": "subleq",
    "iconfractran": "iconfractran",
    "treemo": "treemo",
    "treemo_py": "treemo",
}


def compat_key(exp_cfg_dict: dict) -> str:
    """
    Hash of execution semantics only: the interpreter name and its sub-config.
    Two populations can play each other iff their compat_keys match. Creation
    settings (code bounds, genetics) and the reward are excluded: they shape
    how programs were made, not how they run. treemo and treemo_py are kept
    distinct on purpose — the implementations are meant to agree but have not
    been proven equivalent.
    """
    name = exp_cfg_dict["interpreter"]
    payload = {"interpreter": name, "config": exp_cfg_dict[INTERPRETER_CFG_FIELD[name]]}
    return _canonical_hash(payload)


def method_key(run_cfg_dict: dict) -> str:
    """
    Hash identifying a methodology: the full run config minus run identity
    (seed, paths). Same-methodology samples across seeds share this key.
    """
    excluded = ("seed", "out_dir", "resume_from", "resume_gen", "publish_label", "store_dir")
    payload = {k: v for k, v in run_cfg_dict.items() if k not in excluded}
    return _canonical_hash(payload)


# Keys of configs written before jam ran on the GPU (other interpreters, CPU
# workers): dropped on reading, so stored populations and tournaments stay usable.
_LEGACY_TOP = {"subleq", "iconfractran", "code"}
_LEGACY_PAYOFF = {"n_workers", "chunksize", "backend", "gpu_workers"}
_LEGACY_GENETICS = {"code_mutation_op", "code_crossover_op"}
_TOP = {"interpreter", "reward", "treemo", "payoff", "genetics"}


def exp_cfg_from_dict(d: dict) -> ExperimentConfig:
    """
    Rebuild an ExperimentConfig from a persisted config dict. Legacy keys are
    dropped by name; any other unknown key fails loudly. Payoff resources an
    old config does not mention (GPU memory, batch size) take today's values.
    """
    unknown = set(d) - _TOP - _LEGACY_TOP
    if unknown:
        raise ValueError(f"unknown experiment config keys: {sorted(unknown)}")
    if d["interpreter"] != "treemo":
        raise ValueError(f"jam runs Treemo only; this config is for {d['interpreter']!r}")
    payoff = {k: v for k, v in d["payoff"].items() if k not in _LEGACY_PAYOFF}
    genetics = {k: v for k, v in d["genetics"].items() if k not in _LEGACY_GENETICS}
    return ExperimentConfig(
        interpreter=d["interpreter"],
        reward=d["reward"],
        treemo=TreemoConfig(**d["treemo"]),
        payoff=PayoffConfig(**payoff),
        genetics=GeneticsConfig(**genetics),
    )
