"""
Config-derived identity keys and reconstruction.

Hashes that decide what a persisted config *means*: which populations may
meet in a tournament (compat_key), which samples share a methodology
(method_key), and how to rebuild an ExperimentConfig from a stored dict.
Kept out of config.py so that module stays a declaration of values.
"""

import hashlib
import json

from config import (
    CodeConfig,
    ExperimentConfig,
    GeneticsConfig,
    IconfractranConfig,
    PayoffConfig,
    SubleqConfig,
    TreemoConfig,
)


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
    excluded = ("seed", "out_dir", "resume_from", "publish_label", "store_dir")
    payload = {k: v for k, v in run_cfg_dict.items() if k not in excluded}
    return _canonical_hash(payload)


def exp_cfg_from_dict(d: dict) -> ExperimentConfig:
    """Rebuild an ExperimentConfig from a persisted config dict (fails loudly on drift)."""
    return ExperimentConfig(
        interpreter=d["interpreter"],
        reward=d["reward"],
        subleq=SubleqConfig(**d["subleq"]),
        iconfractran=IconfractranConfig(**d["iconfractran"]),
        treemo=TreemoConfig(**d["treemo"]),
        payoff=PayoffConfig(**d["payoff"]),
        genetics=GeneticsConfig(**d["genetics"]),
        code=CodeConfig(**d["code"]),
    )
