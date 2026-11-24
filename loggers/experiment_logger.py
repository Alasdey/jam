import json
import os
import pickle
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.population import METHODS


def write_checkpoint(out_dir: Path, blob: dict[str, Any]) -> None:
    """Atomically replace the current resume state."""
    tmp = out_dir / "checkpoint.pkl.tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(blob, f)
        os.replace(tmp, out_dir / "checkpoint.pkl")
    finally:
        tmp.unlink(missing_ok=True)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class ExperimentLogger:
    """
    Writes the run directory (formats documented in docs/formats.md):
      config.json             — full config dataclass, resolved seed included
                                (config_resumed_<ts>.json on resume)
      meta.json               — start time, git hash, format_version
                                (meta_resumed_<ts>.json on resume)
      metrics.jsonl           — one JSON record per insertion (append)
      populations/            — XXXXXX counts epochs (RunConfig.epoch_batches insertions)
        births_XXXXXX.jsonl   — new codes inserted during epoch XXXXXX
        survivors_XXXXXX.json — {"gen", "ids", "scores", "next_batch"} at the epoch's end
      payoffs/
        payoff_XXXXXX.npz     — arrays: payoff (int32, square), ids (int64)
      checkpoint.pkl          — volatile resume state, atomically replaced;
                                NOT a stable format (stable data lives in the
                                JSON/JSONL/npz files above)
    """

    def __init__(self, out_dir: str, cfg, **meta_extra: Any):
        self.out_dir = Path(out_dir)
        resuming = (self.out_dir / "config.json").exists()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%dT%H%M%S")
        cfg_name = f"config_resumed_{ts}.json" if resuming else "config.json"
        with open(self.out_dir / cfg_name, "w") as f:
            json.dump(asdict(cfg), f, indent=2)

        meta_name = f"meta_resumed_{ts}.json" if resuming else "meta.json"
        meta = {
            "format_version": 1,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "start_ts": time.time(),
            "git_hash": _git_hash(),
            **meta_extra,
        }
        with open(self.out_dir / meta_name, "w") as f:
            json.dump(meta, f, indent=2)

        self._metrics_path = self.out_dir / "metrics.jsonl"
        self._pops_dir = self.out_dir / "populations"
        self._pops_dir.mkdir(exist_ok=True)
        self._payoffs_dir = self.out_dir / "payoffs"
        self._payoffs_dir.mkdir(exist_ok=True)

    def log_metrics(self, record: dict[str, Any]) -> None:
        with open(self._metrics_path, "a") as f:
            json.dump(record, f)
            f.write("\n")

    def log_population(
        self,
        gen: int,
        births,
        survivor_ids: list[int],
        scores: list[int],
        payoff: Optional[np.ndarray] = None,
        **survivor_fields: Any,
    ) -> None:
        """births: a core.population.Population of the new codes to persist."""
        if len(births):
            with open(self._pops_dir / f"births_{gen:06d}.jsonl", "w") as f:
                for i, bits in enumerate(births.bits()):
                    parents = ", ".join(str(int(p)) for p in births.parents[i] if p >= 0)
                    f.write(f'{{"id": {int(births.ids[i])}, "genome": [{_genome_json(bits)}], '
                            f'"method": "{METHODS[births.method[i]]}", "parents": [{parents}], '
                            f'"born_gen": {int(births.born_gen[i])}}}\n')

        with open(self._pops_dir / f"survivors_{gen:06d}.json", "w") as f:
            json.dump({"gen": gen, "ids": survivor_ids, "scores": scores, **survivor_fields}, f)

        if payoff is not None:
            np.savez_compressed(
                self._payoffs_dir / f"payoff_{gen:06d}.npz",
                payoff=payoff.astype(np.int32),
                ids=np.array(survivor_ids, dtype=np.int64),
            )

    def write_checkpoint(self, blob: dict[str, Any]) -> None:
        write_checkpoint(self.out_dir, blob)


def _genome_json(bits: np.ndarray) -> str:
    """The JSON list body of a 0/1 genome ("1, 0, 1"), built without Python ints."""
    if not len(bits):
        return ""
    text = np.full(3 * len(bits) - 2, ord(" "), dtype=np.uint8)
    text[0::3] = bits + ord("0")
    text[1::3] = ord(",")
    return text.tobytes().decode()
