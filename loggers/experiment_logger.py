import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class ExperimentLogger:
    """
    Writes four files into out_dir:
      config.json    — full dataclass config, serialized once at init
      meta.json      — run metadata (time, git hash, script params)
      metrics.jsonl  — one JSON record per generation/iteration (append)
      work_pop.json  — current working population, overwritten each step
      ref_pop.json   — reference population, overwritten each step
    """

    def __init__(self, out_dir: str, cfg, **meta_extra: Any):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        with open(self.out_dir / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2)

        meta = {
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "start_ts": time.time(),
            "git_hash": _git_hash(),
            **meta_extra,
        }
        with open(self.out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._metrics_path = self.out_dir / "metrics.jsonl"
        self._pops_dir = self.out_dir / "populations"
        self._pops_dir.mkdir()
        self._step = 0

    def log(
        self,
        record: dict[str, Any],
        work_pop: Optional[list] = None,
        ref_pop: Optional[list] = None,
    ) -> None:
        with open(self._metrics_path, "a") as f:
            json.dump(record, f)
            f.write("\n")
        if work_pop is not None:
            self._write_pop(work_pop, f"work_pop_{self._step:06d}.json")
        if ref_pop is not None:
            self._write_pop(ref_pop, f"ref_pop_{self._step:06d}.json")
        self._step += 1

    def _write_pop(self, pop: list, filename: str) -> None:
        with open(self._pops_dir / filename, "w") as f:
            for ind in pop:
                json.dump(ind, f)
                f.write("\n")
