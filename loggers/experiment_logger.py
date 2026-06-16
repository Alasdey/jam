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
    Writes files into out_dir:
      config.json            — full dataclass config, serialized once at init (or config_resumed_<ts>.json)
      meta.json              — run metadata (time, git hash) (or meta_resumed_<ts>.json)
      metrics.jsonl          — one JSON record per generation/iteration (append)
      populations/           — per-step population dumps
        work_pop_XXXXXX.json — one individual dict per line: {id, method, parents, genome}
        ref_pop_XXXXXX.json  — reference population (same format)
    When out_dir already exists, logs resume seamlessly: step counter continues,
    metrics.jsonl is appended, and config/meta are written under timestamped names.
    """

    def __init__(self, out_dir: str, cfg, **meta_extra: Any):
        self.out_dir = Path(out_dir)
        resuming = self.out_dir.exists()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%dT%H%M%S")
        cfg_name = f"config_resumed_{ts}.json" if resuming else "config.json"
        with open(self.out_dir / cfg_name, "w") as f:
            json.dump(asdict(cfg), f, indent=2)

        meta_name = f"meta_resumed_{ts}.json" if resuming else "meta.json"
        meta = {
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

        if resuming:
            existing = sorted(self._pops_dir.glob("work_pop_*.json"))
            self._step = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 0
        else:
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
