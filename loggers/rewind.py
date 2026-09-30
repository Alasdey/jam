"""Back up a run and replace its active history after a generation restart."""

import json
import os
import shutil
import tempfile
from pathlib import Path

from loggers.experiment_logger import write_checkpoint


def rewind_run(run_dir: str, gen: int, checkpoint: dict) -> Path:
    """Commit a validated restart; keep a complete sibling backup for recovery."""
    root = Path(run_dir)
    metrics = root / "metrics.jsonl"
    kept = []
    found = False
    with open(metrics) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["gen"] <= gen:
                kept.append(line if line.endswith("\n") else line + "\n")
            found |= record["gen"] == gen
    if not found:
        raise ValueError(f"No metrics recorded for resume_gen={gen}")

    later = []
    for pattern in ("populations/births_*.jsonl", "populations/survivors_*.json",
                    "payoffs/payoff_*.npz"):
        for path in root.glob(pattern):
            if int(path.stem.rsplit("_", 1)[-1]) > gen:
                later.append(path)

    # Copy completely before modifying anything. A copy failure leaves the
    # source intact; the hidden sibling cannot be mistaken for a new run.
    backup = Path(tempfile.mkdtemp(
        prefix=f".{root.name}.before_gen_{gen:06d}_", dir=root.parent
    ))
    shutil.copytree(root, backup, dirs_exist_ok=True)
    tmp = root / "metrics.jsonl.tmp"
    try:
        tmp.write_text("".join(kept))
        os.replace(tmp, metrics)
        for path in later:
            path.unlink()
        write_checkpoint(root, checkpoint)
    except Exception:
        # Recover the previous run after an ordinary I/O failure. The backup
        # is also retained for recovery after a process/machine interruption.
        shutil.copytree(backup, root, dirs_exist_ok=True)
        raise
    finally:
        tmp.unlink(missing_ok=True)
    return backup
