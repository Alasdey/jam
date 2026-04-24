
import json
import time
from pathlib import Path
from typing import List

from config import RandomBaselineConfig
from utils.experiment_logger import ExperimentLogger
from creation.factory import make_creator
from rewards.payoff import compute_payoff_matrix
from rewards.wrapper import make_reward


def main(rb_cfg: RandomBaselineConfig) -> None:
    cfg = rb_cfg.experiment
    creator = make_creator(cfg)

    ref_pop = [creator.random() for _ in range(rb_cfg.n_ref)]

    logger = ExperimentLogger(
        rb_cfg.out_path, rb_cfg,
        n_ref=rb_cfg.n_ref, n_tested=rb_cfg.n_tested, n_grain=rb_cfg.n_grain,
    )

    reward_fn = make_reward(cfg)
    bests = []

    for i in range(rb_cfg.n_tested):
        print("Up to:", i * rb_cfg.n_grain, end="\r")
        pool = [creator.random() for _ in range(rb_cfg.n_grain)]

        t0 = time.time()
        payoff = compute_payoff_matrix(cfg=cfg, ref=ref_pop, pop=pool, reward_fn=reward_fn)
        elapsed = time.time() - t0

        scores = [int(payoff[:, j].sum()) for j in range(rb_cfg.n_grain)]
        step_bests = [[j, scores[j], pool[j]] for j in range(rb_cfg.n_grain) if scores[j] / len(ref_pop) > 0.7]
        bests.extend(step_bests)

        logger.log(
            {
                "iter": i,
                "t": time.time(),
                "n_tested_so_far": (i + 1) * rb_cfg.n_grain,
                "payoff_s": round(elapsed, 4),
                "score_mean": round(sum(scores) / len(scores), 4),
                "score_min": min(scores),
                "score_max": max(scores),
                "n_bests_step": len(step_bests),
                "n_bests_total": len(bests),
            },
            work_pop=pool,
            ref_pop=ref_pop,
        )

        for entry in step_bests:
            with open(Path(rb_cfg.out_path) / "bests.jsonl", "a") as f:
                json.dump(entry, f)
                f.write("\n")


if __name__ == "__main__":

    main(RandomBaselineConfig())