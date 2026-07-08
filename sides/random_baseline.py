
import json
import time
from pathlib import Path

from config import RandomBaselineConfig
from creation import build_creator
from loggers import ExperimentLogger
from rewards.base import build_reward
from rewards.payoff import PayoffEngine


def main(rb_cfg: RandomBaselineConfig) -> None:
    cfg = rb_cfg.experiment
    creator = build_creator(cfg)

    ref_pop = [creator.random() for _ in range(rb_cfg.n_ref)]

    logger = ExperimentLogger(
        rb_cfg.out_path, rb_cfg,
        n_ref=rb_cfg.n_ref, n_tested=rb_cfg.n_tested, n_grain=rb_cfg.n_grain,
    )
    with open(Path(rb_cfg.out_path) / "ref_pop.jsonl", "w") as f:
        for genome in ref_pop:
            json.dump(genome, f)
            f.write("\n")

    bests = []
    with PayoffEngine(cfg, build_reward(cfg)) as engine:
        for i in range(rb_cfg.n_tested):
            print("Up to:", i * rb_cfg.n_grain, end="\r")
            pool = [creator.random() for _ in range(rb_cfg.n_grain)]

            t0 = time.time()
            payoff = engine.matrix(ref_pop, pool)
            elapsed = time.time() - t0

            scores = [int(payoff[:, j].sum()) for j in range(rb_cfg.n_grain)]
            step_bests = [
                [j, scores[j], pool[j]]
                for j in range(rb_cfg.n_grain)
                if scores[j] / len(ref_pop) > 0.7
            ]
            bests.extend(step_bests)

            logger.log_metrics(
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
                }
            )

            for entry in step_bests:
                with open(Path(rb_cfg.out_path) / "bests.jsonl", "a") as f:
                    json.dump(entry, f)
                    f.write("\n")


if __name__ == "__main__":
    main(RandomBaselineConfig())
