
import json
from pathlib import Path
from typing import List

from config import RandomBaselineConfig
from utils.pop_init import random_code, random_population
from rewards.payoff import compute_payoff_matrix
from rewards.wrapper import make_reward


def main(rb_cfg: RandomBaselineConfig) -> None:
    """
    Random baseline experiment using the interpreter and reward from cfg.
    """
    cfg = rb_cfg.experiment

    # Build reference population
    ref_pop = random_population(
        rb_cfg.n_ref,
        cfg.code.code_length,
        cfg.code.min_val,
        cfg.code.max_val,
    )

    # Prepare output file
    out_dir = Path(rb_cfg.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)


    with open(rb_cfg.out_path + "/" + "ref_pop.json", "a") as f:
        for ind in ref_pop:
            json.dump(ind, f)
            f.write("\n")

    reward_fn = make_reward(cfg)

    out_file = rb_cfg.out_path + "/" + rb_cfg.out_name

    with open(out_file, "a") as f:
        for i in range(rb_cfg.n_tested):
            print("Up to:", i*rb_cfg.n_grain, end="\r") 
            pool = random_population(
                rb_cfg.n_grain,
                cfg.code.code_length,
                cfg.code.min_val,
                cfg.code.max_val,
            )

            payoff = compute_payoff_matrix(
                cfg=cfg,
                ref=ref_pop,
                pop=pool,
                reward_fn=reward_fn,
            )

            # Aggregate score: sum over rows (or columns, depending on convention)
            scores = [int(payoff[:, i].sum()) for i in range(rb_cfg.n_grain)]
            for idx, score in enumerate(scores):
                json.dump(score, f)
                f.write("\n")
                if score/len(ref_pop)>0.7:
                    with open(rb_cfg.out_path + "/" + "bests.json") as f:
                        json.dump([idx, score, pool[idx]])
                        f.write("\n")


if __name__ == "__main__":

    main(RandomBaselineConfig())