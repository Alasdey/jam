
import json
from pathlib import Path

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

    out_file = rb_cfg.out_path + "/" + rb_cfg.out_name

    reward_fn = make_reward(cfg)

    with open(out_file, "a") as f:
        for _ in range(rb_cfg.n_tested):
            indiv = random_code(
                cfg.code.code_length,
                cfg.code.min_val,
                cfg.code.max_val,
            )

            payoff = compute_payoff_matrix(
                cfg=cfg,
                ref=ref_pop,
                pop=[indiv],
                reward_fn=reward_fn,
            )

            # Aggregate score: sum over rows (or columns, depending on convention)
            score = int(payoff[:, 0].sum())
            json.dump(score, f)
            f.write("\n")


if __name__ == "__main__":

    main(RandomBaselineConfig())