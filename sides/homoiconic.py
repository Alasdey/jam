# sides/duel_round_robin.py

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

from config import ExperimentConfig
from interpreters.wrapper import make_interpreter
from rewards.wrapper import make_reward
from rewards.payoff import compute_payoff_matrix


def load_population(path: str) -> List[List[int]]:
    with open(path, "r") as f:
        pop = json.load(f)
    return [[int(x) for x in ind] for ind in pop]


def score_candidate(cfg: ExperimentConfig, pop: List[List[int]], candidate: List[int], reward_fn) -> int:
    payoff = compute_payoff_matrix(cfg=cfg, ref=pop, pop=[candidate], reward_fn=reward_fn)
    return int(payoff[:, 0].sum())

def rounds(n):
    for i in range(n):
        for j in range(n):
            yield j, (i+j)%n 

def main(pop_path: str, out_path: str) -> None:
    cfg = ExperimentConfig()
    interpreter = make_interpreter(cfg)
    reward_fn = make_reward(cfg)

    pop = load_population(pop_path)
    n = len(pop)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("a") as f:
        for (i, j) in rounds(n):
            print(i, j, end="\r")
            code = pop[i]
            input_data = pop[j]

            out_ij, mem_ij, status_ij = interpreter.run(code, input_data)

            candidates: List[Tuple[str, List[int]]] = [
                ("out", out_ij),
                ("mem", mem_ij),
            ]

            for label, cand in candidates:
                t0 = time.time()
                score = score_candidate(cfg, pop, cand, reward_fn)
                dt = time.time() - t0

                record: Dict[str, Any] = {
                    "i": i,
                    "j": j,
                    "label": label,
                    "candidate_len": len(cand),
                    "status": status_ij,
                    "score": score,
                    "time_s": dt,
                }
                json.dump(record, f)
                f.write("\n")
                f.flush()

            # print(f"completed i={i}/{n}", end="\r")

    print("\nDone.")


if __name__ == "__main__":
    pop_path = "outputs/random_skimmed/20251209_161419/pop.json"
    out_path = "outputs/homoiconic/" + time.strftime("%Y%m%d_%H%M%S")
    main(pop_path, out_path)
