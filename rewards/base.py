from dataclasses import dataclass
from typing import Callable

import numpy as np

from config import ExperimentConfig
from core.matchups import Matchups
from rewards.blind_reward import reward as blind_reward
from rewards.placeholder_reward import reward as placeholder_reward
from rewards.quine_pressure_reward import reward as quine_pressure_reward

# A reward scores a whole batch of matchups at once, one value per matchup;
# core/matchups.py documents the operations it can use. Written that way, a
# reward runs on every payoff backend without backend-specific code.
RewardFn = Callable[[Matchups], np.ndarray]


@dataclass(frozen=True)
class RewardSpec:
    name: str
    fn: RewardFn
    # True => reward(b, a) == -reward(a, b); lets payoff extension and
    # tournament blocks derive the reverse matchup as -A.T instead of re-running it.
    zero_sum: bool


REWARDS: dict[str, RewardSpec] = {
    "blind": RewardSpec("blind", blind_reward, zero_sum=True),
    "placeholder": RewardSpec("placeholder", placeholder_reward, zero_sum=True),
    "quine_pressure": RewardSpec("quine_pressure", quine_pressure_reward, zero_sum=True),
}


def build_reward(cfg: ExperimentConfig) -> RewardSpec:
    return REWARDS[cfg.reward]
