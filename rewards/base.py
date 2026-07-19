from dataclasses import dataclass
from typing import Callable

from config import ExperimentConfig
from core.types import Interpreter, Program
from rewards.blind_reward import reward as blind_reward
from rewards.placeholder_reward import reward as placeholder_reward
from rewards.quine_pressure_reward import reward as quine_pressure_reward

RewardFn = Callable[[Interpreter, Program, Program], float]


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
