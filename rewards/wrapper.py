# rewards/reward.py
from typing import List

from rewards.blind_reward import reward as blind_reward
from rewards.placeholder_reward import reward as placeholder_reward
from config import ExperimentConfig


def make_reward(cfg: ExperimentConfig):
    """
    Top-level reward factory.
    Returns a reward function with signature: reward(interp, code_a, code_b) -> int
    """
    if cfg.reward == "blind":
        return blind_reward
    elif cfg.reward == "placeholder":
        return placeholder_reward
    
    raise ValueError(f"Unknown reward type: {cfg.reward}")