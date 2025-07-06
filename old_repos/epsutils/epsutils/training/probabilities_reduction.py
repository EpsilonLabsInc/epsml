import statistics
from enum import Enum

import torch


class ProbabilitiesReductionStrategy(Enum):
    MAX = 1
    MEAN = 2
    MEDIAN = 3


def probabilities_reduction(probs, strategy):
    if isinstance(probs, torch.Tensor):
        probs = probs.tolist()

    if strategy == ProbabilitiesReductionStrategy.MAX:
        return max(probs)
    elif strategy == ProbabilitiesReductionStrategy.MEAN:
        return statistics.mean(probs)
    elif strategy == ProbabilitiesReductionStrategy.MEDIAN:
        return statistics.median(probs)
    else:
        raise ValueError(f"Unsupported probabilities reduction strategy {strategy}")
