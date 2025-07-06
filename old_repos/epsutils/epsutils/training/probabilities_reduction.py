import statistics
from enum import Enum

import torch


class ProbabilitiesReductionStrategy(Enum):
    MAX = 1
    MEAN = 2
    MEDIAN = 3


def probabilities_reduction(probs, strategy):
    is_tensor = isinstance(probs, torch.Tensor)
    probs_list = probs.tolist() if is_tensor else probs

    if strategy == ProbabilitiesReductionStrategy.MAX:
        res = max(probs_list)
    elif strategy == ProbabilitiesReductionStrategy.MEAN:
        res = statistics.mean(probs_list)
    elif strategy == ProbabilitiesReductionStrategy.MEDIAN:
        res = statistics.median(probs_list)
    else:
        raise ValueError(f"Unsupported probabilities reduction strategy {strategy}")

    return torch.tensor(res, dtype=probs.dtype) if is_tensor else res
