from __future__ import annotations

import math
from statistics import fmean, variance

ALPHA = 0.05


def confidence_sequence(deltas: list[float], alpha: float = ALPHA) -> tuple[bool, float, float]:
    """Normal-mixture confidence sequence for paired deltas in [-1,1].

    The mixture boundary supports repeated looks. It is intentionally paired,
    so per-case correlations are retained instead of being discarded in an
    aggregate win rate.
    """
    n = len(deltas)
    if n < 5:
        return False, -1.0, 1.0
    mean = fmean(deltas)
    sample_var = max(variance(deltas) if n > 1 else 1.0, 1e-6)
    tau_squared = 0.01
    mixture_variance = sample_var + n * tau_squared
    log_term = math.log(mixture_variance / (alpha ** 2 * sample_var))
    radius = math.sqrt(max(0.0, mixture_variance * log_term / (n ** 2)))
    low, high = max(-1.0, mean - radius), min(1.0, mean + radius)
    return low > 0 or high < 0, low, high


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(max(wins, losses), n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def minimum_detectable_effect(n: int, sample_variance: float, alpha: float = ALPHA, power: float = 0.8) -> float:
    if n <= 0:
        return float("inf")
    # Normal approximation; transparent diagnostic, not used as a promotion criterion.
    return (1.96 + 0.84) * math.sqrt(2 * max(sample_variance, 1e-9) / n)


def paired_deltas(champion: dict[str, float], challenger: dict[str, float]) -> tuple[list[float], int, int, int]:
    shared = sorted(set(champion) & set(challenger))
    deltas = [challenger[case] - champion[case] for case in shared]
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    return deltas, wins, losses, len(deltas) - wins - losses
