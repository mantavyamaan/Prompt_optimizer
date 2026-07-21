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


def minimum_detectable_effect(
    n: int,
    sample_variance: float,
    alpha: float = ALPHA,
    power: float = 0.8,
) -> float:
    """Minimum detectable effect using normal approximation.

    This is a transparent diagnostic; it is not used as a promotion criterion.
    """
    if n <= 0:
        return float("inf")
    # Z-scores derived from alpha and power (not hardcoded).
    z_alpha = _z_score(1 - alpha / 2)
    z_power = _z_score(power)
    return (z_alpha + z_power) * math.sqrt(2 * max(sample_variance, 1e-9) / n)


def _z_score(p: float) -> float:
    """Rational approximation of the inverse normal CDF (Abramowitz & Stegun 26.2.17)."""
    p = max(1e-9, min(1 - 1e-9, p))
    t = math.sqrt(-2 * math.log(min(p, 1 - p)))
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    approx = t - (c[0] + c[1] * t + c[2] * t ** 2) / (1 + d[0] * t + d[1] * t ** 2 + d[2] * t ** 3)
    return approx if p >= 0.5 else -approx


def paired_deltas(champion: dict[str, float], challenger: dict[str, float]) -> tuple[list[float], int, int, int]:
    shared = sorted(set(champion) & set(challenger))
    deltas = [challenger[case] - champion[case] for case in shared]
    wins = sum(delta > 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    return deltas, wins, losses, len(deltas) - wins - losses
