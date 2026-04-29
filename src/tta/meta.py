"""Random-effects meta-analysis pooling for atlas MA-rollup.

v0.2.0 introduces real per-MA pooled estimates so `crosses_null` reflects
the pooled effect's CI, not the first trial's CI. Implementation:

  - DerSimonian-Laird estimator for τ² (closed-form, no scipy dependency).
    Per lessons.md, DL is biased for k<10 — but for the v0.1.x fixtures
    (k=2-3 per MA) any iterative estimator is sample-size limited; DL is
    the honest choice given the data.
  - Hartung-Knapp-Sidik-Jonkman (HKSJ) variance correction with the
    `max(1, Q/(k-1))` floor (lessons.md HKSJ rule).
  - t-distribution critical values at df=k-1 (lessons.md HKSJ df rule).
  - For k=1 (single trial), pooled effect == trial effect; CI returned
    is the trial's own CI (degenerate but defined).
  - For k=0, returns None.

All inputs are on the LOG scale (logHR/logOR/logRR/SMD); outputs are on
the same scale. Caller back-transforms if needed for display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


# Two-sided t critical values at α=0.05 (i.e. t_{0.975, df}). Hard-coded
# for df 1..30 to avoid a scipy dependency. For df>30, asymptotic z=1.96
# is within 5% of the true t value and is the standard convention.
_T_CRIT_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
_Z_975 = 1.96


def t_critical_975(df: int) -> float:
    """Two-sided t critical at α=0.05 for given df. df>30 → asymptotic z."""
    if df <= 0:
        # Degenerate — caller should have rejected this case; return a
        # very wide value that effectively gives an uninformative CI.
        return float("inf")
    return _T_CRIT_975.get(df, _Z_975)


@dataclass(frozen=True)
class PoolResult:
    """Pooled random-effects estimate for one meta-analysis.

    `mu` and CI bounds are on the same scale as the input effects (LOG
    scale by atlas convention). `tau2` is the DL between-study variance
    estimate. `k` is the number of trials pooled. `method` indicates
    which pooling path was taken (RE-DL+HKSJ, single-trial-passthrough,
    or insufficient).
    """
    mu: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    tau2: Optional[float]
    k: int
    method: str

    @property
    def crosses_null(self) -> Optional[bool]:
        if self.ci_low is None or self.ci_high is None:
            return None
        return self.ci_low <= 0.0 <= self.ci_high


def dl_tau2(effects: Sequence[float], variances: Sequence[float]) -> float:
    """DerSimonian-Laird between-study variance estimate.

    Returns 0.0 if estimate would be negative (truncation) or if k<2.
    """
    y = list(effects)
    v = list(variances)
    k = len(y)
    if k < 2:
        return 0.0
    # Fixed-effects weights
    w = [1.0 / vi for vi in v]
    sw = sum(w)
    if sw <= 0:
        return 0.0
    mu_fe = sum(wi * yi for wi, yi in zip(w, y)) / sw
    Q = sum(wi * (yi - mu_fe) ** 2 for wi, yi in zip(w, y))
    df = k - 1
    sw2 = sum(wi * wi for wi in w)
    C = sw - sw2 / sw
    if C <= 0:
        return 0.0
    return max(0.0, (Q - df) / C)


def random_effects_pool(
    effects: Sequence[float],
    variances: Sequence[float],
) -> PoolResult:
    """Random-effects pool with DL τ² + HKSJ-adjusted CI.

    Inputs aligned 1:1; both on the LOG scale. Returns a PoolResult with
    `method` describing the path taken.
    """
    y = [float(e) for e in effects]
    v = [float(vi) for vi in variances]
    k = len(y)

    if k == 0:
        return PoolResult(None, None, None, None, 0, "no_trials")

    if k == 1:
        # Single trial — pooled effect IS the trial effect. CI is the
        # trial's own ±1.96·SE (no τ² possible with one trial).
        mu = y[0]
        se = math.sqrt(v[0]) if v[0] > 0 else None
        if se is None:
            return PoolResult(mu, None, None, 0.0, 1, "single_trial_no_se")
        return PoolResult(
            mu=mu,
            ci_low=mu - _Z_975 * se,
            ci_high=mu + _Z_975 * se,
            tau2=0.0,
            k=1,
            method="single_trial_passthrough",
        )

    tau2 = dl_tau2(y, v)
    # Random-effects weights
    w_re = [1.0 / (vi + tau2) for vi in v]
    sw_re = sum(w_re)
    if sw_re <= 0:
        return PoolResult(None, None, None, tau2, k, "degenerate_weights")

    mu = sum(wi * yi for wi, yi in zip(w_re, y)) / sw_re
    var_mu = 1.0 / sw_re

    # HKSJ adjustment with the lessons.md floor:
    # If Q < k-1, HKSJ narrows CI below DL — set q* floor to 1.0 so we
    # never produce a tighter-than-DL interval.
    Q_re = sum(wi * (yi - mu) ** 2 for wi, yi in zip(w_re, y))
    df = k - 1
    q_star = max(1.0, Q_re / df) if df > 0 else 1.0
    se_hksj = math.sqrt(q_star * var_mu)

    t_crit = t_critical_975(df)
    return PoolResult(
        mu=mu,
        ci_low=mu - t_crit * se_hksj,
        ci_high=mu + t_crit * se_hksj,
        tau2=tau2,
        k=k,
        method="random_effects_dl_hksj",
    )


def variance_from_ci(effect: float, ci_low: float, ci_high: float) -> Optional[float]:
    """Recover within-study variance from an effect + 95 % CI bounds.

    Assumes the CI was computed as effect ± 1.96·SE (the Pairwise70
    convention). Returns None if the bounds are non-finite or zero-width.
    """
    if not (math.isfinite(effect) and math.isfinite(ci_low) and math.isfinite(ci_high)):
        return None
    half_width = (ci_high - ci_low) / 2.0
    if half_width <= 0:
        return None
    se = half_width / _Z_975
    return se * se
