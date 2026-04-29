"""Tests for meta.py — random-effects pooling for v0.2.0 crosses_null."""

from __future__ import annotations

import math

import pytest

from tta import meta


def test_t_critical_known_values():
    # Spot-check against R qt(0.975, df)
    assert meta.t_critical_975(1) == 12.706
    assert meta.t_critical_975(9) == 2.262
    assert meta.t_critical_975(29) == 2.045
    # df > 30 → asymptotic z
    assert meta.t_critical_975(50) == 1.96
    assert meta.t_critical_975(100) == 1.96


def test_dl_tau2_zero_when_homogeneous():
    """When trial effects all equal mu_FE, Q = 0 → tau2 = 0."""
    # Three trials, identical effect, same variance
    tau2 = meta.dl_tau2([0.1, 0.1, 0.1], [0.01, 0.01, 0.01])
    assert tau2 == pytest.approx(0.0, abs=1e-12)


def test_dl_tau2_truncates_negative():
    """When Q < df, the formula yields negative tau2; we truncate to 0."""
    # Trials with smaller-than-FE variance (Q small)
    tau2 = meta.dl_tau2([0.0, 0.001, -0.001], [0.5, 0.5, 0.5])
    assert tau2 == 0.0


def test_dl_tau2_positive_when_heterogeneous():
    """Heterogeneous effects → positive tau2."""
    # Three studies with quite different effects
    tau2 = meta.dl_tau2([-0.5, 0.0, 0.5], [0.01, 0.01, 0.01])
    assert tau2 > 0.0


def test_dl_tau2_zero_when_k_lt_2():
    assert meta.dl_tau2([], []) == 0.0
    assert meta.dl_tau2([0.5], [0.01]) == 0.0


def test_pool_no_trials():
    r = meta.random_effects_pool([], [])
    assert r.k == 0
    assert r.method == "no_trials"
    assert r.mu is None
    assert r.crosses_null is None


def test_pool_single_trial_passthrough():
    """k=1: pooled effect IS the trial effect; CI from trial SE."""
    r = meta.random_effects_pool([-0.2], [0.01])  # SE = 0.1
    assert r.k == 1
    assert r.method == "single_trial_passthrough"
    assert r.mu == pytest.approx(-0.2)
    # CI = -0.2 ± 1.96 * 0.1 = (-0.396, -0.004)
    assert r.ci_low == pytest.approx(-0.396, abs=1e-3)
    assert r.ci_high == pytest.approx(-0.004, abs=1e-3)
    assert r.crosses_null is False


def test_pool_two_homogeneous_trials():
    """Two trials with same effect: pooled effect equals shared effect.
    Tau2 = 0; HKSJ floor activates (Q=0 < k-1=1)."""
    r = meta.random_effects_pool([-0.2, -0.2], [0.01, 0.01])
    assert r.k == 2
    assert r.method == "random_effects_dl_hksj"
    assert r.mu == pytest.approx(-0.2, abs=1e-9)
    assert r.tau2 == 0.0
    # Q = 0 → q* = max(1, 0/1) = 1; var_mu = 1/(1/0.01 + 1/0.01) = 0.005
    # SE_HKSJ = sqrt(1 * 0.005) = 0.0707
    # CI = -0.2 ± t_{0.975, 1} * 0.0707 = -0.2 ± 12.706 * 0.0707 ≈ ±0.898
    assert r.ci_low == pytest.approx(-0.2 - 12.706 * math.sqrt(0.005), abs=1e-3)
    assert r.ci_high == pytest.approx(-0.2 + 12.706 * math.sqrt(0.005), abs=1e-3)
    # CI crosses null (huge t-crit at df=1 makes interval wide)
    assert r.crosses_null is True


def test_pool_three_decisive_trials():
    """Three trials with consistent moderate-effect: pooled CI does not cross 0."""
    # log-HRs around -0.2 with tight CIs
    r = meta.random_effects_pool([-0.20, -0.18, -0.22], [0.005, 0.005, 0.005])
    assert r.k == 3
    assert r.method == "random_effects_dl_hksj"
    assert r.mu == pytest.approx(-0.20, abs=1e-2)
    # CI should be well below 0 (decisive favourable effect)
    assert r.ci_high < 0
    assert r.crosses_null is False


def test_pool_three_null_straddling_trials():
    """Three trials whose effects bracket null: pooled CI crosses 0."""
    r = meta.random_effects_pool([-0.10, 0.0, 0.10], [0.05, 0.05, 0.05])
    assert r.k == 3
    assert r.mu == pytest.approx(0.0, abs=1e-9)
    assert r.crosses_null is True


def test_pool_result_dataclass_is_frozen():
    """PoolResult is frozen so MA-level state can't be mutated downstream."""
    r = meta.random_effects_pool([-0.1, -0.2], [0.01, 0.01])
    with pytest.raises(Exception):
        r.mu = 999.0  # type: ignore[misc]


def test_variance_from_ci_recovers_se():
    """CI half-width / 1.96 = SE; SE² = variance. Spot-check round-trip."""
    # log(HR=0.80) = -0.223; log(0.73) = -0.314; log(0.87) = -0.139
    # half-width = (−0.139 − (−0.314)) / 2 = 0.0875
    # SE = 0.0875 / 1.96 ≈ 0.0446; var ≈ 0.001995
    v = meta.variance_from_ci(-0.223, -0.314, -0.139)
    assert v == pytest.approx((0.0875 / 1.96) ** 2, abs=1e-6)


def test_variance_from_ci_returns_none_on_degenerate():
    assert meta.variance_from_ci(0.0, 0.0, 0.0) is None
    assert meta.variance_from_ci(0.0, 0.5, -0.5) is None  # inverted bounds → half<=0
    assert meta.variance_from_ci(float("nan"), -0.1, 0.1) is None
