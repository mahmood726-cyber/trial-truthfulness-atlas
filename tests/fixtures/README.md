# Fixtures

`cardio_5_trials.json` — five hand-curated cardiology trials with known
ground truth across all five integrity flags. Used by
`tests/test_integration_5trial.py` to gate v0.1.0 byte-reproducibility.

Trials 1–3 (PARADIGM-HF, VICTORIA, GRIPHON) are real and drawn from
DossierGap v0.3.0. Their NCT, PMID, N, and outcome values were verified
manually against published Cochrane reviews and CT.gov on 2026-04-29.

Trial 4 (FIXTURE-N) is synthetic — flags an N-drift case with
deterministic values.

Trial 5 (Unknown Author 1999) is synthetic — exercises the unbridgeable
+ all-unscoreable path.

If you change this fixture, regenerate `baseline/cardio_v0.1.0.json` and
update the integration test's pinned `atlas.csv` snapshot.
