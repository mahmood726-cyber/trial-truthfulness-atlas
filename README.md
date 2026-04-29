<!-- sentinel:skip-file — paths in this README are user-facing setup documentation, not shipped code -->
# Trial Truthfulness Atlas (TTA) — v0.1.0

Fifth Pairwise70 atlas. Asks the **upstream** question the prior four
atlases skip: are the trials feeding a Cochrane meta-analysis coherent
between their CT.gov registration and the number Cochrane pooled?

For every cardiology Cochrane MA in scope, TTA computes five integrity
flags per trial:

| Flag | Question |
|------|----------|
| 0 — Bridge | Can we resolve `Study (author year)` to an `nct_id`? |
| 1 — Outcome drift | Did the primary outcome label change? (LLM-judged) |
| 2 — N drift | Did the analysed-N differ > 10 % from registered enrolment? |
| 3 — Direction concordance | Does AACT effect sign match MA effect sign? |
| 4 — Results-posting compliance | Was FDAAA-required CT.gov results posting actually done? |

**Outputs:** `outputs/atlas.csv`, `outputs/ma_rollup.csv`,
`outputs/dashboard.html`, `outputs/verification.html`.

**Quick start:**

```bash
pip install -e ".[dev]"
python -m tta.cli preflight              # verify external prereqs
python -m tta.cli build --fixture-mode   # v0.1.0 fixture pipeline
pytest -v                                # all green
```

**Inputs:**
- AACT snapshot at `D:\AACT-storage\AACT\2026-04-12\` (12 GB, 38 tables)
- Pairwise70 at `C:\Projects\Pairwise70\data\` (595 .rda files)
- Local `gemma2:9b` via `ollama serve`
- DossierGap fixture at `C:\Projects\dossiergap\outputs\dossier_trials.v0.3.0.csv`

**Spec:** `docs/superpowers/specs/2026-04-29-trial-truthfulness-atlas-design.md`
**Plan:** `docs/superpowers/plans/2026-04-29-trial-truthfulness-atlas-v0.1.0.md`

**v0.2.0 roadmap:** widen to all ~6,386 Pairwise70 MAs, add bridge methods 2–4, real-AACT integration test, Crossref DOI.

License: MIT.
