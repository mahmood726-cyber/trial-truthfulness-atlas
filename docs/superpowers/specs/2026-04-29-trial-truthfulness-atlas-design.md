# Trial Truthfulness Atlas — v0.1.0 Design

**Spec date:** 2026-04-29
**Status:** Approved (brainstorm), pre-plan
**Author:** mahmood726-cyber
**Repo:** `C:\Projects\trial-truthfulness-atlas\`
**Submission target:** Synthēsis Methods Note + E156 micro-paper companion
**Atlas-series number:** 5th Pairwise70 atlas (after repro-floor, cochrane-modern-re, PI-atlas, responder-floor)

---

## 1. Problem statement

The four existing Pairwise70 atlases (repro-floor, cochrane-modern-re, PI-atlas, responder-floor) all ask **"is the meta-analysis reproducible?"**. None ask the upstream question:

> Are the **trials feeding** the meta-analysis coherent between their CT.gov registration and the number Cochrane pooled?

Known base rates from the literature already in our memory:
- ~24 % silent primary-outcome dropping in ITS (medRxiv 2025.11.06)
- 52 % protocol-vs-paper outcome-set discrepancy
- ~63.6 % registry-to-publication linkage (TrialScout, medRxiv 2026.03.15)
- 36 % retrospective registration (California audit)

These are population-level. **Nobody has measured them conditional on being inside a Cochrane MA** — i.e., conditional on being the trials that drive guideline recommendations. v0.1.0 produces that measurement for the cardiology subset.

## 2. Why this is uniquely shippable from this machine

- `D:\AACT-storage\AACT\2026-04-12\` — full 12 GB CT.gov snapshot, 38 tables, 17 days old (no API rate limit on 50K joins)
- `D:\ollama\models\` — local LLM (`gemma2:9b`, no API cost on tens of thousands of judgments)
- `C:\Projects\Pairwise70\data\` — 595 `.rda` files, ~6,386 MAs already extracted
- `C:\Projects\dossiergap\outputs\dossier_trials.v0.3.0.csv` — known-good cardiology PMID×NCT fixture rows
- Sentinel + Overmind already enforce the integrity discipline this paper is *about*

No external dependency needs to come online for v0.1.0 to ship.

## 3. Scope (in / out for v0.1.0)

**In:**
- Cardiology Cochrane MAs (Cochrane Heart Group + cardiovascular reviews)
- Five integrity flags (Flag 0 NCT-bridge + Flags 1–4 truthfulness; see §5)
- Outputs: `atlas.csv`, `ma_rollup.csv`, `dashboard.html`, `verification.html`
- Local-only judging via `ollama gemma2:9b` (no remote API)
- Workbook entry + README + FORKING.md + Sentinel pre-push hook

**Out (deferred to ≥v0.2.0):**
- Non-cardiology MAs (full 6,386-MA sweep)
- Causal claims about *why* trials drift
- Correction of pooled MA effects
- Retraction-Watch integration
- Prospective ITS workflow
- Author attribution / questionable-research-practice scoring

## 4. Architecture

```
trial-truthfulness-atlas/
  data/                          # gitignored; 1–2 GB working copies
    aact_cardio.parquet          # cardio NCTs from D:\AACT\…
    pairwise70_cardio.parquet    # MA × Study × outcome rows
    nct_bridge.parquet           # Study/PMID → NCT mapping (Flag 0 output)
    judge_cache/                 # sha256(prompt+input).json
  src/tta/
    __init__.py
    ingest.py                    # AACT TSV → parquet; Pairwise70 .rda → parquet
    bridge.py                    # Flag 0 — Study/PMID → NCT
    flags/
      outcome_drift.py           # Flag 1 — LLM-judged
      n_drift.py                 # Flag 2 — numeric
      direction_concordance.py   # Flag 3 — numeric
      publication_status.py      # Flag 4 — set membership
    judge/
      ollama_client.py           # local LLM wrapper, frozen prompts
      prompts.py                 # versioned + sha256-hashed
    aggregate.py                 # MA-level rollup
    cli.py                       # `tta build|sweep|verify-one <NCT>|preflight`
  outputs/
    atlas.csv                    # NCT × flag0..4 × MA_id
    ma_rollup.csv                # MA × flag-density × pooled-effect × crosses-null
    dashboard.html               # filterable (ARAC pattern, inline-SVG charts)
    verification.html            # one-trial-at-a-time UI for human spot-check
  tests/
    fixtures/                    # 5 cardio trials with hand-curated ground truth
    test_*.py
  baseline/
    cardio_v0.1.0.json           # numerical baseline for Sentinel/Overmind
  docs/
    superpowers/specs/2026-04-29-trial-truthfulness-atlas-design.md
    methods.md
  E156-PROTOCOL.md
  README.md
  FORKING.md
  pyproject.toml
  requirements.txt
  .gitignore
  push.sh
```

## 5. The five integrity flags (formal definitions)

### Flag 0 — NCT bridge (preliminary; enables Flags 1–4)

**Input:** Pairwise70 row with `Study` (author-year) + `review_doi` (Cochrane review)
**Output:** `{nct_id: str, bridge_method: enum, bridge_confidence: float}` or `unbridgeable`

**Methods, in waterfall order:**
1. Direct match in DossierGap output (highest confidence)
2. PubMed E-utilities → PMID → AACT `id_information.id_value` cross-ref where `id_type ∈ {pmid, secondary}`
3. Cochrane review's published "Characteristics of included studies" table (HTML scrape via `review_url`) → trial NCT(s)
4. Heuristic match: AACT `official_title` contains the Pairwise70 study's first-author surname + year ±1 within the MA's condition

**Headline finding from Flag 0 alone:** "X % of trials inside Cochrane cardiology MAs cannot be resolved to a CT.gov registration." Itself a publishable observation.

### Flag 1 — Outcome drift (LLM-judged)

**Input:** AACT `design_outcomes` rows (registered) for the NCT + the outcome label Cochrane extracted into Pairwise70
**Method:** `gemma2:9b` semantic comparison via frozen prompt v1
**Output:** `{identical, refinement, substantively_different, unscoreable}`
**Reproducibility:** prompt sha256 + model version + seed=42 logged per row; cache keyed by `sha256(prompt + inputs + model_version)`

### Flag 2 — N drift (numeric)

**Input:** AACT `calculated_values.actual_enrollment` vs Pairwise70 `Experimental.N + Control.N`
**Threshold:** absolute relative difference > 10 % → flagged. Threshold value is calibrated on fixtures and recorded in baseline JSON.
**Output:** `{flagged, not_flagged, unscoreable}`
**Pre-processing:** scrub negation patterns (`not randomized`, `non-randomized`, `withdrawn before randomization`) per Verquvo lesson before extracting numerics from any free-text fallback.

### Flag 3 — Direction concordance (numeric)

**Input:** AACT `outcome_analyses` (where `param_type ∈ {Hazard Ratio, Odds Ratio, Risk Ratio, Mean Difference}`) sign vs Pairwise70 `Mean` sign for the same effect direction
**Method:** sign-compare with epsilon for near-zero (`|effect| < 0.01` → unscoreable)
**Output:** `{concordant, flipped, unscoreable}`

### Flag 4 — Results-posting compliance (CT.gov, FDAAA-relevant)

**Note on framing:** the original "publication status" framing collapses inside the cardio-MA subset because every included trial is by definition published-or-extractable by Cochrane. The discriminating integrity gap inside this subset is whether the **registration** also has results posted on CT.gov as FDAAA requires.

**Input:** AACT `studies.results_first_posted_date`, `studies.completion_date`, `studies.study_type`, `studies.phase`, `interventions.intervention_type`, US-site presence (FDAAA scope)
**Method:** Compute FDAAA applicability (interventional drug/device trial, completed > 12 mo before snapshot, US site or US-FDA-regulated). Then check posting status.
**Output:** `{posted, required_not_posted, not_required, unscoreable}`
**Why not the original framing:** ~ 0 % expected rate of unpublished-but-in-MA. Pivoted to results-posting compliance which is a known ~ 50 % violation rate in the literature and clinically meaningful.

## 6. Data flow

```
D:/AACT/2026-04-12/*.txt  ── ingest.py ──┐
                                          │
                             ── bridge.py ──→ nct_bridge.parquet  (Flag 0)
                                          │
C:/Projects/Pairwise70/data/*.rda ────────┘
            │
            ▼
   pairwise70_cardio.parquet
            │
            ▼
   ┌─── flags/{1,2,3,4}.py ───┐
   │  judge cache (sha256-keyed, idempotent)
   └────────────┬─────────────┘
                ▼
       atlas.csv ──→ aggregate.py ──→ ma_rollup.csv
                                    ──→ dashboard.html
                                    ──→ verification.html
```

## 7. Error handling (drawn from `lessons.md`)

| Risk | Source lesson | Mitigation |
|------|---------------|------------|
| Missing AACT field | substitution-on-missing-required (2026-04-28) | Tier as `unscoreable`, never impute |
| Negated counts | Verquvo (2026-04-15) | Scrub `not/non/never randomized N` before numeric extract |
| Empty DataFrame | Sentinel P1-empty-dataframe-access | Guard before `.iloc/.iat`; honor `if df.empty` early-return |
| Stale AACT snapshot | Confident-tone tool failures (2026-04-28) | Fail closed if snapshot age > 90 days; today: 17 d, OK |
| Ollama unavailable | Same | Flag 1 → `unscoreable`; Flags 0,2,3,4 still run |
| LLM citation misattribution (~4 %) | LLM citation baseline (2026-04-28) | Cap LLM context to ≤ 10 inputs per judgment; DOI-resolve any LLM-emitted refs |
| Cache hash drift | Confident-tone tool failures | Re-judge on hash mismatch and log diff |
| Idempotency violation | Idempotent edits (2026-04-28) | Pinned prompt sha256 + model version per row; rerun must match |
| Long-context safety collapse > 100 K tokens | (2026-04-28) | Per-judgment context bounded; never accumulate |
| Hardcoded local paths in shipped code | Lessons | All paths via config / CLI args; only `data/` and `outputs/` are repo-relative |

## 8. Testing strategy

- **Fixtures:** 5 hand-curated cardiology trials with known ground-truth values for all five flags. At least one drawn from DossierGap PARADIGM-HF / VICTORIA / GRIPHON known-positive set.
- **Per-flag unit tests:** ≥ 3 cases each, including the unscoreable tier
- **Integration test:** full pipeline on the 5-trial fixture must reproduce a pinned `atlas.csv` byte-for-byte
- **Numerical baseline:** per-flag fraction with REML-pooled CI on fixture; pinned in `baseline/cardio_v0.1.0.json` per the numerical-baseline-contract rule
- **Negation-regex test:** Flag 2 must NOT extract from `Not Randomized 1,807` (Verquvo regression)
- **Empty-DataFrame guard test:** atlas builder on empty cardio subset returns empty CSV without IndexError
- **Idempotency test:** rerun with identical inputs must match (cache + pinned prompt-hash)
- **Dual-LLM screening over-includes (2026-04-28):** Flag 1 LLM judgments expected to be conservative-biased; budget ≥5× human spot-check capacity for the `substantively_different` tier
- **Preflight test (Task 0 of plan):** verify ollama responsive + AACT path + Pairwise70 path + DossierGap fixture present; fail closed with action list if any missing
- **Sentinel pre-push:** install hook; require 0 BLOCK before push
- **Bounded loop:** cap retries per failure; log blockers to `STUCK_FAILURES.md`

## 9. Outputs

| Artifact | Format | Audience |
|----------|--------|----------|
| `outputs/atlas.csv` | CSV, NCT × 5 flags × MA_id | downstream analysis, reviewers |
| `outputs/ma_rollup.csv` | CSV, MA × flag-density × pooled-effect × crosses-null | headline statistics |
| `outputs/dashboard.html` | self-contained HTML, inline SVG | public Pages site |
| `outputs/verification.html` | self-contained HTML, RapidMeta-style single-trial UI | human spot-check workflow |
| `baseline/cardio_v0.1.0.json` | pinned numerical baseline | Sentinel + Overmind |
| `E156-PROTOCOL.md` | Markdown | E156 workbook entry |
| `docs/methods.md` | Markdown | Synthēsis Methods Note draft |

## 10. Reproducibility contract

- All randomness seeded (`seed=42` for any sampling)
- Ollama model version recorded per row: `gemma2:9b@<sha-from-ollama-list>`
- Prompt text + sha256 stored in `src/tta/judge/prompts.py`; any change → new version + cache invalidation
- AACT snapshot date stamped in `baseline/`
- Pairwise70 file checksum stamped in `baseline/`
- DossierGap fixture version pinned to `v0.3.0`

## 11. Open questions (resolve in plan, not spec)

1. **Cardiology subset definition.** Cochrane Heart Group review IDs vs. broader cardiovascular MeSH filter? Decide in Task 0 preflight.
2. **Flag 0 confidence threshold for "bridged".** > 0.7? > 0.5? Calibrate on fixture.
3. **Flag 2 N-drift threshold.** Default 10 %; revisit after fixture calibration.
4. **PubMed access.** E-utilities vs. cached corpus? E-utilities default; cap rate.
5. **Cochrane HTML scrape (Flag 0 method 3).** Cache aggressively; respect Cochrane robots.
6. **Flag 4 framing — confirm pivot.** Spec now uses "results-posting compliance" not "publication status" (see Flag 4 note). User to confirm in spec review; revert if preferred.

## 12. Non-goals (explicit, to prevent scope creep)

- No causal claim about *why* trials drift
- No automatic correction of MA pooled effects
- No publication-bias re-estimation
- No non-cardiology MAs in v0.1.0
- No retraction integration
- No prospective ITS workflow
- No author / sponsor attribution
- No clinical-recommendation overlay

## 13. Predicted headline (to be measured, not assumed)

> "**X %** of trials inside Cochrane cardiology meta-analyses carry ≥1 silent integrity flag between CT.gov registration and the number Cochrane pooled. Of MAs whose pooled effect crosses null, **Y %** contain disproportionately more flagged trials than MAs whose pooled effect is decisive (rate ratio Z, 95 % CI …)."

If X is small (< 5 %), the paper is a methodological null with reassurance value. If X is large (> 30 %), the paper is a critique-of-evidence-base. Either outcome is publishable; we are pre-committed to reporting whichever holds.

## 14. Atlas-series fit

Slot 5 in the Pairwise70 atlas series:
1. **Reproduction Floor Atlas** — pooled-effect reproducibility
2. **cochrane-modern-re** — DL→REML+HKSJ+PI flip rates
3. **PI Atlas** — PI calibration (compute-pending)
4. **Responder Floor Atlas** — responder MID inflation
5. **Trial Truthfulness Atlas** ← this — upstream trial-level integrity

The series moves outward from "is the math right?" to "is the input right?" — TTA is the rightmost step, asking about the data the prior four atlases all assume valid.

## 15. Preflight requirements (must pass before plan Task 1)

A `python -m tta.cli preflight` command must pass on a clean clone:

- [ ] `ollama list` returns `gemma2:9b`
- [ ] `D:\AACT-storage\AACT\2026-04-12\` exists with all 38 tables
- [ ] `C:\Projects\Pairwise70\data\` exists with ≥ 590 `.rda` files
- [ ] `C:\Projects\dossiergap\outputs\dossier_trials.v0.3.0.csv` exists
- [ ] Python ≥ 3.11, `pyreadr`, `pandas`, `pyarrow`, `requests`, `pytest` importable
- [ ] PubMed E-utilities reachable (HTTP 200 on probe)

Fail-closed with the missing-prereq list. Do not proceed to Task 1 until preflight is green. (Per the "Preflight external prereqs" lesson — Evidence Forecast Phase-1 lost 16 tasks of work to a missing baseline; this prevents recurrence.)
