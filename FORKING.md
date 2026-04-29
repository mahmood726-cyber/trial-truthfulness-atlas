<!-- sentinel:skip-file — multi-tenant template; user-strings are intentional -->
# Forking the Trial Truthfulness Atlas

This repo is part of the mahmood726-cyber portfolio. To fork for your
own institution / drug class:

1. Replace `src/tta/data/cochrane_heart_review_dois.txt` with your own
   review-DOI list (or topic-mesh terms).
2. Repoint `TTA_AACT_DIR`, `TTA_PAIRWISE70_DIR`, `TTA_DOSSIERGAP_FIXTURE`
   via env vars in your CI.
3. Update `baseline/cardio_v0.1.0.json` to your fixture's expected counts.
4. Re-run `pytest -v` and `python -m tta.cli build --fixture-mode`.

Author block, ORCID, contact email — see top-level CLAUDE.md /
AGENTS.md for portfolio convention.
