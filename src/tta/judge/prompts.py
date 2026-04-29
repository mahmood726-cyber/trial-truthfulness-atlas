"""Frozen prompt strings for the local-LLM judge layer.

Any change to a prompt string requires bumping its version suffix
(`_V1` -> `_V2`) and its sha256 constant. Cached judgments key on the
sha256, so a new version invalidates the cache automatically.
"""

from __future__ import annotations

import hashlib

OUTCOME_DRIFT_V1 = """\
You are a clinical-trials methodologist. Compare two descriptions of a
trial outcome:

  registered_outcome: {registered_outcome}
  ma_extracted_outcome: {ma_extracted_outcome}

Reply with EXACTLY ONE of these labels (lowercase, no punctuation):

  identical                — same construct, same time-point, same definition
  refinement               — same construct but narrower / clearer wording
  substantively_different  — different construct or time-point or definition

Reply with ONLY the label. No explanation.
"""

OUTCOME_DRIFT_V1_SHA256 = hashlib.sha256(OUTCOME_DRIFT_V1.encode("utf-8")).hexdigest()


def render_outcome_drift(registered_outcome: str, ma_extracted_outcome: str) -> str:
    return OUTCOME_DRIFT_V1.format(
        registered_outcome=registered_outcome,
        ma_extracted_outcome=ma_extracted_outcome,
    )
