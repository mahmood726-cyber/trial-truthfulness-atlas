"""Flag 0 — bridge Pairwise70 (Study, review_doi) -> CT.gov NCT.

Waterfall:
  1. DossierGap direct match by (study_label, review_doi)
  2. AACT id_information cross-ref (deferred to v0.2.0)
  3. Cochrane HTML scrape (deferred to v0.2.0)
  4. Heuristic match by author surname + year +/-1 (deferred to v0.2.0)

v0.1.0 ships only method 1; methods 2-4 land in v0.2.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class BridgeResult:
    nct_id: Optional[str]
    method: str
    confidence: float


def load_dossiergap(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    required = {"nct_id", "study_label", "review_doi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DossierGap CSV missing columns: {missing}")
    return df


def bridge_one(
    pw_row: pd.Series,
    dossiergap: pd.DataFrame,
    aact_id_information: Optional[pd.DataFrame],
) -> BridgeResult:
    study = pw_row.get("Study")
    doi = pw_row.get("review_doi")
    if pd.isna(study) or pd.isna(doi):
        return BridgeResult(nct_id=None, method="unbridgeable", confidence=0.0)

    direct = dossiergap[
        (dossiergap["study_label"] == study) & (dossiergap["review_doi"] == doi)
    ]
    if not direct.empty:
        return BridgeResult(
            nct_id=str(direct["nct_id"].iloc[0]),
            method="dossiergap_direct",
            confidence=0.99,
        )

    return BridgeResult(nct_id=None, method="unbridgeable", confidence=0.0)


def bridge_pairwise70(
    pw_df: pd.DataFrame,
    dossiergap: pd.DataFrame,
    aact_id_information: Optional[pd.DataFrame],
) -> pd.DataFrame:
    out = pw_df.copy()
    results = [bridge_one(row, dossiergap, aact_id_information) for _, row in pw_df.iterrows()]
    out["nct_id"] = [r.nct_id for r in results]
    out["bridge_method"] = [r.method for r in results]
    out["bridge_confidence"] = [r.confidence for r in results]
    return out


def resolution_rate(bridged: pd.DataFrame) -> float:
    if bridged.empty:
        return 0.0
    return float((bridged["bridge_method"] != "unbridgeable").mean())
