"""Flag 0 — bridge Pairwise70 (Study, review_doi) -> CT.gov NCT.

Waterfall (highest-confidence first):
  1. DossierGap direct match by (study_label, review_doi)           — confidence 0.99
  2. AACT id_information cross-ref by PMID                          — confidence 0.95
  3. Cochrane HTML scrape (network; fails closed — deferred stub)   — confidence 0.90
  4. Heuristic: author surname + year ±1 against AACT official_title — confidence 0.65

v0.1.0 shipped only method 1. v0.2.1 adds methods 2 and 4; method 3 is a
documented stub that always fails closed (no network access in tests).
"""

from __future__ import annotations

import re
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


def _method2_pmid_crossref(
    pmid: Optional[str],
    aact_id_information: pd.DataFrame,
) -> Optional[str]:
    """Return NCT id if `pmid` appears in aact_id_information, else None."""
    if not pmid or pd.isnull(pmid):
        return None
    pmid = str(pmid).strip()
    matches = aact_id_information[
        (aact_id_information["id_type"].str.upper() == "PMID") &
        (aact_id_information["id_value"].astype(str).str.strip() == pmid)
    ]
    if matches.empty:
        return None
    return str(matches["nct_id"].iloc[0])


# Method 3 stub: Cochrane HTML scrape requires live network access.
# Always fails closed so CI and offline runs are unaffected.
def _method3_cochrane_scrape_stub(
    study: Optional[str],
    review_doi: Optional[str],
) -> Optional[str]:
    """Network-dependent; not implemented. Always returns None (fails closed)."""
    return None


def _method4_surname_year(
    study: Optional[str],
    aact_studies: pd.DataFrame,
) -> Optional[str]:
    """Heuristic: parse '<Surname> <Year>' from Study, search official_title ±1 year.

    Confidence ~0.65. Returns the first matching NCT id, or None.
    """
    if not study or pd.isnull(study):
        return None
    if "official_title" not in aact_studies.columns:
        return None

    # Parse surname and year. Accepts formats like "McMurray 2014" or
    # "McMurray et al 2014". Year must be 4 consecutive digits.
    m = re.search(r"(\b[A-Z][a-z]+)\b.*?\b((?:19|20)\d{2})\b", str(study))
    if not m:
        return None
    surname = m.group(1).lower()
    year = int(m.group(2))

    # Filter to rows whose official_title contains the surname (case-insensitive).
    title_col = aact_studies["official_title"].fillna("").str.lower()
    surname_match = title_col.str.contains(surname, regex=False)
    candidates = aact_studies[surname_match].copy()
    if candidates.empty:
        return None

    # Check year ±1 against completion_date or primary_completion_date.
    for date_col in ("completion_date", "primary_completion_date"):
        if date_col not in candidates.columns:
            continue
        parsed = pd.to_datetime(candidates[date_col], errors="coerce")
        year_match = parsed.dt.year.between(year - 1, year + 1)
        hits = candidates[year_match]
        if not hits.empty:
            return str(hits["nct_id"].iloc[0])
    return None


def bridge_one(
    pw_row: pd.Series,
    dossiergap: pd.DataFrame,
    aact_id_information: Optional[pd.DataFrame],
    aact_studies: Optional[pd.DataFrame] = None,
) -> BridgeResult:
    study = pw_row.get("Study")
    doi = pw_row.get("review_doi")
    if pd.isnull(study) if study is None or isinstance(study, float) else not study:
        return BridgeResult(nct_id=None, method="unbridgeable", confidence=0.0)
    if pd.isnull(doi) if doi is None or isinstance(doi, float) else not doi:
        return BridgeResult(nct_id=None, method="unbridgeable", confidence=0.0)

    # Method 1: DossierGap direct match.
    direct = dossiergap[
        (dossiergap["study_label"] == study) & (dossiergap["review_doi"] == doi)
    ]
    if not direct.empty:
        nct = direct["nct_id"].iloc[0]
        if isinstance(nct, str) and nct.strip():
            return BridgeResult(
                nct_id=nct.strip(),
                method="dossiergap_direct",
                confidence=0.99,
            )

    # Method 2: PMID cross-ref via AACT id_information.
    if aact_id_information is not None:
        pmid = pw_row.get("expected_pmid")
        nct = _method2_pmid_crossref(pmid, aact_id_information)
        if nct:
            return BridgeResult(nct_id=nct, method="pmid_crossref", confidence=0.95)

    # Method 3: Cochrane HTML scrape — fails closed; no network in tests.
    # nct = _method3_cochrane_scrape_stub(study, doi)  # always None

    # Method 4: Surname + year heuristic.
    if aact_studies is not None:
        nct = _method4_surname_year(study, aact_studies)
        if nct:
            return BridgeResult(nct_id=nct, method="surname_year", confidence=0.65)

    return BridgeResult(nct_id=None, method="unbridgeable", confidence=0.0)


def bridge_pairwise70(
    pw_df: pd.DataFrame,
    dossiergap: pd.DataFrame,
    aact_id_information: Optional[pd.DataFrame],
    aact_studies: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    out = pw_df.copy()

    # Wire expected_pmid from DossierGap into each row so method 2 can use it.
    if "expected_pmid" in dossiergap.columns:
        pmid_map = (
            dossiergap[["study_label", "review_doi", "expected_pmid"]]
            .drop_duplicates(subset=["study_label", "review_doi"])
            .set_index(["study_label", "review_doi"])["expected_pmid"]
        )
        def _lookup_pmid(row):
            key = (row.get("Study"), row.get("review_doi"))
            return pmid_map.get(key, None)
        out["expected_pmid"] = [_lookup_pmid(row) for _, row in out.iterrows()]

    results = [
        bridge_one(row, dossiergap, aact_id_information, aact_studies=aact_studies)
        for _, row in out.iterrows()
    ]
    out["nct_id"] = [r.nct_id for r in results]
    out["bridge_method"] = [r.method for r in results]
    out["bridge_confidence"] = [r.confidence for r in results]
    return out


def resolution_rate(bridged: pd.DataFrame) -> float:
    if bridged.empty:
        return 0.0
    return float((bridged["bridge_method"] != "unbridgeable").mean())
