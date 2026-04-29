"""Cardiology subset filter for v0.1.0."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd

DOI_LIST_PATH = Path(__file__).parent / "data" / "cochrane_heart_review_dois.txt"

CARDIOVASCULAR_MESH_TERMS = frozenset(
    {
        "heart failure",
        "myocardial infarction",
        "coronary artery disease",
        "coronary disease",
        "atrial fibrillation",
        "hypertension",
        "stroke",
        "cardiovascular diseases",
        "pulmonary arterial hypertension",
        "valvular heart disease",
        "arrhythmias, cardiac",
    }
)


@lru_cache(maxsize=1)
def load_heart_group_dois() -> Set[str]:
    lines = DOI_LIST_PATH.read_text(encoding="utf-8").splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}


def filter_pairwise70(df: pd.DataFrame) -> pd.DataFrame:
    dois = load_heart_group_dois()
    return df[df["review_doi"].isin(dois)].copy()


def filter_aact_browse_conditions(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["downcase_mesh_term"].isin(CARDIOVASCULAR_MESH_TERMS)
    return df[mask].copy()


def cardio_nct_set(browse_conditions: pd.DataFrame) -> Set[str]:
    return set(filter_aact_browse_conditions(browse_conditions)["nct_id"])
