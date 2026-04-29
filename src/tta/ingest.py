"""Ingest AACT TSV snapshots and Pairwise70 .rda files into parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import pyreadr

DATE_COLUMNS_BY_TABLE = {
    "studies": ["completion_date", "results_first_posted_date"],
}


def load_aact_table(snapshot_dir: Path, table: str) -> pd.DataFrame:
    path = snapshot_dir / f"{table}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    parse_dates = DATE_COLUMNS_BY_TABLE.get(table, [])
    df = pd.read_csv(
        path,
        sep="|",
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        engine="c",
    )
    for col in parse_dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def materialise_aact(
    snapshot_dir: Path,
    out_dir: Path,
    tables: Iterable[str],
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for table in tables:
        df = load_aact_table(snapshot_dir, table)
        out_path = out_dir / f"aact_{table}.parquet"
        df.to_parquet(out_path, index=False)
        written[table] = out_path
    return written


def _review_id_from_filename(path: Path) -> str:
    name = path.stem
    if name.endswith("_data"):
        name = name[: -len("_data")]
    return name


def load_pairwise70_rda(path: Path) -> pd.DataFrame:
    result = pyreadr.read_r(str(path))
    if not result:
        raise ValueError(f"no R objects in {path}")
    obj_name, df = next(iter(result.items()))
    df = df.copy()
    df.attrs["review_id"] = _review_id_from_filename(path)
    df["review_id"] = _review_id_from_filename(path)
    return df


def load_pairwise70_dir(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.rda"))
    if not files:
        raise FileNotFoundError(f"no .rda files in {directory}")
    frames = [load_pairwise70_rda(p) for p in files]
    return pd.concat(frames, ignore_index=True, sort=False)


def materialise_pairwise70(directory: Path, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_pairwise70_dir(directory)
    df.to_parquet(out_path, index=False)
    return out_path
