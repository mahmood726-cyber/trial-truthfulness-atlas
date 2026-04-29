"""One-shot generator. Run once, commit the .rda outputs.

    python tests/fixtures/_make_pairwise70_sample.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadr

OUT = Path(__file__).parent / "pairwise70_sample"
OUT.mkdir(exist_ok=True)


df1 = pd.DataFrame({
    "Analysis.group": ["1.1", "1.1", "1.1"],
    "Analysis.number": [1, 1, 1],
    "Analysis.name": ["CV death or HF hosp"] * 3,
    "Subgroup": [None, None, None],
    "Study": ["McMurray 2014", "Armstrong 2020", "Sitbon 2015"],
    "Study.year": [2014, 2020, 2015],
    "Experimental.N": [4209, 2526, 574],
    "Control.N": [4233, 2524, 582],
    "Mean": [-0.223, -0.105, -0.405],
    "CI.start": [-0.314, -0.198, -0.776],
    "CI.end": [-0.139, -0.020, -0.211],
    "review_doi": [
        "10.1002/14651858.CD012612.pub2",
        "10.1002/14651858.CD013650.pub2",
        "10.1002/14651858.CD011162.pub2",
    ],
})
pyreadr.write_rdata(str(OUT / "CDFAKE001_pub1_data.rda"), df1, df_name="CDFAKE001_pub1_data")


df2 = pd.DataFrame({
    "Analysis.group": ["1.1", "1.1"],
    "Analysis.number": [1, 1],
    "Analysis.name": ["Mortality"] * 2,
    "Subgroup": [None, None],
    "Study": ["Synthetic-NDrift 2018", "Unknown Author 1999"],
    "Study.year": [2018, 1999],
    "Experimental.N": [350, 100],
    "Control.N": [350, 100],
    "Mean": [-0.15, -0.05],
    "CI.start": [-0.30, -0.18],
    "CI.end": [0.0, 0.08],
    "review_doi": [
        "10.1002/14651858.CDFAKE001.pub1",
        "10.1002/14651858.CDFAKE002.pub1",
    ],
})
pyreadr.write_rdata(str(OUT / "CDFAKE002_pub1_data.rda"), df2, df_name="CDFAKE002_pub1_data")

print("Wrote", list(OUT.glob("*.rda")))
