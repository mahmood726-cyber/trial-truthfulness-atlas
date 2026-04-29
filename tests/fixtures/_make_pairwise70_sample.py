"""One-shot generator. Run once, commit the .rda outputs.

    python tests/fixtures/_make_pairwise70_sample.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadr

OUT = Path(__file__).parent / "pairwise70_sample"
OUT.mkdir(exist_ok=True)


# CDFAKE001: HFrEF reviews — PARADIGM-HF + VICTORIA share Analysis.name
# "CV death or HF hosp". Two HFrEF trials in one outcome row.
df1 = pd.DataFrame({
    "Analysis.group": ["1.1", "1.1"],
    "Analysis.number": [1, 1],
    "Analysis.name": ["CV death or HF hosp"] * 2,
    "Subgroup": [None, None],
    "Study": ["McMurray 2014", "Armstrong 2020"],
    "Study.year": [2014, 2020],
    "Experimental.N": [4209, 2526],
    "Control.N": [4233, 2524],
    "Mean": [-0.223, -0.105],
    "CI.start": [-0.314, -0.198],
    "CI.end": [-0.139, -0.020],
    "review_doi": [
        "10.1002/14651858.CD012612.pub2",
        "10.1002/14651858.CD013650.pub2",
    ],
})
pyreadr.write_rdata(str(OUT / "CDFAKE001_pub1_data.rda"), df1, df_name="CDFAKE001_pub1_data")


# CDFAKE002: synthetic test cases (FIXTURE-N + Unknown Author).
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


# CDFAKE003: GRIPHON (PAH, selexipag). Separate review because no Cochrane MA
# pools PAH and HFrEF. Real published primary HR was 0.61; log = -0.4943.
# N: 574 + 582 = 1156 (matching Sitbon NEJM 2015).
df3 = pd.DataFrame({
    "Analysis.group": ["1.1"],
    "Analysis.number": [1],
    "Analysis.name": ["Morbidity-mortality composite"],
    "Subgroup": [None],
    "Study": ["Sitbon 2015"],
    "Study.year": [2015],
    "Experimental.N": [574],
    "Control.N": [582],
    "Mean": [-0.4943],
    "CI.start": [-0.776],
    "CI.end": [-0.211],
    "review_doi": ["10.1002/14651858.CD011162.pub2"],
})
pyreadr.write_rdata(str(OUT / "CDFAKE003_pub1_data.rda"), df3, df_name="CDFAKE003_pub1_data")

print("Wrote", list(OUT.glob("*.rda")))
