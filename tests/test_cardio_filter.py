from __future__ import annotations

import pandas as pd

from tta import cardio_filter


def test_load_heart_group_doi_list_skips_comments():
    dois = cardio_filter.load_heart_group_dois()
    assert all(not d.startswith("#") for d in dois)
    assert "10.1002/14651858.CD012612.pub2" in dois


def test_filter_pairwise70_to_cardio(fixtures_dir):
    from tta import ingest

    df = ingest.load_pairwise70_dir(fixtures_dir / "pairwise70_sample")
    out = cardio_filter.filter_pairwise70(df)
    assert len(out) == 5
    assert set(out["review_doi"].unique()) <= set(cardio_filter.load_heart_group_dois())


def test_filter_drops_non_cardio_reviews():
    df = pd.DataFrame({
        "Study": ["x", "y"],
        "review_doi": ["10.1002/14651858.CD000999.pub1", "10.1002/14651858.CD012612.pub2"],
        "Mean": [0.0, -0.1],
    })
    out = cardio_filter.filter_pairwise70(df)
    assert len(out) == 1
    assert out["review_doi"].iloc[0] == "10.1002/14651858.CD012612.pub2"


def test_filter_aact_to_cv_mesh(fixtures_dir):
    from tta import ingest

    bc = ingest.load_aact_table(fixtures_dir / "aact_sample", "browse_conditions")
    out = cardio_filter.filter_aact_browse_conditions(bc)
    assert "NCT99999009" not in set(out["nct_id"])
    assert "NCT01035255" in set(out["nct_id"])
