"""The external ring-test data file loads verbatim and the derived diameter changes are right."""
import pytest

from surrogates.gp_bo.validation_pop2023 import load_rows


def test_pop2023_rows_and_derived_changes():
    doc, rows = load_rows()
    assert doc["source"]["doi"] == "10.3390/ma16052045"
    assert len(rows) == 9 and {r["condition"] for r in rows} == {"Dry", "Mineral oil", "Graphite in oil"}
    dry02 = rows[0]
    assert dry02["d1i_mm"] == 14.1 and dry02["eps"] == 0.2 and dry02["mu"] == 0.15
    assert dry02["id_change_pct"] == pytest.approx(-6.0)
    assert dry02["id_change_pct_lo"] < dry02["id_change_pct"] < dry02["id_change_pct_hi"]
    assert rows[8]["od_change_pct"] == pytest.approx(21.0)
    for r in rows:   # stated reduction agrees with the heights
        assert r["height_reduction_from_h1"] == pytest.approx(r["eps"])
