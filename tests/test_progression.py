from datetime import datetime, timedelta

import pytest

from neuro_scan.progression import ScanMeasurement, compare_scans


def test_comparison_is_sorted_and_computes_relative_change():
    start = datetime(2025, 1, 1)
    scans = [
        ScanMeasurement("later", start + timedelta(days=60), 150, 10.0),
        ScanMeasurement("earlier", start, 100, 8.0),
    ]
    result = compare_scans(scans)[0]
    assert result["from"] == "earlier"
    assert result["area_change_pixels"] == 50
    assert result["percent_change"] == 50.0
    assert result["trend"] == "increased"


def test_zero_baseline_does_not_divide_by_zero():
    start = datetime(2025, 1, 1)
    scans = [
        ScanMeasurement("a", start, 0, 0.0),
        ScanMeasurement("b", start + timedelta(days=1), 5, 1.0),
    ]
    assert compare_scans(scans)[0]["percent_change"] is None


def test_requires_two_scans():
    with pytest.raises(ValueError):
        compare_scans([ScanMeasurement("a", datetime.now(), 1, 1.0)])
