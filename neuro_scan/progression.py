"""Descriptive comparisons between scans explicitly grouped as one case.

These functions compare supplied measurements only; they do not forecast growth.
Physical volume requires validated spatial metadata and 3-D segmentation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence


@dataclass(frozen=True)
class ScanMeasurement:
    label: str
    date: datetime
    area_pixels: int
    area_percent: float


def compare_scans(scans: Sequence[ScanMeasurement]) -> list[dict[str, object]]:
    """Sort provided scans by date and describe pairwise pixel-area changes."""
    if len(scans) < 2:
        raise ValueError("At least two scans from the same case are required.")
    ordered = sorted(scans, key=lambda scan: scan.date)
    if any(scan.area_pixels < 0 for scan in ordered):
        raise ValueError("Tumor area cannot be negative.")
    results = []
    for previous, current in zip(ordered, ordered[1:]):
        delta = current.area_pixels - previous.area_pixels
        if previous.area_pixels == 0:
            percent = None
            trend = "not_computable_from_zero_baseline" if current.area_pixels else "stable"
        else:
            percent = 100.0 * delta / previous.area_pixels
            trend = "increased" if delta > 0 else "decreased" if delta < 0 else "stable"
        results.append({
            "from": previous.label,
            "to": current.label,
            "area_change_pixels": delta,
            "percent_change": percent,
            "trend": trend,
        })
    return results
