import math
from datetime import timedelta
from typing import List

import pandas as pd
from skypro.common.notice.notice import Notice, NoticeLevel


def missing_data_warnings(df: pd.DataFrame, data_name: str) -> List[Notice]:
    num_missing = df.isna().sum().sum()
    num_total = len(df) * len(df.columns)

    if num_missing > 0:
        pct = (num_missing / num_total) * 100
        return [
             Notice(
                detail=f"{pct:.1f}% of '{data_name}' data is missing ({num_missing} NaN fields)",
                level=pct_to_notice_level(pct)
            )
        ]

    return []


def energy_discrepancy_warnings(total_energy_1: float, total_energy_2: float, description: str) -> List[Notice]:
    """
    Presents the user with a warning about energy totals not adding up.
    """
    if not math.isclose(total_energy_1, total_energy_2, rel_tol=0.005):
        pct = ((total_energy_1 - total_energy_2) / total_energy_1) * 100
        return [
            # This could be due to the granularity of the telemetry or missing telemetry
            Notice(
                detail=f"{pct:.1f}% discrepancy in '{description}' ({total_energy_1:.0f}kWh vs {total_energy_2:.0f}kWh)",
                level=pct_to_notice_level(pct)
            )
        ]
    return []


def pct_to_notice_level(pct: float) -> NoticeLevel:
    """
    Returns a notice level (indicating the seriousness of the issue) given the percentage error.
    """
    if pct > 5:
        return NoticeLevel.SEVERE
    elif pct > 2:
        return NoticeLevel.NOTEWORTHY
    else:
        return NoticeLevel.INFO


def duration_to_notice_level(duration: timedelta) -> NoticeLevel:
    """
    Returns a notice level (indicating the seriousness of the issue) given the duration that data is missing for.
    """
    if duration > timedelta(hours=12):
        return NoticeLevel.SEVERE
    elif duration > timedelta(hours=2):
        return NoticeLevel.NOTEWORTHY
    else:
        return NoticeLevel.INFO
