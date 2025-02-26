from enum import Enum

import numpy as np


class SystemState(Enum):
    UNKNOWN = 1
    SHORT = 2
    LONG = 3


def get_system_state(df_in, t, volume_cutoff_for_assumption: float) -> SystemState:
    """
    Returns the predicted SystemState at time t, ideally using the predicted imbalance volume for this SP. But, if
    that is not available yet, then the previous SPs imbalance volume may be used as an assumption if it's volume
    is greater than `volume_cutoff_for_assumption`.
    """
    # Optionally only allow this for the first 10m? df_in.loc[t, "time_into_sp"]<timedelta(minutes=10)

    imbalance_volume_assumed = df_in.loc[t, "imbalance_volume_live"]
    if np.isnan(imbalance_volume_assumed) and \
            abs(df_in.loc[t, "prev_sp_imbalance_volume_live"]) * 1e3 >= volume_cutoff_for_assumption:
        imbalance_volume_assumed = df_in.loc[t, "prev_sp_imbalance_volume_live"]

    if np.isnan(imbalance_volume_assumed):
        return SystemState.UNKNOWN
    elif imbalance_volume_assumed > 0:
        return SystemState.SHORT
    else:
        return SystemState.LONG
