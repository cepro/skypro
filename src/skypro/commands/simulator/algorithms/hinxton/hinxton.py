import pandas as pd

from skypro.commands.simulator.algorithms.utils import get_power, cap_power, get_energy
from skypro.commands.simulator.config.config import HinxtonAlgo


def run_hinxton_algo(
        df_in: pd.DataFrame,
        battery_energy_capacity: float,
        battery_charge_efficiency: float,
        config: HinxtonAlgo,
) -> pd.DataFrame:
    # TODO: the surrounding 'harness' code should be brought out to be shared with all algos

    # Create a separate dataframe for working values
    df = pd.DataFrame(index=df_in.index)

    # These vars keep track of the previous settlement periods values
    last_soe = battery_energy_capacity / 2  # initial SoE is 50%
    last_energy_delta = 0
    last_bess_losses = 0

    time_step = pd.to_timedelta(df_in.index.freq)
    time_step_hours = time_step.total_seconds() / 3600

    # Run through each row (where each row represents a time step) and apply the strategy
    for t in df_in.index:

        # Show the user some progress status
        if (t == df_in.index[0]) or (t.date().day == 1 and t.time().hour == 0 and t.time().minute == 0):
            print(f"Simulating {t.date()}...")

        # Set the `soe` column to the value at the start of this time step (the previous value plus the energy
        # transferred in the previous time step)
        soe = last_soe + last_energy_delta - last_bess_losses
        df.loc[t, "soe"] = soe


        # Just try and make site imports/exports zero
        target_energy_delta = -df_in.loc[t, "microgrid_residual_power"] * time_step_hours
        power = get_power(target_energy_delta, time_step)

        power = cap_power(power, df_in.loc[t, "bess_max_power_charge"], df_in.loc[t, "bess_max_power_discharge"])
        energy_delta = get_energy(power, time_step)

        # Cap the SoE at the physical limits of the battery
        if soe + energy_delta > battery_energy_capacity:
            energy_delta = battery_energy_capacity - soe
        elif soe + energy_delta < 0:
            energy_delta = -soe

        # Apply a charge efficiency
        if energy_delta > 0:
            bess_losses = energy_delta * (1 - battery_charge_efficiency)
        else:
            bess_losses = 0

        df.loc[t, "power"] = power
        df.loc[t, "energy_delta"] = energy_delta
        df.loc[t, "bess_losses"] = bess_losses

        # Save for next iteration...
        last_soe = soe
        last_energy_delta = energy_delta
        last_bess_losses = bess_losses

    return df[["soe", "energy_delta", "bess_losses"]]
