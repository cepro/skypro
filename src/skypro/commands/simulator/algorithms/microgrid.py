from skypro.commands.simulator.algorithms.system_state import SystemState
from skypro.commands.simulator.config.config import BasicMicrogrid


def get_microgrid_algo_energy(
    system_state: SystemState,
    microgrid_residual_energy: float,
    config: BasicMicrogrid
) -> float:

    if system_state == SystemState.UNKNOWN:
        return 0

    microgrid_algo_energy = 0
    if config.discharge_into_load_when_short and system_state == SystemState.SHORT and microgrid_residual_energy > 0:
        # The system is short (so prices are high) and the microgrid is importing from the grid, so we should
        # try to discharge the battery to cover the load
        microgrid_algo_energy = -microgrid_residual_energy
    elif config.charge_from_solar_when_long and system_state == SystemState.LONG and microgrid_residual_energy < 0:
        # The system is long (so prices are low) and the microgrid is exporting to the grid, so we should
        # try to charge the battery to stop the export
        microgrid_algo_energy = -microgrid_residual_energy

    return microgrid_algo_energy
