from skypro.commands.simulator.algorithms.price_curve.system_state import SystemState
from skypro.commands.simulator.config import Microgrid


def get_microgrid_algo_energy(
    config: Microgrid,
    microgrid_residual_energy: float,
    system_state: SystemState,
) -> float:
    """
    The simulation can be configured with a 'microgrid algorithm' which can do:
    - "import avoidance" to prevent the site from importing into loads
    - "export avoidance" tp prevent the site from exporting solar into the grid

    The above can also be configured to only activate when the network is 'long' or 'short' which is when prices
    tend to be low or high respectively.
    """

    if config.local_control:
        if config.local_control.import_avoidance and microgrid_residual_energy > 0:
            return -microgrid_residual_energy
        if config.local_control.export_avoidance and microgrid_residual_energy < 0:
            return -microgrid_residual_energy

    if config.imbalance_control:
        if config.imbalance_control.discharge_into_load_when_short and system_state == SystemState.SHORT and microgrid_residual_energy > 0:
            # The system is short (so prices are high) and the microgrid is importing from the grid, so we should
            # try to discharge the battery to cover the load
            return -microgrid_residual_energy

        if config.imbalance_control.charge_from_solar_when_long and system_state == SystemState.LONG and microgrid_residual_energy < 0:
            # The system is long (so prices are low) and the microgrid is exporting to the grid, so we should
            # try to charge the battery to stop the export
            return -microgrid_residual_energy

    return 0
