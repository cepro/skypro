import unittest
from datetime import datetime, time, timedelta

import pytz

from skypro.commands.simulator.algorithms.price_curve.peak import (
    _find_active_peak,
    _find_next_peak,
    get_peak_power,
)
from skypro.commands.simulator.algorithms.price_curve.system_state import SystemState
from skypro.commands.simulator.config.config import Approach, Peak, PeakDynamic
from skypro.common.timeutils.clock_time_period import ClockTimePeriod
from skypro.common.timeutils.dayed_period import DayedPeriod
from skypro.common.timeutils.days import Days

TZ = pytz.timezone("Europe/London")


def _make_peak(
    start_h: int,
    end_h: int,
    days_name: str = "weekdays",
    dynamic: "PeakDynamic | None" = None,
) -> Peak:
    return Peak(
        period=DayedPeriod(
            days=Days(name=days_name, tz_str="Europe/London"),
            period=ClockTimePeriod(start=time(start_h), end=time(end_h), tz_str="Europe/London"),
        ),
        approach=Approach(
            to_soe=100,
            encourage_to_soe=None,
            assumed_charge_power=200,
            encourage_charge_duration_factor=0,
            force_charge_duration_factor=1,
            charge_cushion=timedelta(minutes=0),
        ),
        dynamic=dynamic,
    )


class TestFindActivePeak(unittest.TestCase):

    def test_empty_list(self):
        t = TZ.localize(datetime(2026, 1, 5, 12, 0))  # Mon
        self.assertIsNone(_find_active_peak([], t))

    def test_single_peak_inside(self):
        peak = _make_peak(17, 19)
        t = TZ.localize(datetime(2026, 1, 5, 17, 30))  # Mon 17:30
        self.assertIs(_find_active_peak([peak], t), peak)

    def test_single_peak_outside(self):
        peak = _make_peak(17, 19)
        t = TZ.localize(datetime(2026, 1, 5, 8, 0))  # Mon 08:00, before peak
        self.assertIsNone(_find_active_peak([peak], t))

    def test_two_peaks_in_first(self):
        morning = _make_peak(7, 9)
        evening = _make_peak(17, 19)
        t = TZ.localize(datetime(2026, 1, 5, 8, 0))
        self.assertIs(_find_active_peak([morning, evening], t), morning)

    def test_two_peaks_in_second(self):
        morning = _make_peak(7, 9)
        evening = _make_peak(17, 19)
        t = TZ.localize(datetime(2026, 1, 5, 17, 30))
        self.assertIs(_find_active_peak([morning, evening], t), evening)

    def test_two_peaks_between(self):
        morning = _make_peak(7, 9)
        evening = _make_peak(17, 19)
        t = TZ.localize(datetime(2026, 1, 5, 12, 0))
        self.assertIsNone(_find_active_peak([morning, evening], t))

    def test_overlapping_peaks_raises(self):
        a = _make_peak(7, 12)
        b = _make_peak(10, 14)
        t = TZ.localize(datetime(2026, 1, 5, 11, 0))
        with self.assertRaises(ValueError):
            _find_active_peak([a, b], t)

    def test_day_class_filter_weekend(self):
        # Peak only on weekdays — Saturday should not match
        peak = _make_peak(17, 19, days_name="weekdays")
        t = TZ.localize(datetime(2026, 1, 3, 17, 30))  # Sat 17:30
        self.assertIsNone(_find_active_peak([peak], t))


class TestFindNextPeak(unittest.TestCase):

    def test_empty_list(self):
        t = TZ.localize(datetime(2026, 1, 5, 6, 0))
        self.assertIsNone(_find_next_peak([], t))

    def test_before_single_peak(self):
        peak = _make_peak(17, 19)
        t = TZ.localize(datetime(2026, 1, 5, 8, 0))
        self.assertIs(_find_next_peak([peak], t), peak)

    def test_after_single_peak_today(self):
        # No peak left today
        peak = _make_peak(17, 19)
        t = TZ.localize(datetime(2026, 1, 5, 20, 0))
        self.assertIsNone(_find_next_peak([peak], t))

    def test_picks_earliest_upcoming(self):
        morning = _make_peak(7, 9)
        evening = _make_peak(17, 19)
        # At 06:00 both are upcoming — should pick the morning one (earliest start)
        t = TZ.localize(datetime(2026, 1, 5, 6, 0))
        self.assertIs(_find_next_peak([morning, evening], t), morning)

    def test_after_morning_returns_evening(self):
        morning = _make_peak(7, 9)
        evening = _make_peak(17, 19)
        # After morning peak ends, evening is the next-upcoming
        t = TZ.localize(datetime(2026, 1, 5, 10, 0))
        self.assertIs(_find_next_peak([morning, evening], t), evening)

    def test_during_morning_returns_evening(self):
        # While morning peak is active, the next-upcoming is the evening peak
        # (the active peak has start <= t, so it's excluded by the `> t` filter).
        morning = _make_peak(7, 9)
        evening = _make_peak(17, 19)
        t = TZ.localize(datetime(2026, 1, 5, 8, 0))
        self.assertIs(_find_next_peak([morning, evening], t), evening)

    def test_weekend_with_weekday_only_peaks(self):
        morning = _make_peak(7, 9, days_name="weekdays")
        evening = _make_peak(17, 19, days_name="weekdays")
        t = TZ.localize(datetime(2026, 1, 3, 6, 0))  # Saturday
        self.assertIsNone(_find_next_peak([morning, evening], t))


class TestGetPeakPowerMinEndOfPeakSoe(unittest.TestCase):
    """
    The min_end_of_peak_soe knob reserves SoE for post-peak niv-chase. The key
    behavioural change: time_to_empty is computed against `soe - min_end_of_peak_soe`
    instead of `soe`, which creates slack in the peak window so the dynamic
    HOLD-on-LONG branch can actually fire.
    """

    def _common(self):
        return dict(
            time_step=timedelta(minutes=30),
            bess_max_power_discharge=400.0,    # kW; at 400 kW empties 800 kWh in 2h
            microgrid_residual_power=0.0,
        )

    def test_default_zero_means_legacy_behavior(self):
        # Legacy: with soe=800, max_discharge=400, peak window 2h: time_to_empty=2h,
        # latest_time_before_max = peak_end - 2h = peak_start. So at any t inside peak,
        # t >= peak_start = (latest - time_step) → forced full discharge.
        peak = _make_peak(
            17, 19, dynamic=PeakDynamic(prioritise_residual_load=False, min_end_of_peak_soe=0.0)
        )
        t = TZ.localize(datetime(2026, 4, 7, 17, 30))  # Tue 17:30 — mid-peak
        power = get_peak_power(
            peaks=[peak], t=t, soe=800.0, system_state=SystemState.LONG, **self._common()
        )
        # min_end=0 + LONG → no slack → forced full discharge despite LONG
        self.assertEqual(power, -400.0)

    def test_nonzero_min_end_creates_slack_and_holds_on_long(self):
        # Reserve 400 kWh post-peak: dischargeable_soe = 800-400 = 400.
        # time_to_empty = 400/400 = 1h → latest_time = 19:00-1h = 18:00.
        # At t=17:30 we are early enough → flexibility branch → LONG → HOLD (0.0).
        peak = _make_peak(
            17, 19, dynamic=PeakDynamic(prioritise_residual_load=False, min_end_of_peak_soe=400.0)
        )
        t = TZ.localize(datetime(2026, 4, 7, 17, 30))
        power = get_peak_power(
            peaks=[peak], t=t, soe=800.0, system_state=SystemState.LONG, **self._common()
        )
        self.assertEqual(power, 0.0)

    def test_nonzero_min_end_short_system_full_discharge(self):
        # Same setup as above (reserve 400) but system is SHORT → full discharge.
        peak = _make_peak(
            17, 19, dynamic=PeakDynamic(prioritise_residual_load=False, min_end_of_peak_soe=400.0)
        )
        t = TZ.localize(datetime(2026, 4, 7, 17, 30))
        power = get_peak_power(
            peaks=[peak], t=t, soe=800.0, system_state=SystemState.SHORT, **self._common()
        )
        self.assertEqual(power, -400.0)

    def test_late_in_peak_forces_full_discharge_even_with_reserve(self):
        # At t=18:30, latest_time = 18:00, t > 18:00-30min = 17:30 → force full.
        peak = _make_peak(
            17, 19, dynamic=PeakDynamic(prioritise_residual_load=False, min_end_of_peak_soe=400.0)
        )
        t = TZ.localize(datetime(2026, 4, 7, 18, 30))
        power = get_peak_power(
            peaks=[peak], t=t, soe=800.0, system_state=SystemState.LONG, **self._common()
        )
        self.assertEqual(power, -400.0)

    def test_min_end_above_soe_clamps_to_zero_dischargeable(self):
        # If min_end > soe, dischargeable=0, time_to_empty=0, latest_time=peak_end.
        # So t > peak_end - time_step (~18:30) → must full discharge from 18:30 onwards;
        # at 17:30 (well before latest), we're flexible → LONG → HOLD.
        peak = _make_peak(
            17, 19, dynamic=PeakDynamic(prioritise_residual_load=False, min_end_of_peak_soe=1000.0)
        )
        t = TZ.localize(datetime(2026, 4, 7, 17, 30))
        power = get_peak_power(
            peaks=[peak], t=t, soe=800.0, system_state=SystemState.LONG, **self._common()
        )
        self.assertEqual(power, 0.0)


if __name__ == "__main__":
    unittest.main()
