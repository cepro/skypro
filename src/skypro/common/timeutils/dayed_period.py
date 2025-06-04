from datetime import datetime

from skypro.common.timeutils import ClockTimePeriod
from skypro.common.timeutils.days import Days


class DayedPeriod:
    # DayedPeriod gives a period of time on particular days
    def __init__(self, days: Days, period: ClockTimePeriod):
        self.days = days
        self.period = period

    def contains(self, t: datetime) -> bool:
        # Returns true if the given t is on a suitable day and contained in the period.
        # Inclusive of the start, exclusive of the end.

        if not self.days.is_on_day(t):
            return False

        return self.period.contains(t)

    def __str__(self) -> str:
        return f"{self.period} on {self.days}"

    def __repr__(self) -> str:
        return self.__str__()


