from datetime import date, datetime, time

import pytz


class ClockTimePeriod:
    # Represents a period of time, e.g. "4pm to 7pm" in some timezone. Note there is no date information.
    def __init__(self, start: time, end: time, tz_str: str):
        self.start = start
        self.end = end
        self.tz = pytz.timezone(tz_str)

        if start > end:
            raise ValueError("start must be before end")

    def __str__(self) -> str:
        return f"{self.start} -> {self.end}"

    def __repr__(self) -> str:
        return self.__str__()

    def contains(self, t: datetime) -> bool:
        """
        Returns true if the given t is contained in the period, inclusive of the start, exclusive of the end.
        """

        # Make sure that `t` is in the relevant timezone for the ClockTimePeriod configuration
        t = t.astimezone(self.tz)

        return self.end_absolute(t.date()) > t >= self.start_absolute(t.date())

    def start_absolute(self, d: date) -> datetime:
        """
        Returns the start as an absolute datetime on the given day
        """
        # Note that we cannot pass `self.tz` as the `tzinfo` argument to the `datetime` constructor - it doesn't work
        # with the pytz lib!
        absolute = datetime(d.year, d.month, d.day, self.start.hour, self.start.minute, self.start.second)
        absolute = self.tz.localize(absolute)
        return absolute

    def end_absolute(self, d: date) -> datetime:
        """
        Returns the end as an absolute datetime on the given day
        """
        # Note that we cannot pass `self.tz` as the `tzinfo` argument to the `datetime` constructor - it doesn't work
        # with the pytz lib!
        absolute = datetime(d.year, d.month, d.day, self.end.hour, self.end.minute, self.end.second)
        absolute = self.tz.localize(absolute)
        return absolute
