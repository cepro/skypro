from datetime import datetime

import pytz


class Days:
    """
    Represents a set of days, e.g. "weekdays", "weekends" or "all".
    We need a timezone specified because a given instant in time might be a Sunday in one timezone and a Monday in another timezone.
    """

    def __init__(self, name: str, tz_str: str):
        self.name = name
        self.tz = pytz.timezone(tz_str)

    def __str__(self):
        return self.name

    def is_on_day(self, t: datetime) -> bool:

        # Make sure that `t` is in the relevant timezone for the day configuration.
        t = t.astimezone(self.tz)

        if self.name == "all":
            return True
        elif self.name == "weekdays":
            return is_weekday(t)
        elif self.name == "weekends":
            return not is_weekday(t)
        else:
            raise ValueError(f"Unknown days string: '{self.name}'")


def is_weekday(t: datetime) -> bool:
    """
    Returns True if the given datetime is on a weekday, or false if it's at the weekend
    """
    return t.weekday() < 5
