from datetime import datetime


def is_weekday(t: datetime) -> bool:
    # Returns True if the given datetime is on a weekday, or false if it's at the weekend
    return t.weekday() < 5
