from datetime import time
from typing import Tuple

from marshmallow import Schema, fields
from marshmallow_dataclass import NewType

from skypro.common.timeutils.clock_time_period import ClockTimePeriod
from skypro.common.timeutils.dayed_period import DayedPeriod, Days

"""
This handles parsing of JSON into a DayedPeriod type 
"""


class DayedPeriodSchema(Schema):
    days = fields.Str()
    start = fields.Str()  # This is not just a time, but also contains the timezone location
    end = fields.Str()  # This is not just a time, but also contains the timezone location


def parse_time_str(t_str: str) -> Tuple[time, str]:
    """
    Parses a string in the format "HH:MM:SS:<timezone-location>" and returns the associated `datetime.time` and
    timezone location string.
    """
    components = t_str.split(":")
    if len(components) != 4:
        raise ValueError(f"Time '{t_str}' is not in the format 'HH:MM:SS:<timezone-location>'")

    t = time(hour=int(components[0]), minute=int(components[1]), second=int(components[2]))
    tz_str = components[3]

    return t, tz_str


def parse_day_str(day_str: str) -> Tuple[str, str]:
    """
    Parses a day string in the format "<day-name>:<timezone-location>" and returns the individual components
    """
    components = day_str.split(":")
    if len(components) != 2:
        raise ValueError(f"Day configuration '{day_str}' is not in the format '<day-name>:<timezone-location>'")

    name = components[0]
    tz_str = components[1]

    return name, tz_str


class DayedPeriodField(fields.Field):
    def _deserialize(self, value: dict, attr, data, **kwargs):
        validated_dict = DayedPeriodSchema().load(value)

        day_name, day_tz = parse_day_str(validated_dict["days"])
        start, start_tz = parse_time_str(validated_dict["start"])
        end, end_tz = parse_time_str(validated_dict["end"])

        if start_tz != end_tz:
            raise ValueError("Period contains a start and end time in different timezones")

        return DayedPeriod(
            days=Days(
                name=day_name,
                tz_str=day_tz
            ),
            period=ClockTimePeriod(
                start=start,
                end=end,
                tz_str=start_tz
            )
        )


DayedPeriodType = NewType('DayedPeriod', DayedPeriod, DayedPeriodField)
