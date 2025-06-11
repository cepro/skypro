from datetime import time
from typing import Tuple

from marshmallow import Schema, fields
from marshmallow_dataclass import NewType

from skypro.common.timeutils.clock_time_period import ClockTimePeriod
from skypro.common.timeutils.dayed_period import DayedPeriod, Days

"""
This file handles parsing of YAML into a DayedPeriod type. See the DayedPeriod class for more info.
"""


class DayedPeriodSchema(Schema):
    days = fields.Str()
    start = fields.Str()  # This is not just a time, but also contains the timezone location
    end = fields.Str()  # This is not just a time, but also contains the timezone location


class DayedPeriodField(fields.Field):
    """
    Handles deserializing a 'dayed period' from YAML.
    """
    def _deserialize(self, value: dict, attr, data, **kwargs):
        validated_dict = DayedPeriodSchema().load(value)

        day_name, day_tz = _parse_day_str(validated_dict["days"])
        start, start_tz = _parse_time_str(validated_dict["start"])
        end, end_tz = _parse_time_str(validated_dict["end"])

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


# The marshmallow_dataclass library doesn't use the DayedPeriodField directly, but instead needs a Type defining:
DayedPeriodType = NewType('DayedPeriod', DayedPeriod, DayedPeriodField)


def _parse_time_str(t_str: str) -> Tuple[time, str]:
    """
    Parses a string in the format "HH:MM:SS:<timezone-location>" and returns the associated `datetime.time` and
    timezone location string. An example string could be "12:30:00:Europe/London".
    """
    components = t_str.split(":")
    if len(components) != 4:
        raise ValueError(f"Time '{t_str}' is not in the format 'HH:MM:SS:<timezone-location>'")

    t = time(hour=int(components[0]), minute=int(components[1]), second=int(components[2]))
    tz_str = components[3]

    return t, tz_str


def _parse_day_str(day_str: str) -> Tuple[str, str]:
    """
    Parses a day string in the format "<day-name>:<timezone-location>" and returns the individual components. An
    example string could be "weekdays:Europe/London".
    """
    components = day_str.split(":")
    if len(components) != 2:
        raise ValueError(f"Day configuration '{day_str}' is not in the format '<day-name>:<timezone-location>'")

    name = components[0]
    tz_str = components[1]

    return name, tz_str



