from dataclasses import field
from typing import List
from math import isnan


def name_in_json(json_field_name: str):
    """
    This is just a convenience function for specifying the name that the field should have in JSON.
    JSON is normallyCamelCase, whereas python usually_uses_underscores.
    """
    return field(metadata={"data_key": json_field_name})


def enforce_one_option(options: List, hint: str):
    """
    Raises an exception if there is not exactly one non-None or non-NaN option given in `options`
    """
    num_specified = 0
    for option in options:
        if option is None:
            continue
        if isinstance(option, float) and isnan(option):
            continue
        num_specified += 1

    if num_specified != 1:
        raise ValueError(f"One option must be specified from {hint}.")
