from dataclasses import field, Field
from typing import List, Optional

from numpy import isnan


def field_with_opts(key: Optional[str] = None, default: Optional = None) -> Field:
    """
    This is just a convenience function for specifying the name that the field should have in YAML when defining a
    configuration using marshmallow.
    YAML is normallyCamelCase, whereas python usually_uses_underscores.
    You can also specify a default value for the field if the configuration doesn't contain a value.
    """
    meta_data = {}

    if key is not None:
        meta_data["data_key"] = key

    # TODO: this isn't quite right, because there is no way of specifying a None default value?
    if default is None:
        return field(metadata=meta_data)
    else:
        return field(metadata=meta_data, default=default)


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
