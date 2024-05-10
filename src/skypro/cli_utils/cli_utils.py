import json
import os
from typing import Dict, List

import pandas as pd


def get_user_ack_of_warning_or_exit(warning_str: str):
    """
    This forces the CLI user to read the warning by making them enter 'yes' to continue.
    """
    user_input = input(
            f"Warning: {warning_str}. Would you like to continue anyway? ")
    if user_input.lower() not in ['yes', 'y']:
        print("Exiting")
        exit(-1)


def substitute_vars(string: str, variables: Dict[str, str]) -> str:
    """
    Replaces the variables that are present in `string` with their associated value in the `variables` dictionary.
    """
    for key, val in variables.items():
        string = string.replace(f"${key}", val)

    return string


def read_json_file(file_path: str) -> Dict:
    """
    Reads a json file and returns the contents as a dictionary.
    """
    with open(os.path.expanduser(file_path), 'r') as file:
        parsed = json.load(file)

    return parsed
