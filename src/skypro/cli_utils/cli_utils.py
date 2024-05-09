import os

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
