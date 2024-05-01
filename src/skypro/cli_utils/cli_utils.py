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


def read_directory_of_csvs(directory: str) -> pd.DataFrame:
    """
    Reads all the CSV files from the given directory and concatenates them into a single dataframe.
    """
    dir = os.path.expanduser(directory)
    files = []
    for f in os.listdir(dir):
        path = os.path.join(dir, f)
        if os.path.isfile(path) and f.endswith(".csv"):
            files.append(path)

    df = pd.DataFrame()
    for csv_file in files:
        file_df = pd.read_csv(csv_file)
        df = pd.concat([df, file_df], ignore_index=True)

    return df
