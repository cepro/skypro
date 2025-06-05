import os
from datetime import datetime, timedelta
from typing import Callable, Optional, List

import pandas as pd

from skypro.common.config.data_source_csv import CSVDataSource
from skypro.common.notice.notice import NoticeLevel, Notice
from skypro.common.timeutils.math import floor_hh


def read_directory_of_csvs(directory: str) -> pd.DataFrame:
    """
    Reads all the CSV files from the given directory and concatenates them into a single dataframe.
    """
    directory = os.path.expanduser(directory)
    files = []
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        if os.path.isfile(path) and f.endswith(".csv"):
            files.append(path)

    df = pd.DataFrame()
    for csv_file in files:
        file_df = pd.read_csv(csv_file)
        df = pd.concat([df, file_df], ignore_index=True)

    return df


def get_csv_data_source(source: CSVDataSource, file_path_resolver_func: Callable) -> pd.DataFrame:
    """
    Returns a dataframe representing the CSV data source (either a single CSV file or a directrory of CSV files)
    """
    if source.dir:
        path = file_path_resolver_func(source.dir) if file_path_resolver_func else source.dir
        df = read_directory_of_csvs(path)
    elif source.file:
        path = file_path_resolver_func(source.file) if file_path_resolver_func else source.file
        df = pd.read_csv(path)
    else:
        raise ValueError("Neither file or dir CSV source was specified.")

    return df


def prepare_data_dir(data_dir: str, data_source: str, sub_dir: str, date_tag: datetime) -> str:
    """
    Creates a directory for saving data into and returns the file name to use
    - `data_dir` is the base directory
    - `data_source` will form a directory and is intended to give the source of the data, e.g. 'elexon' or 'flows'.
    - `sub_dir` will form a subdirectory and names the dataset, e.g. 'imbalance_price' or 'bess_readings_30m'
    - `date_tag` gives the start of the date range of the data, which is normally saved monthly.
    """
    directory = os.path.expanduser(os.path.join(data_dir, data_source, sub_dir))
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = f"{data_source}_{sub_dir}_{date_tag:%Y_%m}.csv"
    file_path = os.path.join(directory, file_name)
    return file_path


def drop_extra_rows(df: pd.DataFrame, start: Optional[datetime], end: Optional[datetime]) -> pd.DataFrame:
    """
    Removes any rows outside the start and end time
    """

    if start:
        df = df[df["time"] >= start]
    if end:
        df = df[df["time"] <= end]

    return df


def sanity_checks(df: pd.DataFrame, start, end, context: Optional[str]) -> List[Notice]:
    # Do some sanity checks on the data and returns a list of warnings if things look iffy
    notices = []
    if len(df) == 0:
        notices.append(Notice(
            detail=f"Data source '{context}' is entirely missing for the time range",
            level=NoticeLevel.SEVERE,
        ))
    else:
        if start:
            missing_duration = min(df["time"]) - start
            if missing_duration > timedelta(minutes=0):
                missing_hours = missing_duration.total_seconds()/3600
                notices.append(
                    Notice(
                        detail=f"{missing_hours:.1f}hrs of '{context}' data is missing from the start ({min(df['time'])} vs {start})",
                        level=_notice_level_for_duration(missing_duration),
                    )
                )
        if end:
            end_floor_hh = floor_hh(end)
            missing_duration = end_floor_hh - max(df["time"])
            if missing_duration > timedelta(minutes=0):
                missing_hours = missing_duration.total_seconds()/3600
                notices.append(
                    Notice(
                        detail=f"{missing_hours:.1f}hrs of '{context}' data is missing from the end ({max(df['time'])} vs {end_floor_hh})",
                        level=_notice_level_for_duration(missing_duration),
                    )
                )

    return notices


def _notice_level_for_duration(duration: timedelta) -> NoticeLevel:
    if duration > timedelta(hours=12):
        return NoticeLevel.SEVERE
    elif duration > timedelta(hours=2):
        return NoticeLevel.NOTEWORTHY
    else:
        return NoticeLevel.INFO
