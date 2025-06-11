import logging
from datetime import date, timedelta
from time import sleep
from typing import Callable, Any


def with_retries(func: Callable, max_attempts: int, delay: timedelta) -> Any:
    """
    This will call the given `func`, and retry up to `max_attempt` times if an exception is encountered.
    """
    attempts = 0
    while True:
        try:
            attempts += 1
            result = func()
            return result
        except Exception as e:
            logging.warning(f"Call to function failed: {e}")
            if attempts < max_attempts:
                logging.info(f"Retrying {attempts}/{max_attempts}")
                sleep(delay.total_seconds())
                continue
            else:
                logging.error("Max attempts reached.")
                raise e


def daterange(start: date, end: date):
    """
    Yields every date between start and end.
    """
    for n in range(int((end - start).days + 1)):
        yield start + timedelta(days=n)
