import logging
from dataclasses import dataclass
from enum import Enum


class NoticeLevel(Enum):
    """
    Represents the severity of a notice.
    """
    INFO = 1
    NOTEWORTHY = 2
    SEVERE = 3


@dataclass
class Notice:
    """
    Represents a warning, which should be displayed to the user. The higher the level the more aggressively it should be put in front of the user
    """
    detail: str
    level: NoticeLevel

    def log(self):
        logging.warning(f"Level {self.level} notice: {self.detail}")
