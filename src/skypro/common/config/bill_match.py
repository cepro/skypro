from typing import List

from marshmallow_dataclass import dataclass

from skypro.common.config.utility import field_with_opts


@dataclass
class BillMatchLineItem:
    rate_names: List[str] = field_with_opts(key="rates")
    unit: str

