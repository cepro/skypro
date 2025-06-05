from typing import List

from marshmallow_dataclass import dataclass

from skypro.common.config.utility import field_with_opts


@dataclass
class BillMatchLineItem:
    """
    Each `BillMatchLineItem` represents a line on a Supplier invoice. This line should include the costs associated with all the
    named rates, and be in the specified units (e.g. p/kWh or p/day, etc.)
    """
    rate_names: List[str] = field_with_opts(key="rates")
    unit: str

