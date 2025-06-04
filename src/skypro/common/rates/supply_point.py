class SupplyPoint:
    """
    Represents the billing reference point and multiplier for p/kWh charges, normally either "meter", "grid" or
    "notional".
    """
    def __init__(self, name: str, line_loss_factor: float):
        self.name = name
        self.line_loss_factor = line_loss_factor

    def __str__(self) -> str:
        return f"{self.name}_{self.line_loss_factor}"

    def __repr__(self):
        return self.__str__()
