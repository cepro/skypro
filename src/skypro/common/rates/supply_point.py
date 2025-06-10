class SupplyPoint:
    """
    Represents the billing reference point and multiplier for p/kWh charges.

    These can be defined flexibly, but are actually very pre-defined as:
    - MSP (or meter supply point) which means we are charged based on the volume we see at our meter, so the line loss factor should be 1.00
    - GSP (or grid supply point) which means we are charged based on where power from the transmission network connects with the distribution network, so the line loss factor should be > 1.00
    - NSP (or notional supply point) which means we are charged based on a notional 'middle' of the transmission network. The line loss factor should be > 1.00 and > GSP.
    """
    def __init__(self, name: str, line_loss_factor: float):
        self.name = name
        self.line_loss_factor = line_loss_factor

    def __str__(self) -> str:
        return f"{self.name}_{self.line_loss_factor}"

    def __repr__(self):
        return self.__str__()
