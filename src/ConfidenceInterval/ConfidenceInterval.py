from abc import ABC, abstractmethod
from typing import Tuple


class ConfidenceInterval(ABC):
    @property
    @abstractmethod
    def critical_value(self) -> float:
        ...

    @property
    @abstractmethod
    def standard_error(self) -> float:
        ...

    @abstractmethod
    def get_confidence_interval(self, round_by: int) -> Tuple[float, float]:
        ...

    @property
    def margin_of_error(self) -> float:
        return float(self.critical_value * self.standard_error)

    @property
    def width(self) -> float:
        return 2 * self.margin_of_error
