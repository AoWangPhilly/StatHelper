from abc import ABC, abstractmethod
from typing import Tuple


class ConfidenceInterval(ABC):
    @abstractmethod
    def critical_value(self) -> float:
        ...

    @abstractmethod
    def standard_error(self) -> float:
        ...

    @abstractmethod
    def margin_of_error(self) -> float:
        ...

    @abstractmethod
    def length(self) -> float:
        ...

    @abstractmethod
    def get_confidence_interval(self) -> Tuple[float, float]:
        ...
