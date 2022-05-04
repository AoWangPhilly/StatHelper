from abc import ABC, abstractmethod
from typing import Tuple

from numpy import PINF, NINF

from .Error import InvalidTailError


class ConfidenceInterval(ABC):
    def __init__(self, confidence_level: float, tails: str):
        self.confidence_level = confidence_level
        self.tails = tails

    @property
    @abstractmethod
    def critical_value(self) -> float:
        return 0.0

    @property
    @abstractmethod
    def standard_error(self) -> float:
        return 0.0

    @property
    @abstractmethod
    def confidence_interval(self) -> Tuple[float, float]:
        return 0.0, 0.0

    def _get_alpha(self):
        if self.tails == "=":
            return self.confidence_level + (1 - self.confidence_level) / 2
        elif self.tails == ">=":
            return 1 - self.confidence_level
        elif self.tails == "<=":
            return self.confidence_level
        else:
            raise InvalidTailError("Must either be two tailed (=), or one tailed (>=, <=)")

    def _get_confidence_interval(self, value: float) -> Tuple[float, float]:
        if self.tails == "=":
            return value - self.margin_of_error, value + self.margin_of_error
        elif self.tails == ">=":
            return value - self.margin_of_error, PINF
        else:
            return NINF, value + self.margin_of_error

    @property
    def margin_of_error(self) -> float:
        return float(self.critical_value * self.standard_error)

    @property
    def width(self) -> float:
        return 2 * self.margin_of_error
