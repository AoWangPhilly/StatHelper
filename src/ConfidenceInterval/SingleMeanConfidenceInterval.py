from dataclasses import dataclass
from typing import Tuple, Union

from numpy import sqrt

from .util import get_z_score, get_t_score


class InvalidScoreError(Exception):
    pass


@dataclass
class SingleMeanConfidenceInterval:
    mean: float
    confidence_level: float
    variance: float
    sample_size: int
    score: str

    @property
    def critical_value(self) -> float:
        if self.score == "z":
            return get_z_score(confidence_level=self.confidence_level)
        elif self.score == "t":
            return get_t_score(confidence_level=self.confidence_level,
                               degree_of_freedom=self.sample_size - 1)
        else:
            raise InvalidScoreError("Score must either be a Z or T-critical value")

    @property
    def standard_error(self) -> float:
        return float(sqrt(self.variance / self.sample_size))

    @property
    def margin_of_error(self) -> float:
        return float(self.critical_value * self.standard_error)

    @property
    def length(self) -> float:
        return 2 * self.margin_of_error

    def get_confidence_interval(self, round_by: Union[int, None] = None) -> Tuple[float, float]:
        if round_by is None:
            return (self.mean - self.margin_of_error,
                    self.mean + self.margin_of_error)
        return (round(self.mean - self.margin_of_error, round_by),
                round(self.mean + self.margin_of_error, round_by))
