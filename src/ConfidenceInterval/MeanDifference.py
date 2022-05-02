from dataclasses import dataclass
from typing import Tuple, cast, Union

from .ConfidenceInterval import ConfidenceInterval


@dataclass
class MeanDifference(ConfidenceInterval):
    mean_x: float
    mean_y: float
    confidence_level: float
    sample_variance_x: float
    sample_variance_y: float
    sample_size_n: int
    sample_size_m: int

    @property
    def critical_value(self) -> float:
        return 0.0

    @property
    def standard_error(self) -> float:
        return 0.0

    def get_confidence_interval(self, round_by: Union[int, None] = None) -> Tuple[float, float]:
        ci = (self.mean_x - self.mean_y - self.margin_of_error,
              self.mean_x - self.mean_y + self.margin_of_error)
        if round_by is None:
            return ci
        return cast(Tuple[float, float], tuple(map(lambda x: round(x, round_by), ci)))
