from typing import Tuple

from numpy import sqrt

from .ConfidenceInterval import ConfidenceInterval
from .util import get_z_score


class MeanDifferenceKnownVariance(ConfidenceInterval):
    def __init__(self, mean_x: float, mean_y: float, sample_variance_x: float, sample_variance_y: float,
                 sample_size_n: int, sample_size_m: int, confidence_level: float, tails: str = "="):
        super().__init__(confidence_level, tails)
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.sample_variance_x = sample_variance_x
        self.sample_variance_y = sample_variance_y
        self.sample_size_n = sample_size_n
        self.sample_size_m = sample_size_m

    @property
    def critical_value(self) -> float:
        return get_z_score(self.confidence_level)

    @property
    def standard_error(self) -> float:
        return float(sqrt(self.sample_variance_x / self.sample_size_n +
                          self.sample_variance_y / self.sample_size_m))

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        return self._get_confidence_interval(self.mean_x - self.mean_y)
