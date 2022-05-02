from dataclasses import dataclass

from numpy import sqrt

from .MeanDifference import MeanDifference
from .util import get_z_score


@dataclass
class MeanDifferenceKnownVariance(MeanDifference):
    @property
    def critical_value(self) -> float:
        return get_z_score(self.confidence_level)

    @property
    def standard_error(self) -> float:
        return float(sqrt(self.sample_variance_x / self.sample_size_n +
                          self.sample_variance_y / self.sample_size_m))
