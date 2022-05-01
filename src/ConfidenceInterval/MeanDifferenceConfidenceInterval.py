from dataclasses import dataclass
from typing import Tuple

from numpy import sqrt

from .util import get_degree_of_freedom, get_pooled_sample_variance, get_t_score


@dataclass
class MeanDifferenceConfidenceInterval:
    mean_x: float
    mean_y: float
    confidence_level: float
    sample_variance_x: float
    sample_variance_y: float
    sample_size_n: int
    sample_size_m: int
    variance_is_equal: bool

    @property
    def critical_value(self) -> float:
        if self.variance_is_equal:
            return get_t_score(
                confidence_level=self.confidence_level,
                degree_of_freedom=self.sample_size_n + self.sample_size_m - 2
            )

        degree_of_freedom = get_degree_of_freedom(
            sample_variance_x=self.sample_variance_x,
            sample_variance_y=self.sample_variance_y,
            sample_size_n=self.sample_size_n,
            sample_size_m=self.sample_size_m
        )
        critical_t = get_t_score(confidence_level=self.confidence_level,
                                 degree_of_freedom=degree_of_freedom)
        return critical_t

    @property
    def standard_error(self) -> float:
        if self.variance_is_equal:
            pooled_variance = get_pooled_sample_variance(
                sample_variance_x=self.sample_variance_x,
                sample_variance_y=self.sample_variance_y,
                sample_size_n=self.sample_size_n,
                sample_size_m=self.sample_size_m
            )
            return float(sqrt(pooled_variance) * sqrt(1 / self.sample_size_n + 1 / self.sample_size_m))

        return float(sqrt(self.sample_variance_x / self.sample_size_n +
                          self.sample_variance_y / self.sample_size_m))

    @property
    def margin_of_error(self) -> float:
        return self.critical_value * self.standard_error

    @property
    def get_confidence_interval(self) -> Tuple[float, float]:
        return (self.mean_x - self.mean_y - self.margin_of_error,
                self.mean_x - self.mean_y + self.margin_of_error)
