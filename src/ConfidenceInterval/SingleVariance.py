from dataclasses import dataclass
from typing import Tuple

from .util import get_chi_squared_critical_values


@dataclass
class SingleVariance:
    sample_variance: float
    confidence_level: float
    sample_size: int
    
    @property
    def confidence_interval(self) -> Tuple[float, float]:
        critical_values = get_chi_squared_critical_values(
            confidence_level=self.confidence_level,
            degree_of_freedom=self.sample_size - 1
        )
        a, b = critical_values["a"], critical_values["b"]
        return ((self.sample_size - 1) * self.sample_variance / b,
                (self.sample_size - 1) * self.sample_variance / a)

    @property
    def width(self) -> float:
        confidence_interval = self.confidence_interval
        lower_limit, upper_limit = confidence_interval
        return upper_limit - lower_limit
