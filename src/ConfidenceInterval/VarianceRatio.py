from dataclasses import dataclass
from typing import Tuple

from .util import get_f_critical_value


@dataclass
class VarianceRatio:
    sample_variance_x: float
    sample_variance_y: float
    confidence_level: float
    sample_size_n: int
    sample_size_m: int

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        f_critical_value_1 = get_f_critical_value(
            confidence_level=self.confidence_level,
            dfn=self.sample_size_n - 1,
            dfm=self.sample_size_m - 1
        )

        f_critical_value_2 = get_f_critical_value(
            confidence_level=self.confidence_level,
            dfn=self.sample_size_m - 1,
            dfm=self.sample_size_n - 1
        )
        return ((1 / f_critical_value_1) * self.sample_variance_x / self.sample_variance_y,
                f_critical_value_2 * self.sample_variance_x / self.sample_variance_y)

    @property
    def width(self) -> float:
        lower_limit, upper_limit = self.confidence_interval
        return upper_limit - lower_limit
