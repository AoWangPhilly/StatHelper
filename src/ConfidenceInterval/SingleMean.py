from dataclasses import dataclass
from typing import Tuple

import scipy.stats as st
from numpy import sqrt

from .ConfidenceInterval import ConfidenceInterval
from .Error import InvalidScoreError


@dataclass
class SingleMean(ConfidenceInterval):
    def __init__(self, mean: float, variance: float, sample_size: int,
                 score: str, confidence_level: float, tails: str = "=") -> None:
        super().__init__(confidence_level, tails)
        self.mean = mean
        self.variance = variance
        self.sample_size = sample_size
        self.score = score

    @property
    def critical_value(self) -> float:
        alpha = self._get_alpha()
        if self.score == "z":
            return st.norm.ppf(alpha)
        elif self.score == "t":
            return st.t.ppf(alpha, self.sample_size - 1)
        else:
            raise InvalidScoreError("Score must either be a Z or T-critical value")

    @property
    def standard_error(self) -> float:
        return float(sqrt(self.variance / self.sample_size))

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        return self._get_confidence_interval(self.mean)
