from typing import Tuple, cast

import pytest


class Helpers:
    @staticmethod
    def round_confidence_interval(confidence_interval, round_it) -> Tuple[float, float]:
        return cast(Tuple[float, float], tuple(map(lambda x: round(x, round_it), confidence_interval)))


@pytest.fixture
def helpers():
    return Helpers
