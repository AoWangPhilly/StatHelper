from typing import Tuple, cast

import pytest
from ConfidenceInterval.Error import InvalidScoreError
from ConfidenceInterval.SingleMean import SingleMean


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
                {
                    "mean": 37.6,
                    "confidence_level": 0.9,
                    "variance": 3.77 ** 2,
                    "sample_size": 130,
                    "score": "z"
                },
                (37.06, 38.14)
        ),
        (
                {
                    "mean": 37.6,
                    "confidence_level": 0.95,
                    "variance": 3.77 ** 2,
                    "sample_size": 130,
                    "score": "z"
                },
                (36.95, 38.25)
        ),
        (
                {
                    "mean": 37.6,
                    "confidence_level": 0.99,
                    "variance": 3.77 ** 2,
                    "sample_size": 130,
                    "score": "z"
                },
                (36.75, 38.45)
        ),
        (
                {
                    "mean": 29.2,
                    "confidence_level": 0.95,
                    "variance": 7.5 ** 2,
                    "sample_size": 126,
                    "score": "z"
                },
                (27.89, 30.51)
        ),
        (
                {
                    "mean": 4.85,
                    "confidence_level": 0.95,
                    "variance": 0.75 ** 2,
                    "sample_size": 20,
                    "score": "z"
                },
                (4.52, 5.18)
        ),
        (
                {
                    "mean": 44922,
                    "confidence_level": 0.93,
                    "variance": 9821 ** 2,
                    "sample_size": 100,
                    "score": "z"
                },
                (43142.52, 46701.48)
        ),
        (
                {
                    "mean": 21,
                    "confidence_level": 0.95,
                    "variance": 1.76 ** 2,
                    "sample_size": 20,
                    "score": "t"
                },
                (20.18, 21.82)
        ),
        (
                {
                    "mean": 103,
                    "confidence_level": 0.95,
                    "variance": 12 ** 2,
                    "sample_size": 30,
                    "score": "t"
                },
                (98.52, 107.48)
        ),
        (
                {
                    "mean": 21.9,
                    "confidence_level": 0.95,
                    "variance": 4.13414 ** 2,
                    "sample_size": 10,
                    "score": "t"
                },
                (18.94, 24.86)
        )
    ]
)
def test_confidence_interval_round_by_2(test_input, expected):
    assert SingleMean(**test_input).get_confidence_interval(round_by=2) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
                {
                    "mean": 21.9,
                    "confidence_level": 0.95,
                    "variance": 4.13414 ** 2,
                    "sample_size": 10,
                    "score": "t"
                },
                (18.943, 24.857)
        )
    ]
)
def test_confidence_interval_no_rounding(test_input, expected):
    ci = SingleMean(**test_input).get_confidence_interval()
    ci = cast(Tuple[float, float], tuple(map(lambda x: round(x, 3), ci)))
    assert ci == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
                {
                    "mean": 44922,
                    "confidence_level": 0.93,
                    "variance": 9821 ** 2,
                    "sample_size": 100,
                    "score": "z"
                },
                1779.48
        ),
        (
                {
                    "mean": 21.9,
                    "confidence_level": 0.99,
                    "variance": 4.13414 ** 2,
                    "sample_size": 10,
                    "score": "t"
                },
                4.25
        )
    ]
)
def test_margin_of_error(test_input, expected):
    assert round(SingleMean(**test_input).margin_of_error, 2) == expected


def test_length_of_confidence_interval():
    assert round(SingleMean(mean=4.85,
                            confidence_level=0.95,
                            variance=0.75 ** 2,
                            sample_size=54,
                            score="z").width, 2) == 0.4


def test_incorrect_score():
    with pytest.raises(InvalidScoreError):
        SingleMean(
            mean=21,
            confidence_level=0.95,
            variance=1.76 ** 2,
            sample_size=20,
            score="w"
        ).critical_value
