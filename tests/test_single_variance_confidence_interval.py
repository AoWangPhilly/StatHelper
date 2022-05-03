import pytest
from ConfidenceInterval.SingleVariance import SingleVariance


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
                {
                    "sample_variance": 4.2,
                    "confidence_level": 0.95,
                    "sample_size": 10
                },
                (1.987, 13.998)

        ),
        (
                {
                    "sample_variance": 2.81 ** 2,
                    "confidence_level": 0.95,
                    "sample_size": 9
                },
                (3.603, 28.980)
        ),
        (
                {
                    "sample_variance": 6.61266666667,
                    "confidence_level": 0.99,
                    "sample_size": 15
                },
                (2.956, 22.720)
        ),
    ]
)
def test_confidence_interval(test_input, expected, helpers):
    assert helpers.round_confidence_interval(
        SingleVariance(**test_input).find_confidence_interval(),
        round_it=3
    ) == expected
