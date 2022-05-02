import pytest
from ConfidenceInterval.MeanDifferenceUnknownVariance import MeanDifferenceUnknownVariance


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
                {
                    "mean_x": 10.26,
                    "mean_y": 9.02,
                    "confidence_level": 0.95,
                    "sample_variance_x": 6.32,
                    "sample_variance_y": 3.6,
                    "sample_size_n": 10,
                    "sample_size_m": 10,
                    "variance_is_equal": True
                },
                (-0.85, 3.33)
        ),
        (
                {
                    "mean_x": 10.26,
                    "mean_y": 9.02,
                    "confidence_level": 0.95,
                    "sample_variance_x": 6.32,
                    "sample_variance_y": 3.6,
                    "sample_size_n": 10,
                    "sample_size_m": 10,
                    "variance_is_equal": False
                },
                (-0.87, 3.35)
        ),
        (
                {
                    "mean_x": 212.8,
                    "mean_y": 182.8,
                    "confidence_level": 0.95,
                    "sample_variance_x": 73 ** 2,
                    "sample_variance_y": 47 ** 2,
                    "sample_size_n": 120,
                    "sample_size_m": 90,
                    "variance_is_equal": True
                },
                (12.63, 47.37)
        ),
        (
                {
                    "mean_x": 212.8,
                    "mean_y": 182.8,
                    "confidence_level": 0.95,
                    "sample_variance_x": 73 ** 2,
                    "sample_variance_y": 47 ** 2,
                    "sample_size_n": 120,
                    "sample_size_m": 90,
                    "variance_is_equal": False
                },
                (13.63, 46.37)
        )
    ]

)
def test_confidence_interval(test_input, expected):
    assert MeanDifferenceUnknownVariance(**test_input).get_confidence_interval(round_by=2) == expected
