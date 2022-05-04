from ConfidenceInterval.MeanDifferenceKnownVariance import MeanDifferenceKnownVariance


def test_confidence_interval(helpers):
    assert helpers.round_confidence_interval(
        MeanDifferenceKnownVariance(
            mean_x=19.8,
            mean_y=24,
            confidence_level=0.9,
            sample_variance_x=25,
            sample_variance_y=36,
            sample_size_n=10,
            sample_size_m=12
        ).confidence_interval,
        round_it=2
    ) == (-8.06, -0.34)
