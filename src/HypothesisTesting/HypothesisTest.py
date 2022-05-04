from dataclasses import dataclass

import scipy.stats as st
from ConfidenceInterval.SingleMean import SingleMean


@dataclass
class HypothesisTest:
    claimed_mean: float
    sample_mean: float
    sample_variance: float
    confidence_level: float
    sample_size: int
    tails: str
    score: str

    @property
    def confidence_interval(self) -> SingleMean:
        return SingleMean(mean=self.sample_mean,
                          variance=self.sample_variance,
                          sample_size=self.sample_size,
                          score=self.score,
                          confidence_level=self.confidence_level,
                          tails=self.tails)

    @property
    def reject_null_hypothesis(self) -> bool:
        test_statistic = (self.sample_mean - self.claimed_mean) / self.confidence_interval.standard_error
        test_star = self.confidence_interval.critical_value

        print(f"{test_statistic=}")
        print(f"{test_star=}")
        if self.tails == "=":
            return test_statistic < -test_star or test_statistic > test_star
        elif self.tails == ">=":
            return test_statistic < test_star
        return test_statistic > test_star

    @property
    def p_value(self) -> float:
        test_statistic = (self.sample_mean - self.claimed_mean) / self.confidence_interval.standard_error
        critical_func = (lambda x: st.t.cdf(x, self.sample_size - 1)) if self.score == "t" else lambda x: st.norm.cdf(x)
        if self.tails == "=":
            return 2 * (1 - critical_func(test_statistic))
        elif self.tails == "<=":
            return 1 - critical_func(test_statistic)
        return critical_func(test_statistic)

    @property
    def beta(self) -> float:
        if self.tails == "<=":
            return st.norm.cdf((self.claimed_mean - self.sample_mean) / self.confidence_interval.standard_error +
                               self.confidence_interval.critical_value)
        elif self.tails == ">=":
            return 1 - st.norm.cdf((self.claimed_mean - self.sample_mean) / self.confidence_interval.standard_error -
                                   self.confidence_interval.critical_value)
        return st.norm.cdf((self.claimed_mean - self.sample_mean) / self.confidence_interval.standard_error +
                           self.confidence_interval.critical_value) - \
               st.norm.cdf((self.claimed_mean - self.sample_mean) / self.confidence_interval.standard_error -
                           self.confidence_interval.critical_value)

    @property
    def power(self) -> float:
        return 1 - self.beta


if __name__ == "__main__":
    # hypothesis_test = HypothesisTest(
    #     claimed_mean=120,
    #     sample_mean=130.1,
    #     sample_variance=21.21 ** 2,
    #     confidence_level=0.95,
    #     sample_size=100,
    #     tails="=",
    #     score="t"
    # )

    # hypothesis_test = HypothesisTest(
    #     claimed_mean=65000,
    #     sample_mean=64000,
    #     sample_variance=4000 ** 2,
    #     confidence_level=0.95,
    #     sample_size=64,
    #     tails=">=",
    #     score="z"
    # )

    # hypothesis_test = HypothesisTest(
    #     claimed_mean=75,
    #     sample_mean=68,
    #     sample_variance=10 ** 2,
    #     confidence_level=0.95,
    #     sample_size=16,
    #     tails=">=",
    #     score="z"
    # )

    # hypothesis_test = HypothesisTest(
    #     claimed_mean=60,
    #     sample_mean=62.75,
    #     sample_variance=10 ** 2,
    #     confidence_level=0.95,
    #     sample_size=52,
    #     tails="<=",
    #     score="z"
    # )

    # hypothesis_test = HypothesisTest(
    #     claimed_mean=4,
    #     sample_mean=4.3,
    #     sample_variance=1.2 ** 2,
    #     confidence_level=0.9,
    #     sample_size=9,
    #     tails="=",
    #     score="t"
    # )

    hypothesis_test = HypothesisTest(
        claimed_mean=100,
        sample_mean=108,
        sample_variance=16 ** 2,
        confidence_level=0.95,
        sample_size=16,
        tails="<=",
        score="z"
    )
    print(hypothesis_test.power)
    print(hypothesis_test.p_value)
    print(hypothesis_test.reject_null_hypothesis)
