import scipy.stats as st


def get_z_score(confidence_level: float) -> float:
    return float(st.norm.ppf(confidence_level + (1 - confidence_level) / 2))


def get_t_score(
        confidence_level: float,
        degree_of_freedom: int
) -> float:
    return float(st.t.ppf(q=confidence_level + (1 - confidence_level) / 2, df=degree_of_freedom))


def get_pooled_sample_variance(
        sample_variance_x: float,
        sample_variance_y: float,
        sample_size_n: int,
        sample_size_m: int
) -> float:
    return ((sample_size_n - 1) * sample_variance_x +
            (sample_size_m - 1) * sample_variance_y) / \
           (sample_size_n + sample_size_m - 2)


def get_degree_of_freedom(
        sample_variance_x: float,
        sample_variance_y: float,
        sample_size_n: int,
        sample_size_m: int
) -> int:
    return int((sample_variance_x / sample_size_n + sample_variance_y / sample_size_m) ** 2 /
               ((sample_variance_x / sample_size_n) ** 2 / (sample_size_n - 1) +
                (sample_variance_y / sample_size_m) ** 2 / (sample_size_m - 1)))
