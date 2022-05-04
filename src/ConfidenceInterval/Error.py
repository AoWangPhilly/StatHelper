class InvalidScoreError(Exception):
    """The critical value must either be a Z or T distribution"""


class InvalidTailError(Exception):
    """The tail must either be two-tailed or one-tailed"""
