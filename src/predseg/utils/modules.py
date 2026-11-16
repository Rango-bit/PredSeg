class Lambda_LinearWarmupHold:
    """
    Linear warmup from 0 to lambda_value between [start_iter, end_iter],
    0 before start_iter and constant (=lambda_value) from end_iter onwards.
    """

    def __init__(self, lambda_value: float, start_iter: int = 15_000, end_iter: int = 30_000):
        assert end_iter > start_iter, "end_iter must be > start_iter"
        self.lambda_value = float(lambda_value)
        self.start = int(start_iter)
        self.end = int(end_iter)
        self.span = self.end - self.start

    def value(self, global_iter: int) -> float:
        if global_iter < self.start:
            return 0.0
        if global_iter >= self.end:
            return self.lambda_value
        alpha = (global_iter - self.start) / self.span
        return self.lambda_value * alpha
