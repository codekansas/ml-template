class Meter:
    def __init__(self) -> None:
        self.min_val: int | float | None = None
        self.max_val: int | float | None = None
        self.total_sum: int | float | None = None
        self.num_vals = 0

    def add(self, value: int | float) -> None:
        self.min_val = value if self.min_val is None else min(self.min_val, value)
        self.max_val = value if self.max_val is None else max(self.max_val, value)
        self.total_sum = value if self.total_sum is None else self.total_sum + value
        self.num_vals += 1

    @property
    def mean_val(self) -> float | None:
        if self.total_sum is None or self.num_vals == 0:
            return None
        return self.total_sum / self.num_vals
