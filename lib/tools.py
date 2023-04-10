from typing import Callable, Sequence
import time


class PrintExecutionTime:
    _func: Callable
    REPEATS: int = 1

    def __init__(self, func: Callable):
        self._func = func

    def __call__(self, *args, **kwargs):
        total_time = 0
        for _ in range(self.REPEATS):
            start_time = time.perf_counter()
            res = self._func(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            total_time += execution_time

        average_time = total_time / self.REPEATS
        print(f"Function: {self._func.__name__}\nTime: {average_time} secs\n")

        return res


def total_elements(seq_2d: Sequence[Sequence]) -> int:
    res = 0
    for row in seq_2d:
        res += len(row)

    return res


def is_ok_2d(seq: Sequence[Sequence]):
    for row in seq:
        for val in row:
            if val < -1000 or val > 1000:
                print(val, type(val))
                return False

    return True


def is_ok(seq: Sequence):
    for val in seq:
        if val < -1000 or val > 1000:
            print(val, type(val))
            return False
