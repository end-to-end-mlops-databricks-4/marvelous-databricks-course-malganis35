"""Timing utilities: decorators and context managers for measuring execution time."""

import time
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")


@contextmanager
def measure_time(task_name: str = "") -> AbstractContextManager[None]:
    """Measure the execution time of a code block.

    :param task_name: A descriptive name for the task being measured.
    :yield: Executes the block of code wrapped by the context manager.

    :example:

        with measure_time("Training model"):
            model.fit(X, y)
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{task_name} took {execution_time:.2f} seconds to execute.")


def timeit(func: Callable[P, R]) -> Callable[P, R]:
    """Measure and log the execution time of a function.

    :param func: The function whose execution time is to be measured.
    :return: A wrapped function that logs its execution duration.
    """

    @wraps(func)
    def timeit_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper
