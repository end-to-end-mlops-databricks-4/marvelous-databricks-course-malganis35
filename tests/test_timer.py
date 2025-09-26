"""Unit tests for the DataProcessor class in mlops_course.feature.data_processor."""

import time
from unittest.mock import MagicMock, patch

from mlops_course.utils.timer import measure_time, timeit


def test_timeit_decorator_returns_result() -> None:
    """Verify that the @timeit decorator returns the correct result.

    :return: Asserts the output of the decorated function is correct.
    """

    @timeit
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


@patch("mlops_course.utils.timer.logger")
def test_timeit_logs_execution_time(mock_logger: MagicMock) -> None:
    """Test that the @timeit decorator logs an execution time message.

    :param mock_logger: Mocked loguru logger.
    :return: Asserts the log message was triggered and formatted correctly.
    """

    @timeit
    def dummy_function() -> str:
        time.sleep(0.01)
        return "done"

    result = dummy_function()
    assert result == "done"

    # Expect a message like "Function dummy_function Took XX.XXXX seconds"
    assert mock_logger.info.call_count == 1
    log_msg = mock_logger.info.call_args[0][0]
    assert "Function dummy_function Took" in log_msg


@patch("mlops_course.utils.timer.logger")
def test_timeit_measures_correct_duration(mock_logger: MagicMock) -> None:
    """Test that the @timeit decorator correctly measures execution time.

    :param mock_logger: Mocked loguru logger.
    :return: Asserts the log message includes correct formatting and function name.
    """

    @timeit
    def test_func() -> int:
        return 42

    result = test_func()
    assert result == 42

    # Extract the log message and validate its content
    assert mock_logger.info.call_count == 1
    log_msg = mock_logger.info.call_args[0][0]
    assert "Function test_func Took" in log_msg
    assert "seconds" in log_msg


@patch("builtins.print")
def test_measure_time_prints_duration(mock_print: MagicMock) -> None:
    """Test that measure_time context manager prints elapsed time.

    :param mock_print: Mocked built-in print function.
    :return: Asserts the printed message contains task name and timing.
    """
    with measure_time("Test block"):
        time.sleep(0.01)  # Simulate some delay

    # Check that print() was called with a message containing the task name
    assert mock_print.call_count == 1
    printed_msg = mock_print.call_args[0][0]
    assert "Test block took" in printed_msg
    assert "seconds to execute" in printed_msg
