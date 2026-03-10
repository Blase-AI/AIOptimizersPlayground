"""Decorator for timing function execution (logs at DEBUG level)."""
import functools
import logging
import time

logger = logging.getLogger(__name__)


def timed(func):
    """Log execution time of the wrapped function at DEBUG level.

    Args:
        func: Callable to wrap.

    Returns:
        Wrapper that preserves func's signature and docstring.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logger.debug("%s executed in %.6f s", func.__name__, duration)
        return result
    return wrapper
