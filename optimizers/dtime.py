import time
import functools
import logging

logging.basicConfig(level=logging.INFO) 

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logging.info(f"{func.__name__} executed in {duration:.6f} seconds")
        return result
    return wrapper
