"""Core logic: test functions, mesh generation, simulation (no Streamlit)."""
from .test_functions import (
    TEST_FUNCTION_NAMES,
    get_test_functions,
    generate_test_data,
)
from .simulation import run_optimization

__all__ = [
    "TEST_FUNCTION_NAMES",
    "get_test_functions",
    "generate_test_data",
    "run_optimization",
]
