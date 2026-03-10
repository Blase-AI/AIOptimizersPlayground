"""Core logic: test functions, mesh generation, simulation (no Streamlit)."""
from .test_functions import (
    TEST_FUNCTION_NAMES,
    get_test_functions,
    generate_test_data,
)
from .simulation import run_optimization
from .presets import get_preset_names, get_preset_by_name
from .glossary import get_glossary_entries, get_param_help
from .formulas import get_optimizer_formula, get_function_formula, OPTIMIZER_FORMULAS, FUNCTION_FORMULAS

__all__ = [
    "TEST_FUNCTION_NAMES",
    "get_test_functions",
    "generate_test_data",
    "run_optimization",
    "get_preset_names",
    "get_preset_by_name",
    "get_glossary_entries",
    "get_param_help",
    "get_optimizer_formula",
    "get_function_formula",
    "OPTIMIZER_FORMULAS",
    "FUNCTION_FORMULAS",
]
