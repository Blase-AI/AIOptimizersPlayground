# AI Optimizers Playground

[![Tests](https://github.com/Blase-AI/AIOptimizersPlayground/actions/workflows/tests.yml/badge.svg)](https://github.com/Blase-AI/AIOptimizersPlayground/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-app-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/tested%20with-pytest-yellow?logo=pytest&logoColor=white)](https://docs.pytest.org/)

Interactive app for comparing and exploring optimizers on test functions. Convergence trajectories are visualized in 2D/3D; metrics, result export, and a regularization section in the Glossary are included.

**Demo:** https://aiopt1mizersplayground.streamlit.app/

---

## Requirements and run

- **Python 3.10–3.12** (see `pyproject.toml` and `.python-version`). On 3.14 some dependencies (e.g. pyarrow) may lack prebuilt wheels — use 3.11 or 3.12.
- A virtual environment is recommended.

```bash
git clone https://github.com/Blase-AI/AIOptimizersPlayground
cd AIOptimizersPlayground
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Use the **Playground** page in the sidebar to compare optimizers.

If installation fails (pyarrow, cmake), recreate the environment with an explicit interpreter (e.g. `python3.12 -m venv .venv`) or install from `requirements-minimal.txt`.

---

## Features

- Compare **12+ optimizers** (SGD, GD, RMSProp, Adagrad, Adam, AdamW, AMSGrad, Sophia, Lion, Adan, MARS, LARS) on a single plot.
- **2D/3D visualization** of the test function landscape and trajectories; animation over iterations.
- **10+ test functions**: Quadratic, Rastrigin, Rosenbrock, Himmelblau, Ackley, Griewank, Schwefel, Levy, Beale, and others.
- Hyperparameter tuning: learning rate, momentum, weight decay, β, trust_coeff; option to compare with/without regularization.
- Export results (ZIP with CSV/JSON), in-app help for optimizers and functions.
- **Glossary**: terms, optimizer and test-function formulas, interactive L2/L1/Elastic net regularization plots.

---

## Project structure

```
app.py                 # Streamlit entry point
pages/
  1_Playground.py      # Optimizer comparison
  2_Glossary.py        # Terms, formulas, regularization
playground/            # Playground page logic
  state.py             # Session state, mesh and optimizer cache
  sidebar.py           # Sidebar
  run.py               # Simulation loop
  visualization.py     # Visualization tab
  metrics.py           # Metrics tab
  description.py       # Description tab
  export.py            # Export (ZIP)
core/                  # Logic without Streamlit
  test_functions.py    # Test functions and mesh
  simulation.py        # Optimization loop
  regularization_viz.py # L2/L1/Elastic net visualizations
  formulas.py, glossary.py, descriptions.py, presets.py
optimizers/            # Implementations (base, sgd, adam, …), registry.py
tests/                 # Tests (pytest)
```

---

## Optimizers

| Name     | File       | Description                          | Typical use                 |
|----------|------------|--------------------------------------|-----------------------------|
| SGD      | sgd.py     | Stochastic gradient descent          | MLP, large datasets         |
| GD       | gd.py      | Batch gradient descent               | Tutorials, small tasks       |
| RMSProp  | rmsprop.py | Second-moment smoothing              | RNN, seq2seq                |
| Adagrad  | adagrad.py | Per-coordinate adaptive lr           | NLP, sparse data            |
| Adam     | adam.py    | Adaptive moments (β1, β2)            | CNN, transformers           |
| AMSGrad  | amsgrad.py | Adam with max over second moment     | When Adam is unstable        |
| AdamW    | adamw.py   | Adam + decoupled weight decay        | Modern CNN, Transformers     |
| Sophia   | sophia.py  | Adaptive step, Hessian approximation | LLM, large models           |
| Lion     | lion.py    | Sign-based momentum                  | Transformers, ViT           |
| LARS     | lars.py    | Layer-wise Adaptive Rate Scaling     | Large CNN, large batches     |
| Adan     | adan.py    | Three moments, lookahead            | Transformers, large CNN      |
| MARS     | mars.py    | Variance reduction, momentum        | LLM, pretraining             |

---

## Test functions

| Name      | Type            | Global minimum      |
|-----------|-----------------|----------------------|
| Quadratic | Convex          | (0, 0)               |
| Rastrigin | Multimodal      | (0, 0)               |
| Rosenbrock| Narrow valley   | (1, 1)               |
| Himmelblau| 4 minima        | (3,2), (−2.8,3.1), … |
| Ackley    | Many local mins | (0, 0)               |
| Griewank  | Plateau + minima| (0, 0)               |
| Schwefel  | Hard            | (420.97, 420.97)     |
| Levy      | Plateaus, slopes| (1, 1)               |
| Beale     | Narrow valleys  | (3, 0.5)             |

Detailed formulas and minima are in the app (Glossary → Test functions).

---

## Interface

- **Sidebar**: presets, optimizer selection, hyperparameters, test function, simulation and visualization options, run and reset buttons.
- **Playground**: Visualization tab (2D/3D, animation), Metrics (loss, gradient norm; optional with/without regularization), Description.
- **Home**: Short guide and project overview.
- **Glossary**: terms, optimizer and function formulas, interactive regularization plots (L2, L1, Elastic net).

---

## Tests and development

```bash
pytest tests/ -v
ruff check optimizers/ core/ pages/ app.py
```

GitHub Actions runs tests and Ruff for Python 3.10–3.12. Install in dev mode: `pip install -e ".[dev]"`.

---

## Adding an optimizer

1. Implement a class in `optimizers/`, subclass `BaseOptimizer`, implement `step(params, grads)`.
2. Add a test in `tests/`.
3. Register it in `optimizers/registry.py` in `OPTIMIZER_REGISTRY` (class, param_spec, build_spec with `_LR`, `_MOMENTUM`, `_WD`, `_param` as needed).
4. Export it in `optimizers/__init__.py`.
5. Optionally add a description in `core/descriptions.py`.

---

## License and contact

- **License:** MIT  
- **Author:** [Blase-AI](https://github.com/Blase-AI)  
- Questions and suggestions: open an Issue or Pull Request.
