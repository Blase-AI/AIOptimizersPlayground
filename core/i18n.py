"""Translations for RU/EN UI. Language is stored in st.session_state["lang"]."""

from typing import Dict

try:
    import streamlit as st
except ImportError:
    st = None

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "app.title": {"ru": "AI Optimizers Playground", "en": "AI Optimizers Playground"},
    "app.subtitle": {"ru": "Интерактивная площадка для сравнения алгоритмов оптимизации.", "en": "Interactive app for comparing optimization algorithms."},
    "app.info_playground": {"ru": "Выберите **Playground** в боковой панели, чтобы запустить сравнение оптимизаторов на тестовых функциях (Rastrigin, Rosenbrock, Ackley и др.).", "en": "Select **Playground** in the sidebar to run optimizer comparison on test functions (Rastrigin, Rosenbrock, Ackley, etc.)."},
    "app.guide_title": {"ru": "📖 Руководство", "en": "📖 Guide"},
    "app.how_to_use": {"ru": "Как пользоваться", "en": "How to use"},
    "app.how_to_use_text": {"ru": "1. В боковой панели выберите оптимизаторы и настройте параметры.\n2. Выберите тестовую функцию (Quadratic, Rastrigin, Rosenbrock и др.).\n3. Нажмите **«Запустить симуляцию»** — траектории появятся на вкладке «Визуализация».\n4. Переключайтесь между вкладками: Визуализация, Метрики, Описание.\n5. **«Сохранить результаты»** — скачать ZIP. **«Сбросить»** — начать заново.", "en": "1. In the sidebar select optimizers and set parameters.\n2. Choose a test function (Quadratic, Rastrigin, Rosenbrock, etc.).\n3. Click **«Run simulation»** — trajectories appear on the Visualization tab.\n4. Switch tabs: Visualization, Metrics, Description.\n5. **«Save results»** — download ZIP. **«Reset»** — start over."},
    "app.params_tips": {"ru": "Параметры и советы", "en": "Parameters and tips"},
    "app.params_tips_text": {"ru": "- **Скорость обучения**, **Моментум**, **Weight Decay** — общие гиперпараметры.\n- **beta1, beta2** (Adam), **beta** (Lion), **trust_coeff** (LARS) — параметры методов.\n- Для мультимодальных функций (Rastrigin, Ackley) увеличьте итерации до 200–500.\n- Отключите шум для стабильных результатов на Beale и Rosenbrock.", "en": "- **Learning rate**, **Momentum**, **Weight Decay** — common hyperparameters.\n- **beta1, beta2** (Adam), **beta** (Lion), **trust_coeff** (LARS) — method-specific.\n- For multimodal functions (Rastrigin, Ackley) increase iterations to 200–500.\n- Disable noise for stable results on Beale and Rosenbrock."},
    "app.about_title": {"ru": "О проекте", "en": "About"},
    "app.about_text": {"ru": "- **SGD, GD, RMSProp, Adagrad, Adam, AdamW, AMSGrad, Sophia, Lion, Adan, MARS, LARS** — все оптимизаторы в одном месте.\n- Визуализация траекторий в 2D/3D, метрики и экспорт результатов.\n- Единый API: везде используется `optimizer.update(params, grads)` для шага с учётом итерации и регуляризации.", "en": "- **SGD, GD, RMSProp, Adagrad, Adam, AdamW, AMSGrad, Sophia, Lion, Adan, MARS, LARS** — all optimizers in one place.\n- 2D/3D trajectory visualization, metrics, and result export.\n- Unified API: `optimizer.update(params, grads)` for each step with iteration and regularization."},
    "lang.label": {"ru": "Язык", "en": "Language"},
    "pg.title": {"ru": "Сравнение оптимизаторов", "en": "Optimizer comparison"},
    "pg.caption": {"ru": "Запустите симуляцию и сравните траектории на 2D/3D ландшафте тестовой функции.", "en": "Run simulation and compare trajectories on the 2D/3D test function landscape."},
    "pg.tab_viz": {"ru": "Визуализация", "en": "Visualization"},
    "pg.tab_metrics": {"ru": "Метрики", "en": "Metrics"},
    "pg.tab_desc": {"ru": "Описание", "en": "Description"},
    "sidebar.settings": {"ru": "Настройки", "en": "Settings"},
    "sidebar.settings_caption": {"ru": "Параметры и запуск симуляции", "en": "Parameters and simulation run"},
    "sidebar.preset": {"ru": "Сценарий", "en": "Preset"},
    "sidebar.apply_preset": {"ru": "Применить сценарий", "en": "Apply preset"},
    "sidebar.preset_help": {"ru": "Готовый набор оптимизаторов и параметров для быстрого старта.", "en": "Ready-made set of optimizers and parameters for quick start."},
    "sidebar.global_params": {"ru": "Глобальные параметры", "en": "Global parameters"},
    "sidebar.learning_rate": {"ru": "Скорость обучения", "en": "Learning rate"},
    "sidebar.lr_help": {"ru": "Множитель шага обновления. Больше — быстрее, но риск нестабильности.", "en": "Step size multiplier. Higher is faster but may be unstable."},
    "sidebar.momentum": {"ru": "Моментум (β)", "en": "Momentum (β)"},
    "sidebar.momentum_help": {"ru": "Инерция обновления: усреднение прошлых градиентов. Обычно 0.9.", "en": "Update momentum: average of past gradients. Usually 0.9."},
    "sidebar.weight_decay_help": {"ru": "L2-регуляризация: штраф за большие веса. Улучшает обобщение.", "en": "L2 regularization: penalty on large weights. Improves generalization."},
    "sidebar.optimizer_params": {"ru": "Параметры оптимизаторов", "en": "Optimizer parameters"},
    "sidebar.select_optimizers": {"ru": "Выберите оптимизаторы", "en": "Select optimizers"},
    "sidebar.simulation": {"ru": "Настройки симуляции", "en": "Simulation settings"},
    "sidebar.iterations": {"ru": "Количество итераций", "en": "Iterations"},
    "sidebar.resolution": {"ru": "Разрешение сетки", "en": "Grid resolution"},
    "sidebar.bounds": {"ru": "Диапазон осей (X, Y)", "en": "Axis range (X, Y)"},
    "sidebar.random_start": {"ru": "Случайная стартовая точка", "en": "Random start point"},
    "sidebar.random_start_help": {"ru": "Если выключено, используются x0 и y0 ниже для воспроизводимости.", "en": "If off, x0 and y0 below are used for reproducibility."},
    "sidebar.seed_label": {"ru": "Seed (0 = авто)", "en": "Seed (0 = auto)"},
    "sidebar.seed_help": {"ru": "Целое число для воспроизводимости случайной стартовой точки.", "en": "Integer for reproducible random start."},
    "sidebar.test_func": {"ru": "Тестовая функция", "en": "Test function"},
    "sidebar.compare_reg": {"ru": "Сравнить с/без регуляризации", "en": "Compare with/without regularization"},
    "sidebar.compare_reg_help": {"ru": "Запустить ещё раз без weight decay и показать оба варианта на графике в Метриках.", "en": "Run again without weight decay and show both on the Metrics chart."},
    "sidebar.visualization": {"ru": "Визуализация", "en": "Visualization"},
    "sidebar.show_surface": {"ru": "Показать поверхность", "en": "Show surface"},
    "sidebar.show_3d": {"ru": "3D визуализация", "en": "3D visualization"},
    "sidebar.show_colorbar": {"ru": "Показать шкалу цветов", "en": "Show colorbar"},
    "sidebar.color_scheme": {"ru": "Цветовая палитра", "en": "Color scheme"},
    "sidebar.add_noise": {"ru": "Добавить шум в градиенты", "en": "Add noise to gradients"},
    "sidebar.noise_level": {"ru": "Уровень шума", "en": "Noise level"},
    "sidebar.realtime": {"ru": "Режим реального времени", "en": "Realtime mode"},
    "sidebar.run": {"ru": "Запустить симуляцию", "en": "Run simulation"},
    "sidebar.reset": {"ru": "Сбросить", "en": "Reset"},
    "sidebar.save_results": {"ru": "Сохранить результаты", "en": "Save results"},
    "error.rerun": {"ru": "Ошибка при перезапуске", "en": "Error on rerun"},
    "viz.no_data_title": {"ru": "Нет данных для визуализации", "en": "No data for visualization"},
    "viz.no_data_text": {"ru": "Выберите оптимизаторы в боковой панели и нажмите **«Запустить симуляцию»**, чтобы построить траектории на ландшафте функции.", "en": "Select optimizers in the sidebar and click **«Run simulation»** to build trajectories on the function landscape."},
    "viz.running": {"ru": "Симуляция выполняется...", "en": "Simulation running..."},
    "viz.done": {"ru": "Симуляция завершена.", "en": "Simulation complete."},
    "viz.no_reg": {"ru": " (без рег.)", "en": " (no reg.)"},
    "metrics.save_baseline": {"ru": "Сохранить как базовый эксперимент", "en": "Save as baseline experiment"},
    "metrics.baseline_caption": {"ru": "Базовый эксперимент", "en": "Baseline"},
    "metrics.no_data_title": {"ru": "Метрики появятся после симуляции", "en": "Metrics will appear after simulation"},
    "metrics.no_data_text": {"ru": "Запустите сравнение оптимизаторов, чтобы увидеть финальный loss, историю сходимости и нормы градиентов.", "en": "Run optimizer comparison to see final loss, convergence history, and gradient norms."},
    "metrics.optimizer": {"ru": "Оптимизатор", "en": "Optimizer"},
    "metrics.final_loss": {"ru": "Финальные потери", "en": "Final loss"},
    "metrics.iterations": {"ru": "Итерации", "en": "Iterations"},
    "metrics.avg_grad_norm": {"ru": "Средняя норма градиентов", "en": "Avg gradient norm"},
    "metrics.lars_lr": {"ru": "Средний local_lr (LARS)", "en": "Avg local_lr (LARS)"},
    "metrics.best_result": {"ru": "Лучший результат", "en": "Best result"},
    "metrics.baseline_section": {"ru": "Метрики базового эксперимента", "en": "Baseline metrics"},
    "metrics.history_loss": {"ru": "История потерь", "en": "Loss history"},
    "metrics.grad_norm": {"ru": "Норма градиентов", "en": "Gradient norm"},
    "desc.test_func": {"ru": "Тестовая функция", "en": "Test function"},
    "desc.global_min": {"ru": "Глобальный минимум", "en": "Global minimum"},
    "desc.optimizers": {"ru": "Оптимизаторы", "en": "Optimizers"},
    "desc.no_description": {"ru": "Нет описания.", "en": "No description."},
    "desc.formulas": {"ru": "Формулы", "en": "Formulas"},
    "desc.code_step": {"ru": "Код шага", "en": "Step code"},
    "export.report_title": {"ru": "Отчёт эксперимента", "en": "Experiment report"},
    "export.params_section": {"ru": "Параметры", "en": "Parameters"},
    "export.test_func": {"ru": "Тестовая функция", "en": "Test function"},
    "export.optimizers": {"ru": "Оптимизаторы", "en": "Optimizers"},
    "export.learning_rate": {"ru": "Скорость обучения", "en": "Learning rate"},
    "export.iterations": {"ru": "Итерации", "en": "Iterations"},
    "export.start_random": {"ru": "Стартовая точка: случайная", "en": "Start point: random"},
    "export.start_fixed": {"ru": "Стартовая точка: фиксированная", "en": "Start point: fixed"},
    "export.metrics_section": {"ru": "Метрики", "en": "Metrics"},
    "export.conclusion": {"ru": "Вывод", "en": "Conclusion"},
    "export.best_loss": {"ru": "Лучший по финальным потерям", "en": "Best by final loss"},
    "glossary.title": {"ru": "Глоссарий", "en": "Glossary"},
    "glossary.caption": {"ru": "Термины, формулы оптимизаторов и тестовых функций, визуализация регуляризации.", "en": "Terms, optimizer and test function formulas, regularization visualization."},
    "glossary.section_label": {"ru": "Раздел", "en": "Section"},
    "glossary.terms": {"ru": "Термины", "en": "Terms"},
    "glossary.optimizers": {"ru": "Оптимизаторы", "en": "Optimizers"},
    "glossary.test_functions": {"ru": "Тестовые функции", "en": "Test functions"},
    "glossary.regularization": {"ru": "Регуляризация", "en": "Regularization"},
    "glossary.no_formula": {"ru": "Формулы не заданы.", "en": "No formulas defined."},
    "glossary.reg_intro": {"ru": "Регуляризация добавляет штраф к функции потерь, чтобы ограничить сложность модели и улучшить обобщение. Ниже — три типа: **L2 (Ridge)**, **L1 (Lasso)** и **Elastic Net**. На графиках: линии уровня потерь $L(\\theta)$, граница «шара» ограничения (круг для L2, ромб для L1) и точка минимума $\\theta^*$, которая смещается при увеличении $\\lambda$.", "en": "Regularization adds a penalty to the loss to limit model complexity and improve generalization. Below: **L2 (Ridge)**, **L1 (Lasso)**, and **Elastic Net**. Plots show loss contours $L(\\theta)$, constraint ball (circle for L2, diamond for L1), and optimum $\\theta^*$ moving as $\\lambda$ increases."},
    "glossary.math_title": {"ru": "Математика", "en": "Math"},
    "glossary.math_expand": {"ru": "Формулы и отличия", "en": "Formulas and differences"},
    "glossary.viz_heading": {"ru": "Интерактивные визуализации (оси θ₁, θ₂)", "en": "Interactive visualizations (axes θ₁, θ₂)"},
    "glossary.viz_caption": {"ru": "Двигайте слайдер **λ**: линии уровня потерь остаются на месте, смещаются круг/ромб ограничения и точка θ*. Кривая **«Путь θ*(λ)»** показывает, как минимум движется при λ от 0 до 3.", "en": "Move the **λ** slider: loss contours stay fixed; constraint ball and θ* move. The **θ*(λ) path** curve shows how the optimum moves as λ goes from 0 to 3."},
    "glossary.slider_lam": {"ru": "Сила регуляризации λ", "en": "Regularization strength λ"},
    "glossary.tab_l2": {"ru": "L2 (Ridge) — круг", "en": "L2 (Ridge) — circle"},
    "glossary.tab_l1": {"ru": "L1 (Lasso) — ромб", "en": "L1 (Lasso) — diamond"},
    "glossary.slider_l1_ratio": {"ru": "l1_ratio (Elastic Net)", "en": "l1_ratio (Elastic Net)"},
    "metrics.baseline_opt": {"ru": " (базовый)", "en": " (baseline)"},
    "metrics.loss_log": {"ru": "Потери (лог. шкала)", "en": "Loss (log scale)"},
    "export.saved_data": {"ru": "Сохраненные данные", "en": "Saved data"},
    "export.download_zip": {"ru": "Скачать результаты (ZIP)", "en": "Download results (ZIP)"},
    "export.download_md": {"ru": "Скачать отчёт (Markdown)", "en": "Download report (Markdown)"},
    "export.ready": {"ru": "Файлы готовы для скачивания! Нажмите на кнопку во вкладке 'Метрики'.", "en": "Files ready for download. Use the buttons in the Metrics tab."},
    "viz.title_3d": {"ru": "Сравнение оптимизаторов · {} (3D)", "en": "Optimizer comparison · {} (3D)"},
    "viz.title_2d": {"ru": "Сравнение оптимизаторов · {} (2D)", "en": "Optimizer comparison · {} (2D)"},
    "viz.hover_iter": {"ru": "Итерация", "en": "Iteration"},
}


def get_lang() -> str:
    """Return current language: 'ru' or 'en'. Default 'ru'."""
    if st is None:
        return "ru"
    return st.session_state.get("lang", "ru")


def set_lang(lang: str) -> None:
    """Set current language. Call before rerun after user changes language."""
    if st is not None and lang in ("ru", "en"):
        st.session_state["lang"] = lang


def t(key: str) -> str:
    """Return translation for key in current language. Falls back to key if missing."""
    lang = get_lang()
    if key not in TRANSLATIONS:
        return key
    return TRANSLATIONS[key].get(lang, TRANSLATIONS[key].get("en", key))


def render_language_switcher() -> None:
    """Render RU/EN selector in the sidebar and update session_state on change."""
    if st is None:
        return
    if "lang" not in st.session_state:
        st.session_state["lang"] = "ru"
    lang_labels = {"ru": "RU", "en": "ENG"}
    current = st.session_state["lang"]
    choice = st.sidebar.selectbox(
        t("lang.label"),
        options=["ru", "en"],
        format_func=lambda x: lang_labels[x],
        index=0 if current == "ru" else 1,
        key="i18n_lang_select",
    )
    if choice != current:
        set_lang(choice)
        try:
            st.rerun()
        except Exception:
            pass
