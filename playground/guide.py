"""Guide tab: how to use the interface and tips."""
import streamlit as st


def render_guide_tab(tab):
    """Render tab with user guide and parameter tips."""
    with tab:
        st.markdown("#### Как пользоваться Playground")
        with st.expander("Пошаговая инструкция", expanded=True):
            st.markdown("""
            1. **В боковой панели** выберите один или несколько оптимизаторов и при необходимости настройте параметры.
            2. **Выберите тестовую функцию** (Quadratic, Rastrigin, Rosenbrock и др.) для анализа ландшафта.
            3. Нажмите **«Запустить симуляцию»** — траектории появятся на вкладке «Визуализация».
            4. **Переключайтесь между вкладками**: Визуализация, Метрики, Описание.
            5. **«Сохранить результаты»** — скачать ZIP с CSV и JSON. **«Сбросить»** — очистить и начать заново.
            """)
        with st.expander("Параметры", expanded=False):
            st.markdown("""
            - **Скорость обучения**, **Моментум**, **Weight Decay** — общие гиперпараметры для всех методов.
            - **Итерации**, **Разрешение сетки**, **Диапазон осей** — настройки симуляции и визуализации.
            - **beta1, beta2** (Adam, AdamW, AMSGrad, Sophia), **beta** (Lion), **trust_coeff** (LARS) — специфичные параметры методов.
            """)
        with st.expander("Советы по выбору функции", expanded=False):
            st.markdown("""
            - Начните с **Quadratic** для проверки сходимости, затем **Ackley** или **Levy**.
            - Для мультимодальных функций (Rastrigin, Ackley) увеличивайте **итерации** до 200–500.
            - Отключите шум для стабильных результатов на **Beale** или **Rosenbrock**.
            """)
