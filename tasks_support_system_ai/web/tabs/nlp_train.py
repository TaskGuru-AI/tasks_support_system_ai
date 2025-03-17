import plotly.graph_objects as go
import requests
import streamlit as st

from tasks_support_system_ai.core.config import settings

backend_url = "http://backend:8000/nlp" if settings.is_docker else "http://localhost:8000/nlp"


def select_model():
    st.title("Обучение модели NLP")
    model_type = st.sidebar.selectbox(
        "Выберите модель для обучения:", ["Logistic Regression", "SVM",
                                          "Catboost", "XGBoost", "LightGBM"], key="model_sb"
    )
    config = {}
    if model_type == "Logistic Regression":
        config = {
            "C": st.sidebar.slider("Параметр C (регуляризация):", 0.01, 10.0, 1.0),
            "solver": st.sidebar.selectbox(
                "Алгоритм оптимизации:", ["lbfgs", "liblinear", "saga"], key="log_sb"
            ),
        }
    elif model_type == "Catboost":
        config = {
            "iterations": st.sidebar.number_input("Итерации", min_value=100, max_value=1000, value=100),
            "depth": st.sidebar.slider("Глубина", min_value=3, max_value=16, value=8),
            "learning_rate": st.sidebar.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.09,
                                                     step=0.01),
            "l2_leaf_reg": st.sidebar.number_input("L2 Leaf Reg", min_value=1, max_value=10, value=5),
        }
    elif model_type == "XGBoost":
        config = {
            "max_depth": st.sidebar.slider("Глубина", min_value=3, max_value=16, value=8),
            "learning_rate": st.sidebar.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.09,
                                                     step=0.01),
            "num_boost_round": st.sidebar.number_input("Boosting Rounds", min_value=100, max_value=20000, value=10000),
            "num_class": st.sidebar.number_input("Количество классов", min_value=2, max_value=100, value=10),
        }
    elif model_type == "LightGBM":
        config = {
            "learning_rate": st.sidebar.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.09,
                                                     step=0.01),
            "num_leaves": st.sidebar.number_input("Num Leaves", min_value=2, max_value=100, value=31),
            "max_depth": st.sidebar.slider("Max Depth", min_value=3, max_value=16, value=8),
            "n_estimators": st.sidebar.number_input("N Estimators", min_value=50, max_value=500, value=100),
        }
    elif model_type == "SVM":
        config["C"] = st.sidebar.slider("Параметр C (регуляризация):", 0.01, 10.0, 1.0)
        config["kernel"] = st.sidebar.selectbox(
            "Тип ядра:", ["linear", "poly", "rbf", "sigmoid"], key="svm_sb"
        )
        class_weight_input = st.sidebar.text_input(
            "Class Weight (опционально, e.g. {0: 1.0, 1: 2.0} или 'balanced')", value=None
        )

        class_weight_value = None
        if class_weight_input:
            if class_weight_input.strip().lower() == "balanced":
                class_weight_value = "balanced"
            else:
                try:
                    class_weight_value = eval(class_weight_input)
                except Exception as e:
                    st.error(f"Ошибка в формате class_weight: {str(e)}")
                    class_weight_value = None

        config["class_weight"] = class_weight_value

    model = {"Logistic Regression": "logistic", "SVM": "svm", "Catboost": "catboost",
             "LightGBM": "lightgbm", "XGBoost": "xgboost"}.get(model_type)
    return model, config


def train_model(model, config):
    if st.sidebar.button("Train Model"):
        with st.spinner("Обучение модели..."):
            try:
                response = requests.post(
                    f"{backend_url}/api/fit_nlp",
                    json={"model": model, "config": config},
                )
                model_id = response.json()
                st.session_state.model_ids.append(model_id)
                st.session_state.model_configs[model_id] = [model, config]
                st.success(f"Модель {model} обучена!")

            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка при соединении с сервером: {e}")


def show_statistics():
    if st.session_state.model_ids:
        st.subheader("Статистика")
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.model_config = st.selectbox(
                "Выберите модель для отображения:",
                list(st.session_state.model_configs.values()),
                key="stat_sb",
            )

        with col2:
            st.session_state.metric = st.selectbox(
                "Выберите метрику:", ["weighted_avg", "macro_avg"]
            )
        model_id = next(
            (
                id
                for id, config in st.session_state.model_configs.items()
                if config == st.session_state.model_config
            ),
            None,
        )
        if model_id:
            if model_id not in st.session_state.statistics:
                try:
                    response = requests.get(f"{backend_url}/api/statistics/{model_id}")
                    st.session_state.statistics[model_id] = response.json()
                except requests.exceptions.RequestException as e:
                    st.error(f"Ошибка при соединении с сервером: {e}")
            stats = st.session_state.statistics.get(model_id, {})

            if stats:
                roc_auc = round(stats.get("roc_auc"), 3)
                st.markdown(f"#### **ROC AUC:** {roc_auc}")

                metric_data = stats.get(st.session_state.metric)

                precision = metric_data.get("precision", None)
                recall = metric_data.get("recall", None)
                f1_score = metric_data.get("f1-score", None)

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[precision, 0],
                        mode="lines+markers",
                        name="Precision",
                        line={"color": "blue", "width": 2},
                        fill="tozeroy",
                        fillcolor="rgba(0, 0, 205, 0.2)",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[recall, 0],
                        mode="lines+markers",
                        name="Recall",
                        line={"color": "orange", "width": 2},
                        fill="tozeroy",
                        fillcolor="rgba(255, 165, 0, 0.05)",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[f1_score, 0],
                        mode="lines+markers",
                        name="F1-Score",
                        line={"color": "green", "width": 2},
                        fill="tozeroy",
                        fillcolor="rgba(0, 255, 0, 0.1)",
                    )
                )
                fig.update_layout(
                    title=f"Метрики для {st.session_state.metric}",
                    xaxis_title="Пороги",
                    yaxis_title="Значение метрики",
                    template="plotly_dark",
                    xaxis={"tickvals": [0, 1], "ticktext": ["0", "1"]},
                    yaxis={"range": [0, 1]},
                )

                st.plotly_chart(fig)
        else:
            st.warning("Выберите модель для анализа.")
    else:
        st.info("Нет доступных моделей.")


def remove_model():
    if st.session_state.model_ids:
        st.subheader("Удаление модели")
        model_config = st.selectbox(
            "Выберите модель для удаления:",
            list(st.session_state.model_configs.values()),
            key="rm_sb",
        )
        model_id = next(
            (id for id, config in st.session_state.model_configs.items() if config == model_config),
            None,
        )
        if st.button("Remove Model"):
            with st.spinner("Removing model..."):
                try:
                    requests.delete(f"{backend_url}/api/remove_nlp/{model_id}")
                    st.session_state.model_ids.remove(model_id)
                    del st.session_state.model_configs[model_id]
                    if model_id in st.session_state.statistics:
                        del st.session_state.statistics[model_id]
                    st.success(f"Модель {model_config} удалена!")
                except requests.exceptions.RequestException as e:
                    st.error(f"Ошибка при соединении с сервером: {e}")


def render_train_tab():
    if "model_ids" not in st.session_state:
        st.session_state.model_ids = []
    if "statistics" not in st.session_state:
        st.session_state.statistics = {}
    if "model_configs" not in st.session_state:
        st.session_state.model_configs = {}

    model, config = select_model()
    train_model(model, config)
    show_statistics()
    remove_model()
