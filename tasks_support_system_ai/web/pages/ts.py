from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from tasks_support_system_ai.api.models.ts import QueueStats, TimeGranularity
from tasks_support_system_ai.core.config import settings
from tasks_support_system_ai.core.logger import streamlit_logger as logger

st.title("Анализ нагрузки очередей")
st.write("""
**TS часть проекта анализа задач службы техподдержки**

Сервис оперирует двумя датасетами:
- Дневная нагрузка на очереди
- Структура очередей (иерархическая)

Цели сервиса:
- Прогнозировать нагрузку очереди в будущем

Функциональность:
- По умолчанию доступны исходные данные, но можно загрузить свои, \
если совпадает формат и связь между данными
""")

api_url = "http://backend:8000/ts" if settings.is_docker else "http://localhost:8000/ts"

if "data_available" not in st.session_state:
    st.session_state.data_available = False


@st.cache_data(ttl=10)
def check_data_availability():
    try:
        response = requests.get(f"{api_url}/api/data-status")
        return response.json()["status"]
    except requests.exceptions.RequestException:
        logger.error("Data is not available")
        return False


@st.cache_data(ttl=600)
def fetch_all_queues():
    """Fetch queue statistics for all levels"""
    try:
        response = requests.get(f"{api_url}/api/queues")
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching queues: {str(e)}")
        return {"queues_by_level": {}, "total_load": 0}


def sidebar_queue_selector(key_prefix="main"):
    all_queues_data = fetch_all_queues()

    if not all_queues_data["queues_by_level"]:
        return None

    get_all_queues = st.sidebar.checkbox(
        f"Корневая очередь (суммарная нагрузка: {all_queues_data['total_load']})",
        value=False,
        key=f"{key_prefix}_root_queue_checkbox",  # Добавляем уникальный ключ
    )

    if get_all_queues:
        return {"id": 0, "name": "Корневая очередь", "load": all_queues_data["total_load"]}

    st.sidebar.markdown("### Выбранная очередь")

    available_levels = [
        level for level, queues in all_queues_data["queues_by_level"].items() if queues
    ]

    if not available_levels:
        st.sidebar.warning("No queues available")
        return None

    level_queue = st.sidebar.selectbox(
        "Уровень очереди",
        options=available_levels,
        key=f"{key_prefix}_level_selector",  # Обновляем ключ с префиксом
    )

    level_queues = all_queues_data["queues_by_level"].get(str(level_queue), [])

    if not level_queues:
        st.sidebar.warning(f"No queues available for level {level_queue}")
        return None

    selected_queue = st.sidebar.selectbox(
        "Выберите очередь",
        options=level_queues,
        format_func=lambda x: f"Очередь {x['id']} (Нагрузка: {x['load']})",
        key=f"{key_prefix}_queue_selector",  # Обновляем ключ с префиксом
    )

    return selected_queue


st.session_state.data_available = check_data_availability()
if "selected_queue" not in st.session_state:
    st.session_state.selected_queue = None


# Показываем выбор очереди в sidebar для всего приложения
def update_selected_queue():
    st.session_state.selected_queue = sidebar_queue_selector()


update_selected_queue()


def handle_reload():
    try:
        response = requests.post(f"{api_url}/api/reload_local_data")

        if response.ok:
            st.session_state.operation_status = {
                "type": "success",
                "message": "Data reloaded successfully!",
                "result": response.json(),
            }
            fetch_all_queues.clear()
        else:
            st.session_state.operation_status = {
                "type": "error",
                "message": f"Error: {response.status_code} - {response.text}",
            }
    except Exception as e:
        logger.error({"error": str(e)})
        st.session_state.operation_status = {
            "type": "error",
            "message": f"Failed to reload data: {str(e)}",
        }


def update_button():
    if st.button("Восстановить данные по умолчанию"):
        handle_reload()


update_button()


def get_sample_data(df_type: str) -> str:
    """Get sample data from backend"""
    try:
        response = requests.get(f"{api_url}/api/sample_data", params={"df_type": df_type})
        return response.json()["data"]
    except Exception as e:
        logger.exception(e)
        # Fallback sample data
        if df_type == "tickets":
            return "queueId;date;new_tickets\n1;2017-01-01;10"
        return "queue_id,level,immediateDescendants,allDescendants\n1,1,'[2;3]','[2;3;4]'"


def init_session_state():
    if "operation_status" not in st.session_state:
        st.session_state.operation_status = None
    if "last_upload" not in st.session_state:
        st.session_state.last_upload = None


def handle_upload(df_type: str, file):
    try:
        file.seek(0)
        files = {"file": file}
        data = {"df_type": df_type}

        with st.spinner(f"Uploading {df_type} data..."):
            response = requests.post(url=f"{api_url}/api/upload_data", files=files, data=data)

            if response.ok:
                st.session_state.operation_status = {
                    "type": "success",
                    "message": f"{df_type.title()} data uploaded successfully!",
                    "result": response.json() if response.content else None,
                }
                st.session_state.last_upload = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                st.session_state.operation_status = {
                    "type": "error",
                    "message": f"Upload failed: {response.json().get('detail', 'Unknown error')}",
                }
    except Exception as e:
        st.session_state.operation_status = {
            "type": "error",
            "message": f"Error during upload: {str(e)}",
        }


def display_status():
    if st.session_state.operation_status:
        if st.session_state.operation_status["type"] == "success":
            st.success(st.session_state.operation_status["message"])
        else:
            st.error(st.session_state.operation_status["message"])


def upload_section():
    init_session_state()
    st.header("Загрузка данных")

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Tickets Data Upload")
        with st.expander("Show sample format for tickets"):
            st.table(get_sample_data("tickets"))

        tickets_file = st.file_uploader(
            "Upload tickets CSV file (;-separated)", type=["csv"], key="tickets_upload"
        )

        if tickets_file and st.button("Upload Tickets Data"):
            handle_upload("tickets", tickets_file)
            fetch_all_queues.clear()
            st.rerun()

    with right_col:
        st.subheader("Hierarchy Data Upload")
        with st.expander("Show sample format for hierarchy"):
            st.table(get_sample_data("hierarchy"))

        hierarchy_file = st.file_uploader(
            "Upload hierarchy CSV file", type=["csv"], key="hierarchy_upload"
        )

        if hierarchy_file and st.button("Upload Hierarchy Data"):
            handle_upload("hierarchy", hierarchy_file)
            fetch_all_queues.clear()
            st.rerun()

    display_status()


if not st.session_state.data_available:
    st.warning("⚠️ Данные недоступны")
    st.markdown("""
        ## Как получить доступ к данным:
        ### Предустановка:
        1. `poetry install`
        2. установить `just`

        ### Хороший вариант через MiniO
        1. `just pull-data`

        ### Запасной вариант
        1. Убедитесь, что у вас есть доступ к репозиторию с данными
            https://drive.google.com/drive/folders/14b6lcjdD4IZNkyiVbwLm3H_2K3ZXt2HX?usp=sharing
        2. Скачайте данные из папки data
        3. Разместите их в локальном репозитории в папке `./data/`
        4. Установить just и запустите `just generate_data`
    """)
    st.stop()


@st.cache_data(ttl=10)
def fetch_queue_data(queue_id: int, start_date: str, end_date: str, granularity: TimeGranularity):
    response = requests.get(
        f"{api_url}/api/historical",
        params={
            "queue_id": queue_id,
            "start_date": start_date,
            "end_date": end_date,
            "granularity": granularity.value,
        },
    )
    return response.json()


def fetch_queue_structure(queue_id):
    # Simulate API call
    response = requests.get(f"{api_url}/api/structure/{queue_id}")
    return response.json()


def create_time_series_plot(df, granularity):
    fig = go.Figure()

    # Main time series
    fig.add_trace(go.Scatter(x=df.index, y=df["value"], name="Load", line={"color": "blue"}))

    # Add trend line
    df["MA7"] = df["value"].rolling(window=7).mean()
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MA7"], name="7-day MA", line={"color": "red", "dash": "dash"})
    )

    fig.update_layout(
        title="Queue Load Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Tickets",
        height=500,
    )
    return fig


def create_weekday_distribution(queue_id, date_start, date_end):
    weekday_response = requests.get(
        f"{api_url}/api/daily_average/",
        params={"queue_id": queue_id, "date_start": date_start, "date_end": date_end},
    )
    response = weekday_response.json()

    fig = go.Figure(data=[go.Bar(x=response["weekdays"], y=response["average_load"])])
    fig.update_layout(
        title="Средняя нагрузка по дням недели",
        xaxis_title="Day of Week",
        yaxis_title="Average Number of Tickets",
    )
    return fig


def create_weekly_distribution(start_date=None, end_date=None):
    try:
        params = {}
        if start_date and end_date:
            params["start_date"] = start_date
            params["end_date"] = end_date
            url = f"{api_url}/api/weekly_average/{selected_queue['id']}"
            weekly_response = requests.get(url=url, params=params)
        else:
            weekly_response = requests.get(f"{api_url}/api/weekly_average/{selected_queue['id']}")
        response = weekly_response.json()
        average_load = []
        for load in response["average_load"]:
            if load != 0:
                average_load.append(load)
        fig = go.Figure(data=[go.Bar(x=response["week"], y=average_load)])
        fig.update_layout(
            title="Average Weekly Load",
            xaxis_title="Week",
            yaxis_title="Average Number of Tickets",
        )
        return fig
    except requests.exceptions.HTTPError as http_err:
        error_detail = http_err.response.json().get("detail", "Unknown error")
        st.error(f"HTTP error occurred: {error_detail}")


def create_subqueues_stack_plot(data):
    fig = px.area(
        data, x="timestamp", y="value", color="subqueue", title="Load Distribution Across Subqueues"
    )
    return fig


def display_queue_stats(queue_stats: QueueStats):
    """Нарисовать статистику очереди."""
    st.header(f"Queue {queue_stats.queue_id} Analysis")

    st.subheader("Анализ структуры очереди")
    struct_col1, struct_col2, struct_col3 = st.columns(3)
    with struct_col1:
        st.metric("Уровень", queue_stats.structure.level)
        st.metric("Прямые наследники", queue_stats.structure.direct_children)
    with struct_col2:
        st.metric("Все наследники", queue_stats.structure.all_descendants)
        st.metric("Непрямые наследники", queue_stats.structure.leaf_nodes)
    with struct_col3:
        st.metric("Глубина очереди", queue_stats.structure.depth)

    st.subheader("Анализ нагрузки")
    load_col1, load_col2, load_col3 = st.columns(3)
    with load_col1:
        st.metric("Всего тикетов", f"{queue_stats.load.total_tickets:,}")
        st.metric("Средняя нагрузка", f"{queue_stats.load.avg_tickets:.1f}")
        st.metric("Средняя нагрузка (медиана)", f"{queue_stats.load.median_load:.1f}")
    with load_col2:
        st.metric("Пиковая нагрузка", queue_stats.load.peak_load)
        st.metric("Минимальная нагрузка", queue_stats.load.min_load)
    with load_col3:
        st.metric("90 перцентиль", f"{queue_stats.load.percentile_90:.1f}")
        st.metric("Стандартное отклонение", f"{queue_stats.load.std_dev:.1f}")

    st.subheader("Временные паттерны")
    time_col1, time_col2 = st.columns(2)
    with time_col1:
        st.metric(
            "Выходные / Будни",
            f"{queue_stats.time.weekend_avg:.1f} / {queue_stats.time.weekday_avg:.1f}",
        )
    with time_col2:
        st.metric("Самый загруженный день", queue_stats.time.busiest_day.strftime("%Y-%m-%d"))
        st.metric("Самый спокойный день", queue_stats.time.quietest_day.strftime("%Y-%m-%d"))


# upload_section()

queues = fetch_all_queues()


def fetch_queue_stats(queue_id: int, start_date, end_date, granularity: TimeGranularity):
    try:
        response = requests.get(
            f"{api_url}/api/queue_stats",
            params={
                "queue_id": queue_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "granularity": granularity.value,
            },
        )
        response.raise_for_status()

        return QueueStats.model_validate(response.json())

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching queue stats: {str(e)}")
        return None
    except ValueError as e:
        st.error(f"Error parsing response: {str(e)}")
        return None


def create_forecast_tab():  # noqa
    st.header("Прогнозирование временного ряда")

    if "forecasting_models" not in st.session_state:
        st.session_state.forecasting_models = {}

    if "model_metrics" not in st.session_state:
        st.session_state.model_metrics = {}

    if not st.session_state.selected_queue:
        st.warning("Пожалуйста, выберите очередь в боковой панели")
        return
    selected_queue = st.session_state.selected_queue

    queue_id = selected_queue["id"]

    # Настройки прогнозирования
    st.sidebar.markdown("### Настройки прогнозирования")
    forecast_horizon = st.sidebar.slider("Горизонт прогнозирования (дни)", 7, 365, 30)

    # Выбор временного интервала для отображения
    col1, col2 = st.columns(2)
    with col1:
        ...
        # history_days = st.slider("Исторический период (дни)", 7, 365, 90)

    # Панель моделей
    st.subheader("Выбор модели прогнозирования")

    model_tabs = st.tabs(
        [
            "Наивная модель",
            "Экспоненциальное сглаживание",
            "Prophet",
            "CatBoost",
            "Линейная регрессия",
            "Рекуррентная нейронная сеть",
        ]
    )

    model_types = ["naive", "es", "prophet", "catboost", "linear", "rnn"]
    model_names = {
        "naive": "Наивная модель",
        "es": "Экспоненциальное сглаживание",
        "prophet": "Prophet",
        "catboost": "CatBoost",
        "linear": "Линейная регрессия",
        "rnn": "Рекуррентная нейронная сеть",
    }

    # Фиксированная дата для демонстрации
    train_start_date = datetime(2019, 9, 1)
    forecast_start_date = datetime(2020, 1, 1)
    forecast_end_date = forecast_start_date + timedelta(days=forecast_horizon)

    st.info(
        f"Прогноз с {forecast_start_date.strftime('%d.%m.%Y')} на {forecast_horizon} дней вперед"
    )

    train_data = fetch_queue_data(
        queue_id,
        train_start_date.strftime("%Y-%m-%d"),  # От самого начала
        (forecast_start_date - timedelta(days=1)).strftime("%Y-%m-%d"),
        TimeGranularity.DAILY,
    )

    test_data = fetch_queue_data(
        queue_id,
        forecast_start_date.strftime("%Y-%m-%d"),
        forecast_end_date.strftime("%Y-%m-%d"),
        TimeGranularity.DAILY,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train_data["timestamps"],
            y=train_data["values"],
            name="Данные для обучения",
            line={"color": "blue"},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_data["timestamps"],
            y=test_data["values"],
            name="Фактические данные",
            line={"color": "black", "dash": "dot"},
        )
    )

    for i, (tab, model_type) in enumerate(zip(model_tabs, model_types)):
        with tab:
            st.markdown(f"### {model_names[model_type]}")

            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown("Описание модели:")
                if model_type == "naive":
                    st.info(
                        "Наивная сезонная модель использует повторение последнего сезонного паттерна. Простая, но эффективная для данных с явной сезонностью."  # noqa: E501
                    )
                elif model_type == "es":
                    st.info(
                        "Экспоненциальное сглаживание учитывает тренд и сезонность, давая больший вес недавним наблюдениям. Хорошо работает для данных с относительно стабильным поведением."  # noqa: E501
                    )
                elif model_type == "prophet":
                    st.info(
                        "Prophet - модель от Facebook, спроектированная для обработки сезонных данных с выбросами. Хорошо справляется с отсутствующими данными и изменениями трендов."  # noqa: E501
                    )
                elif model_type == "catboost":
                    st.info(
                        "CatBoost - градиентный бустинг деревьев решений с использованием временных признаков. Отлично обнаруживает нелинейные зависимости."  # noqa: E501
                    )
                elif model_type == "linear":
                    st.info(
                        "Линейная регрессия с временными признаками - простая, но эффективная модель для рядов с линейным трендом."  # noqa: E501
                    )
                elif model_type == "rnn":
                    st.info("Рекуррентная нейронная сеть. Хорошо справляется с нелинейностями.")

            with col2:
                train_button = st.button("Обучить модель", key=f"train_{model_type}")

            if (
                queue_id in st.session_state.model_metrics
                and model_type in st.session_state.model_metrics[queue_id]
            ):
                metrics = st.session_state.model_metrics[queue_id][model_type]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                with col2:
                    st.metric("MAE", f"{metrics['mae']:.2f}")
                with col3:
                    st.metric("MAPE", f"{metrics['mape']:.2f}%")

            if train_button:
                with st.spinner(f"Обучение модели {model_names[model_type]}..."):
                    try:
                        train_end_date = forecast_start_date - timedelta(days=1)
                        train_start_date = train_end_date - timedelta(days=365)  # Год для обучения
                        request_data = {
                            "queue_id": queue_id,
                            "forecast_horizon": forecast_horizon,
                            "model_type": model_type,
                            "train_start_date": train_start_date.strftime("%Y-%m-%d"),
                            "train_end_date": train_end_date.strftime("%Y-%m-%d"),
                            "forecast_start_date": forecast_start_date.strftime("%Y-%m-%d"),
                        }
                        response = requests.post(
                            f"{api_url}/api/train_model",
                            json=request_data,
                        )

                        if response.status_code == 200:  # noqa: PLR2004
                            metrics = response.json()

                            if queue_id not in st.session_state.model_metrics:
                                st.session_state.model_metrics[queue_id] = {}

                            st.session_state.model_metrics[queue_id][model_type] = metrics

                            pred_response = requests.post(
                                f"{api_url}/api/forecast",
                                json={
                                    "queue_id": queue_id,
                                    "forecast_horizon": forecast_horizon,
                                    "model_type": model_type,
                                    "forecast_start_date": forecast_start_date.strftime("%Y-%m-%d"),
                                },
                            )

                            if pred_response.status_code == 200:  # noqa: PLR2004
                                pred_data = pred_response.json()

                                if queue_id not in st.session_state.forecasting_models:
                                    st.session_state.forecasting_models[queue_id] = {}

                                st.session_state.forecasting_models[queue_id][model_type] = (
                                    pred_data
                                )

                                st.success(f"Модель {model_names[model_type]} успешно обучена!")
                                st.rerun()
                        else:
                            st.error(f"Ошибка при обучении модели: {response.text}")

                    except Exception as e:
                        st.error(f"Произошла ошибка: {str(e)}")

    if queue_id in st.session_state.forecasting_models:
        st.subheader("Сравнение прогнозов")

        colors = {
            "naive": "red",
            "es": "green",
            "prophet": "orange",
            "catboost": "purple",
            "linear": "brown",
            "rnn": "blue",
        }

        for model_type, pred_data in st.session_state.forecasting_models[queue_id].items():
            fig.add_trace(
                go.Scatter(
                    x=pred_data["timestamps"],
                    y=pred_data["values"],
                    name=f"Прогноз ({model_names[model_type]})",
                    line={"color": colors[model_type]},
                )
            )

    fig.update_layout(
        title=f"Прогноз для очереди {queue_id}",
        xaxis_title="Дата",
        yaxis_title="Количество тикетов",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    st.plotly_chart(fig, use_container_width=True)

    if (
        queue_id in st.session_state.model_metrics
        and len(st.session_state.model_metrics[queue_id]) > 0
    ):
        st.subheader("Сравнение метрик моделей")

        metrics_data = []
        for model_type, metrics in st.session_state.model_metrics[queue_id].items():
            metrics_data.append(
                {
                    "Модель": model_names[model_type],
                    "RMSE": f"{metrics['rmse']:.2f}",
                    "MAE": f"{metrics['mae']:.2f}",
                    "MAPE (%)": f"{metrics['mape']:.2f}",
                }
            )

        metrics_df = pd.DataFrame(metrics_data)  # move pandas out of here
        st.dataframe(metrics_df, use_container_width=True)

    # Кнопка для очистки моделей
    st.sidebar.markdown("### Управление моделями")
    if st.sidebar.button("Очистить все модели"):
        try:
            response = requests.delete(f"{api_url}/api/clear_models/{queue_id}")
            if response.status_code == 200:  # noqa: PLR2004
                st.session_state.forecasting_models.pop(queue_id, None)
                st.session_state.model_metrics.pop(queue_id, None)
                st.success("Модели успешно очищены")
                st.rerun()
            else:
                st.error(f"Ошибка при очистке моделей: {response.text}")
        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")


tab1, tab2, tab3 = st.tabs(["Анализ очереди", "Загрузка данных", "Прогнозирование"])


with tab2:
    upload_section()

with tab3:
    create_forecast_tab()
with tab1:
    if queues and st.session_state.selected_queue:
        selected_queue = st.session_state.selected_queue

        st.sidebar.markdown("### Настройки ряда")
        start_date = datetime(2018, 1, 1) - timedelta(days=1000)
        end_date = datetime(2018, 1, 1) + timedelta(days=1000)
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(start_date, end_date),
            max_value=datetime.now(),
        )
        if len(date_range) == 2:  # noqa: PLR2004
            start_date, end_date = date_range
        elif len(date_range) == 1:
            start_date = date_range[0]

        GRANULARITY_DISPLAY = {
            "[D] Daily": TimeGranularity.DAILY,
            "[W] Weekly": TimeGranularity.WEEKLY,
            "[M] Monthly": TimeGranularity.MONTHLY,
        }

        selected_granularity = st.sidebar.selectbox(
            "Time Granularity", options=list(GRANULARITY_DISPLAY.keys()), key="granularity"
        )
        granularity = GRANULARITY_DISPLAY[selected_granularity]

        if selected_queue["id"] == 0:
            st.markdown("### Статистика для общей нагрузки пока не реализована")
        else:
            queue_stats = fetch_queue_stats(
                queue_id=selected_queue["id"],
                start_date=start_date,
                end_date=end_date,
                granularity=granularity,
            )
            display_queue_stats(queue_stats)
        if selected_queue:
            try:
                hist_data = fetch_queue_data(
                    selected_queue["id"],
                    start_date,
                    end_date,
                    granularity,
                )
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=hist_data["timestamps"],
                        y=hist_data["values"],
                        name="Исторические данные",
                        line={"color": "blue"},
                    )
                )

                if granularity is TimeGranularity.DAILY:
                    st.plotly_chart(
                        create_weekday_distribution(selected_queue["id"], start_date, end_date)
                    )

                fig.update_layout(
                    title=f"Queue {selected_queue['id']} - Исторически данные",
                    xaxis_title="Date",
                    yaxis_title="Number of Tickets",
                    hovermode="x unified",
                )

                st.plotly_chart(fig, use_container_width=True)

            except requests.exceptions.RequestException as e:
                logger.error(e)
                st.error(f"Error fetching data: {str(e)}")
    else:
        logger.error("No queues available")
        st.error("No queues available")
