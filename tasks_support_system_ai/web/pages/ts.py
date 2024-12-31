from datetime import datetime, timedelta

import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from tasks_support_system_ai.api.models.ts import ForecastRequest
from tasks_support_system_ai.core.config import settings
from tasks_support_system_ai.core.logger import streamlit_logger as logger

EXPECTED_DATE_RANGE_LENGTH = 2

st.title("Анализ нагрузки очередей")

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


st.session_state.data_available = check_data_availability()


st.write("Загрузите файлы CSV с данными о тикетах и иерархии.")


def handle_reload():
    try:
        response = requests.get(f"{api_url}/api/reload_local_data")

        if response.ok:
            st.session_state.operation_status = {
                "type": "success",
                "message": "Data reloaded successfully!",
                "result": response.json(),
            }
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
    if st.button("Reload Local Data"):
        handle_reload()
        st.rerun()


update_button()


def get_sample_data(df_type: str) -> str:
    """Get sample data from backend"""
    try:
        response = requests.get(f"{api_url}/api/sample_data", params={"df_type": df_type})
        return response.json()["data"]
    except Exception:
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
    st.header("Data Upload")

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Tickets Data Upload")
        # with st.expander("Show sample format for tickets"):
        #     st.table(get_sample_data("tickets"))

        tickets_file = st.file_uploader(
            "Upload tickets CSV file (;-separated)", type=["csv"], key="tickets_upload"
        )

        if tickets_file and st.button("Upload Tickets Data"):
            handle_upload("tickets", tickets_file)
            st.rerun()

    with right_col:
        st.subheader("Hierarchy Data Upload")
        # with st.expander("Show sample format for hierarchy"):
        #     st.table(get_sample_data("hierarchy"))

        hierarchy_file = st.file_uploader(
            "Upload hierarchy CSV file", type=["csv"], key="hierarchy_upload"
        )

        if hierarchy_file and st.button("Upload Hierarchy Data"):
            handle_upload("hierarchy", hierarchy_file)
            st.rerun()

    display_status()

    # не работает, и сообщение об успехе постоянно висит
    # if st.session_state.operation_status:
    #     time.sleep(2)  # Show message for 2 seconds
    #     st.session_state.operation_status = None


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


# @st.cache_data(ttl=600)
def fetch_queues():
    try:
        response = requests.get(f"{api_url}/api/queues")
        return response.json()["queues"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching queues: {str(e)}")
        return []


def fetch_queue_data(queue_id, start_date, end_date, granularity):
    # Simulate API call - replace with actual endpoint
    response = requests.get(
        f"{api_url}/api/historical/{queue_id}",
        params={"start_date": start_date, "end_date": end_date, "granularity": granularity},
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


def create_weekday_distribution():
    weekday_response = requests.get(f"{api_url}/api/daily_average/{selected_queue['id']}")
    response = weekday_response.json()

    fig = go.Figure(data=[go.Bar(x=response["weekdays"], y=response["average_load"])])
    fig.update_layout(
        title="Average Load by Weekday",
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


queues = fetch_queues()

if queues:
    selected_queue = st.sidebar.selectbox(
        "Выберите очередь",
        options=queues,
        format_func=lambda x: f"Queue {x['id']} (Load: {x['load']})",
        key="queue_selector",
    )
    st.sidebar.header("Controls")
    st.sidebar.markdown("### Настройки прогноза")
    days_ahead = st.sidebar.slider("Дней вперед", 1, 300, 25)
    show_prediction = st.sidebar.checkbox("Показать прогноз", value=True)

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(datetime(2018, 1, 1) - timedelta(days=30), datetime(2018, 1, 1)),
        max_value=datetime.now(),
    )

    # Извлечение начальной и конечной даты
    if isinstance(date_range, tuple) and len(date_range) == EXPECTED_DATE_RANGE_LENGTH:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    granularity = st.sidebar.selectbox(
        "Time Granularity", options=["Daily", "Weekly", "Monthly"], key="granularity"
    )
    upload_section()

    if selected_queue:
        try:
            # EDA PART
            # make plot
            # make table with hierarchy stats
            # make table with time series stats

            hist_response = requests.get(f"{api_url}/api/historical/{selected_queue['id']}")
            hist_data = hist_response.json()

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=hist_data["timestamps"],
                    y=hist_data["values"],
                    name="Исторические данные",
                    line={"color": "blue"},
                )
            )
            # queue_data = fetch_queue_data(
            #     selected_queue["id"], date_range[0], date_range[1], granularity.lower()
            # )

            # # Convert to DataFrame
            # df = pd.DataFrame(queue_data)
            # df["timestamp"] = pd.to_datetime(df["timestamps"])
            # df.set_index("timestamp", inplace=True)

            # # Time series plot
            # st.plotly_chart(create_time_series_plot(df, granularity))

            # Weekday distribution (only for daily data)
            if granularity == "Daily":
                st.plotly_chart(create_weekday_distribution())

            if granularity == "Weekly":
                if date_range:
                    st.plotly_chart(
                        create_weekly_distribution(start_date=start_date, end_date=end_date)
                    )
                else:
                    st.plotly_chart(create_weekly_distribution())

            # if len(df) > 14:  # Minimum required for decomposition
            #     decomposition = seasonal_decompose(df["value"], period=7)

            #     fig = go.Figure()
            #     fig.add_trace(go.Scatter(x=df.index, y=decomposition.trend, name="Trend"))
            #     fig.add_trace(go.Scatter(x=df.index, y=decomposition.seasonal, name="Seasonal"))
            #     fig.add_trace(go.Scatter(x=df.index, y=decomposition.resid, name="Residual"))
            #     fig.update_layout(title="Time Series Decomposition")
            #     st.plotly_chart(fig)

            # # Queue structure visualization
            # structure = fetch_queue_structure(selected_queue["id"])
            # st.subheader("Queue Structure")
            # st.json(structure)  # You might want to create a better visualization

            # # Top subqueues stacked plot
            # # Assuming you have an endpoint that returns subqueue data
            # subqueues_data = fetch_subqueues_data(
            #     selected_queue["id"]
            # )  # You'll need to implement this
            # if subqueues_data:
            #     st.plotly_chart(create_subqueues_stack_plot(subqueues_data))

            if show_prediction:
                with st.spinner("Генерация прогноза..."):
                    pred_response = requests.post(
                        f"{api_url}/api/forecast",
                        json=ForecastRequest(
                            queue_id=selected_queue["id"],
                            forecast_horizon=days_ahead,
                        ).model_dump(),
                    )
                    pred_data = pred_response.json()

                    fig.add_trace(
                        go.Scatter(
                            x=pred_data["timestamps"],
                            y=pred_data["values"],
                            name="Прогноз",
                            line={"color": "red"},
                        )
                    )

            fig.update_layout(
                title=f"Queue {selected_queue['id']} - Historical Data and Forecast",
                xaxis_title="Date",
                yaxis_title="Number of Tickets",
                hovermode="x unified",
            )

            # ML part
            # TODO: add models to choose, some parameters and hyperparameters
            # models could be reused
            # Display metrics (TODO: add more metrics)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Load", f"{selected_queue['load']:,}")
            with col2:
                if hist_data["values"]:
                    avg_load = sum(hist_data["values"]) / len(hist_data["values"])
                    st.metric("Average Daily Load", f"{avg_load:.1f}")
            with col3:
                if show_prediction and pred_data["values"]:
                    pred_avg = sum(pred_data["values"]) / len(pred_data["values"])
                    change = ((pred_avg - avg_load) / avg_load) * 100
                    st.metric("Predicted Average Load", f"{pred_avg:.1f}", f"{change:+.1f}%")

            st.plotly_chart(fig, use_container_width=True)

        except requests.exceptions.RequestException as e:
            logger.error(e)
            st.error(f"Error fetching data: {str(e)}")
else:
    logger.error("No queues available")
    st.error("No queues available")

with st.expander("About this dashboard"):
    st.write("""
    Историческая нагрузка на очереди
    """)
