import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://backend:8000/ts"

st.title("Анализ нагрузки очередей")


if "data_available" not in st.session_state:
    st.session_state.data_available = False


# @st.cache_data(ttl=600) # better to cache good result
def check_data_availability():
    try:
        response = requests.get(f"{API_URL}/api/data-status")
        return response.json()["has_data"]
    except requests.exceptions.RequestException:
        return False


st.session_state.data_available = check_data_availability()

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
    # TODO: add buttons to upload data (should be reused)
    # add option to download data from miniO
    st.stop()


@st.cache_data(ttl=600)
def fetch_queues():
    try:
        response = requests.get(f"{API_URL}/api/queues")
        return response.json()["queues"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching queues: {str(e)}")
        return []


queues = fetch_queues()

if queues:
    selected_queue = st.sidebar.selectbox(
        "Выберите очередь",
        options=queues,
        format_func=lambda x: f"Queue {x['id']} (Load: {x['load']})",
        key="queue_selector",
    )
    st.sidebar.markdown("### Настройки прогноза")
    days_ahead = st.sidebar.slider("Дней вперед", 1, 300, 25)
    show_prediction = st.sidebar.checkbox("Показать прогноз", value=True)
    # add parameters:
    # date start, date end
    # ts granularity

    if selected_queue:
        try:
            # EDA PART
            # make plot
            # make table with hierarchy stats
            # make table with time series stats

            hist_response = requests.get(f"{API_URL}/api/historical/{selected_queue['id']}")
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

            if show_prediction:
                with st.spinner("Генерация прогноза..."):
                    pred_response = requests.post(
                        f"{API_URL}/api/forecast",
                        json={
                            "queue_id": selected_queue["id"],
                            "days_ahead": days_ahead,
                        },
                    )
                    pred_data = pred_response.json()["forecast"]

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
            st.error(f"Error fetching data: {str(e)}")
else:
    st.error("No queues available")

with st.expander("About this dashboard"):
    st.write("""
    Историческая нагрузка на очереди
    """)
