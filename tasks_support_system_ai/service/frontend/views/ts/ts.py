import time
from http import HTTPStatus

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "http://backend:8000"

st.title("Анализ нагрузки очередей")


def wait_for_backend(max_retries=30, delay=1):
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/api/data-status")
            if response.status_code == HTTPStatus.OK:
                return True
        except requests.RequestException:
            pass
        print(f"Waiting for backend... attempt {i+1}/{max_retries}")
        time.sleep(delay)
    return False


if not wait_for_backend():
    st.error("Backend service is not available")
    st.stop()


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
    st.warning("⚠️ Данные временно недоступны")
    st.markdown("""
        ## Как получить доступ к данным:
        ### Предустановка:
        1. `poetry install`
        2. установить `just`

        ### Хороший вариант через MiniO
        1. `just pull-data`

        ### Запасной вариант
        1. Убедитесь, что у вас есть доступ к репозиторию с данными https://drive.google.com/drive/folders/14b6lcjdD4IZNkyiVbwLm3H_2K3ZXt2HX?usp=sharing
        2. Скачайте данные из папки data
        3. Разместите их в локальном репозитории в папке `./data/`
        4. Установить just и запустите `just generate_data`
    """)
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

    if selected_queue:
        try:
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

            # Display metrics
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

            if st.button("Download Data"):
                import pandas as pd

                df = pd.DataFrame({"Date": hist_data["timestamps"], "Actual": hist_data["values"]})
                if show_prediction:
                    pred_df = pd.DataFrame(
                        {
                            "Date": pred_data["timestamps"],
                            "Predicted": pred_data["values"],
                        }
                    )
                    df = pd.concat([df, pred_df], axis=0)

                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    f"queue_{selected_queue['id']}_data.csv",
                    "text/csv",
                    key="download-csv",
                )

        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data: {str(e)}")
else:
    st.error("No queues available")

with st.expander("About this dashboard"):
    st.write("""
    Историческая нагрузка на очереди
    """)
