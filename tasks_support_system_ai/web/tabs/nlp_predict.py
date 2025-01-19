import io
import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = "http://backend:8000/nlp"

if os.getenv("IS_DOCKER", "0") == "0":
    BACKEND_URL = "http://localhost:8000/nlp"

CLUSTERS_MAP = {
    1: "Списания/денежные средства",
    2: "Проблемы с сим-меню",
    3: "Esim (проблемы подключения, код активации, qr-код)",
    4: "Проблемы с контентом и отключение услуг/ подписок/ сервисов",
    5: "Учетные записи",
    6: "Уведомления о технических работах",
    7: "Сбой работы (в работе оператора)",
    8: "Тесты, проблемы с доставкой, прочие тех. оповещения",
    9: "Черный список",
    10: "Спам, реклама",
}


def nlp_display_collected_params(*args):
    st.success("Текущие параметры предсказания:")
    st.success(args)


def predict_pick_model():
    if st.session_state.model_ids:
        st.session_state.picked_model = st.selectbox(
            label="Выберите модель для предсказания:",
            options=st.session_state.model_ids,
            index=None,
            key="nlp_pick_model",
        )
    if st.session_state.picked_model:
        st.success(
            f"Выбранные параметры предсказания:\n"
            f"{st.session_state.model_configs[st.session_state.picked_model]}"
        )


def nlp_show_input_options():
    if st.session_state.model_ids:
        st.session_state.nlp_predict_input_text = st.text_area(
            label="Текст:", help="Впишите текст обращения"
        )
        st.session_state.nlp_predict_input_file_uploader = st.file_uploader(
            label="CSV:", type=["csv"], help="Добавьте CSV-файл"
        )


def nlp_predict_from_text(model_id, input_text):
    with st.spinner("Предсказываем..."):
        try:
            response = requests.post(
                url=f"{BACKEND_URL}/api/predict_nlp", json={"id": model_id, "text": input_text}
            )
            st.session_state["cluster_from_text"] = response.json()["clusters"]

        except Exception as e:
            st.error(f"Ошибка при формировании предсказания из текста: {e}")


def nlp_predict_from_csv(model_id, input_file):
    with st.spinner("Предсказываем..."):
        try:
            response = requests.post(
                url=f"{BACKEND_URL}/api/predict_nlp_csv?id={model_id}", files={"file": input_file}
            )
            st.session_state["cluster_from_csv"] = response.content.decode("utf-8")
            st.session_state["formatted_cluster_from_csv"] = pd.read_csv(
                io.StringIO(st.session_state["cluster_from_csv"]), sep=","
            )
            st.session_state["formatted_cluster_from_csv"]["prediction_desc"] = st.session_state[
                "formatted_cluster_from_csv"
            ]["prediction"].map(CLUSTERS_MAP)

        except Exception as e:
            st.error(f"Ошибка при формировании предсказания из файла: {e}")


def nlp_predict_buttons():
    if st.session_state.model_ids:
        nlp_predict_button_column1, nlp_predict_button_column2 = st.columns(2)
        with nlp_predict_button_column1:
            if st.button(label="Предсказать на основе текста", type="primary"):
                nlp_predict_from_text(
                    st.session_state.picked_model, st.session_state.nlp_predict_input_text
                )
        with nlp_predict_button_column2:
            if st.button(label="Предсказать на основе CSV-файла", type="primary"):
                nlp_predict_from_csv(
                    st.session_state.picked_model, st.session_state.nlp_predict_input_file_uploader
                )


def nlp_display_result_text():
    if "cluster_from_text" in st.session_state:
        if len(st.session_state["cluster_from_text"]) == 1:
            with st.expander(label="Для введённого текста был определён следующий кластер:"):
                st.write(f"{CLUSTERS_MAP[st.session_state["cluster_from_text"][0]]}")
        else:
            with st.expander(label="Для введённого текста были определены следующие кластеры:"):
                for class_from_text in st.session_state["cluster_from_text"]:
                    st.write(f"{CLUSTERS_MAP[class_from_text]}")


def nlp_display_result_csv():
    if "cluster_from_csv" in st.session_state:
        with st.expander(label="Для загруженного файла классы были определены так:"):
            st.dataframe(
                st.session_state["formatted_cluster_from_csv"],
                hide_index=True,
                use_container_width=True,
            )
            st.download_button(
                label="Скачать",
                data=st.session_state["formatted_cluster_from_csv"]
                .to_csv(index=False)
                .encode("utf-8"),
                file_name="nlp_predicted_from_csv.csv",
                mime="text/csv",
            )


def render_predict_tab():
    if "picked_model" not in st.session_state:
        st.session_state.picked_model = ""
    if "nlp_selected_input" not in st.session_state:
        st.session_state.nlp_selected_input = ""
    if "nlp_predict_input_text" not in st.session_state:
        st.session_state.predict_input_text = ""
    if "nlp_predict_input_file_uploader" not in st.session_state:
        st.session_state.nlp_predict_input_file_uploader = ""
    if not st.session_state.model_ids:
        st.subheader(body="Для выполнения предсказания обучите хотя бы одну модель")
    predict_pick_model()
    # nlp_input_selector()
    nlp_show_input_options()
    nlp_predict_buttons()
    nlp_display_result_text()
    nlp_display_result_csv()
