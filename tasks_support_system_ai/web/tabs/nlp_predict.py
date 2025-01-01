import os

import pandas as pd
import streamlit as st

BACKEND_URL = "http://backend:8000/nlp"

if os.getenv("IS_DOCKER", "0") == "0":
    BACKEND_URL = "http://localhost:8000/nlp"


def nlp_display_collected_params(*args):
    st.success("Текущие параметры предсказания:")
    st.success(args)


def predict_pick_model():
    if st.session_state.model_ids:
        st.session_state.picked_model = st.selectbox(
            label="Выберите модель для предсказания:",
            options=st.session_state.model_configs.values(),
            index=None,
            key="nlp_pick_model"
        )


def nlp_input_selector():
    if st.session_state.model_ids:
        st.session_state.nlp_selected_input = st.radio(
            label="Выберите, с чем работать:",
            options=["Текст", "CSV"],
            index=0,
            horizontal=True
        )

def nlp_show_input_options():
    if st.session_state.model_ids:
        st.session_state.nlp_predict_input_text = st.text_area(
            label="Текст:",
            help="Впишите текст обращения"
        )
        st.session_state.nlp_predict_input_file_uploader = st.file_uploader(
            label="CSV:",
            type=["csv", "py"],
            help="Добавьте CSV-файл"
        )


def nlp_predict_clear_buttons():
    if st.session_state.model_ids:
        st.button(
            label=f"Предсказать с источника: {st.session_state.nlp_selected_input}",
            type="primary",
            on_click=nlp_display_collected_params,
            args=(st.session_state.picked_model, st.session_state.nlp_selected_input)
        )
        st.button(
            label="Сбросить ввод",
            type="secondary"
        )

nlp_sample_results = st.table(
    pd.DataFrame(
        {
            "Text": ["Please buy this cream", "I want a new SIM card", "It doesn't work"],
            "Cluster": ["Spam", "Request", "Tech work"],
        }
    )
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
        st.subheader(
            body="Для выполнения предсказания обучите хотя бы одну модель"
        )
    predict_pick_model()
    nlp_input_selector()
    nlp_show_input_options()
    nlp_predict_clear_buttons()

