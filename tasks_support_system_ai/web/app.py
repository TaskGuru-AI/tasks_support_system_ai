"""Streamlit app entry point."""

import streamlit as st

from tasks_support_system_ai.core.logger import streamlit_logger as logger

st.set_page_config(
    page_icon="🤖",
    page_title="Анализ обращений",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Это приложение команды 34. Мы разработали систему анализа задач службы "
        + "технической поддержки с авторекомендациями"
    },
)

ts_page = st.Page("pages/ts.py", title="TS", icon=":material/calendar_clock:")
nlp_page = st.Page("pages/nlp.py", title="NLP", icon=":material/notes:")

pg = st.navigation([ts_page, nlp_page])

logger.info("streamlit is started")
pg.run()
