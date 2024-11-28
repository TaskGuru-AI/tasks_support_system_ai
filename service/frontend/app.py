# this is an entry point

import streamlit as st

st.set_page_config(
    page_icon="🤖",
    page_title="Анализ обращений",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Это приложение команды 34. Мы разработали систему анализа задач службы технической поддержки с авторекомендациями"
    }
)
# st.write('# Система анализа задач службы технической поддержки с авторекомендациями')

ts_page = st.Page("views/ts/ts.py", title="TS", icon=":material/calendar_clock:")
nlp_page = st.Page("views/nlp/nlp.py", title="NLP", icon=":material/notes:")

pg = st.navigation([ts_page, nlp_page])

pg.run()
