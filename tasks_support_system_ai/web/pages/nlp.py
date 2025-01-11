import numpy as np
import streamlit as st

from tasks_support_system_ai.core.logger import streamlit_logger as logger
from tasks_support_system_ai.web.tabs.nlp_overview import show_overview
from tasks_support_system_ai.web.tabs.nlp_predict import render_predict_tab
from tasks_support_system_ai.web.tabs.nlp_train import render_train_tab

logger.info("NLP page is started loading")

tab1, tab2, tab3 = st.tabs(["Project info", "🗃️ Train", "📊 Prediction"])
data = np.random.randn(10, 1)

with tab1:
    show_overview()

with tab2:
    render_train_tab()

with tab3:
    render_predict_tab()
