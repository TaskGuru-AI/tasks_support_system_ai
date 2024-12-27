import numpy as np
from tasks_support_system_ai.web.tabs import nlp_overview
import streamlit as st

tab1, tab2, tab3 = st.tabs(["Project info", "🗃️ Train", "📊 Prediction"])
data = np.random.randn(10, 1)

with tab1:
    nlp_overview()

tab2.subheader("Описание датасета")
tab2.write(data)

tab3.subheader("Модель классификации")
tab3.line_chart(data)