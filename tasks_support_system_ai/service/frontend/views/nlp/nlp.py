import streamlit as st
import numpy as np

tab1, tab2 = st.tabs(["🗃️ Датасет", "📊 Классификация"])
data = np.random.randn(10, 1)

tab1.subheader("Описание датасета")
tab1.write(data)

tab2.subheader("Модель классификации")
tab2.line_chart(data)
