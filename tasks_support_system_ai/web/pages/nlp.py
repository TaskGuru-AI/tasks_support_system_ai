import logging

import numpy as np
import pandas as pd
import streamlit as st

from tasks_support_system_ai.web.tabs.nlp_overview import show_overview
from tasks_support_system_ai.web.tabs.nlp_train import render_train_tab


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tab1, tab2, tab3 = st.tabs(["Project info", "🗃️ Train", "📊 Prediction"])
data = np.random.randn(10, 1)

with tab1:
    show_overview()

with tab2:
    render_train_tab()

#TODO: reformat this so it looks like in Feodosiya's code

# TAB 3 SECTION - PREDICT



# NEEDED FUNCTIONS
# def nlp_determine_input_selector():
#     if nlp_input_selector == "Text":
#         nlp_predict_input = nlp_predict_form.text_area(label="Text:",
#                                                        help="Provide text")
#     if nlp_input_selector == "CSV":
#         nlp_predict_input = nlp_predict_form.file_uploader(label="CSV:",
#                                                            type=["csv"])

def nlp_display_collected_params(*args):
    tab3.write("These are the parameters stored in Predict function now:")
    tab3.write(args)


# nlp_predict_form = tab3.form(key="nlp_predict_form",
#                              clear_on_submit=False,
#                              enter_to_submit=True,
#                              border=True)

nlp_predict_model_selector = tab3.selectbox(label="Choose model:",
                                            placeholder="Choose model",
                                            options=["model_id_1", "model_id_2"],
                                            index=None) # This should be the list of models

nlp_input_selector = tab3.radio(label="Text or CSV file:",
                                            options=["Text", "CSV"],
                                            index=0,
                                            horizontal=True)

# Kept both text input and file selector as user may decide mid-action they need them

nlp_predict_input_text = tab3.text_area(label="Text:",
                                        help="Please provide text of the conversation")

nlp_predict_input_file_uploader = tab3.file_uploader(label="CSV:",
                                                     type=["csv", "py"],
                                                     help="Please provide a CSV file")

nlp_predict_button = tab3.button(label=f"Predict from {nlp_input_selector}",
                                 type="primary",
                                 on_click=nlp_display_collected_params, # This calls API
args=(nlp_predict_model_selector, nlp_input_selector, nlp_predict_input_text))

nlp_clear_form_button = tab3.button(label="Clear form",
                                    type="secondary")

nlp_prediction_results = tab3.table(pd.DataFrame({"Text": ["Please buy this cream",
                                                           "I want a new SIM card",
                                                           "It doesn't work"],
                                                  "Cluster": ["Spam", "Request", "Tech work"]}))



# tab3.subheader("Модель классификации")
# tab3.line_chart(data)
