# import asyncio
import logging

# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor
# from pathlib import Path
#
# import joblib
# import numpy as np
# from tasks_support_system_ai.data.reader import DataFrames, read_data
# from tasks_support_system_ai.utils.utils import data_checker, get_correct_data_path
import uuid

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
from tasks_support_system_ai.api.models.nlp import LogisticConfig, SVMConfig
from tasks_support_system_ai.utils.nlp import save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NLPPredictor:
    def __init__(self, data):
        super().__init__()
        """
        Initialize NLPPredictor with data.
        :param data: The data to be used for predictions.
        """
        self.train = data

    def predict(self):
        """
        Stub for prediction logic.
        """
        # Add prediction logic here
        raise NotImplementedError("Prediction logic is not implemented yet.")

    def train(self, model: str, config: LogisticConfig | SVMConfig) -> str:
        """
        Train a model based on the given type and configuration.
        :param model: Model type ("logistic" or "svm").
        :param config: Configuration for the model.
        :return: Training result as a string.
        """
        if model == "logistic":
            return train_logistic_model(self.train, config)
        elif model == "svm":
            return train_svm_model(self.train, config)
        else:
            raise ValueError("Invalid model name")


def train_logistic_model(data: pd.DataFrame, config: LogisticConfig) -> str:
    """
    Method to train a Logistic Regression model
    :param config: Model configuration
    Returns: str: ID model
    """
    print("data")
    X_train, y_train = data["vector"], data["cluster"]
    # X_test, y_test = data["vector"], data["cluster"]
    model = LogisticRegression(C=config.C, solver=config.solver)
    model.fit(X_train, y_train)
    model_id = str(uuid.uuid4())
    # y_pred = model.predict(X_test)
    save_model(model, f"data/{model_id}.pickle")

    # roc_auc = roc_auc_score(y_test, y_pred)
    # fpr, tpr, _ = roc_curve(y_test, y_pred)
    # statistics = {
    #     "roc_auc": roc_auc,
    #     "classification_report": classification_report(y_test, y_pred),
    #     "fpr": fpr,
    #     "tpr": tpr,
    # }
    return model_id


def train_svm_model(data: pd.DataFrame, config: LogisticConfig) -> str:
    """ "
    Method to train an SVM model
    :param config: Model configuration
    Returns: str: ID model
    """
    X_train, y_train = data["vector"], data["cluster"]
    # X_test, y_test = data["vector"], data["cluster"]
    model = SVC(C=config.C, kernel=config.kernel, class_weight=config.class_weight)
    model.fit(X_train, y_train)
    model_id = str(uuid.uuid4())
    # y_pred = model.predict(X_test)
    save_model(model, f"data/{model_id}.pickle")

    # roc_auc = roc_auc_score(y_test, y_pred)
    # fpr, tpr, _ = roc_curve(y_test, y_pred)
    # statistics = {
    #     "roc_auc": roc_auc,
    #     "classification_report": classification_report(y_test, y_pred),
    #     "fpr": fpr,
    #     "tpr": tpr,
    # }
    return model_id
