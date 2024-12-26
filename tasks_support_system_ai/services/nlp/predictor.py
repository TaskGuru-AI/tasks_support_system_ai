# import asyncio
import logging
from ast import literal_eval
# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor
# from pathlib import Path
#
# import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.multiclass import OneVsRestClassifier
# from tasks_support_system_ai.data.reader import DataFrames, read_data
# from tasks_support_system_ai.utils.utils import data_checker, get_correct_data_path
from tasks_support_system_ai.data.nlp.reader import NLPTicketsData
from tasks_support_system_ai.utils.nlp import vector_transform
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
    def __init__(self, data: NLPTicketsData):
        """
        Initialize NLPPredictor with data.
        :param data: The data to be used for predictions.
        """
        self.train_data = data.get_train_data()
        self.test_data = data.get_test_data()


    def predict(self):
        """
        Stub for prediction logic.
        """
        raise NotImplementedError("Prediction logic is not implemented yet.")

    def train(self, model: str, config: LogisticConfig | SVMConfig) -> str:
        """
        Train a model based on the given type and configuration.
        :param model: Model type ("logistic" or "svm").
        :param config: Configuration for the model.
        :return: Training result as a string.
        """
        if model == "logistic":
            return train_logistic_model(self.train_data, self.test_data, config)
        elif model == "svm":
            return train_svm_model(self.train_data, self.test_data, config)
        else:
            raise ValueError("Invalid model name")

def train_logistic_model(train: pd.DataFrame, test: pd.DataFrame, config: LogisticConfig) -> str:
    """
    Method to train a Logistic Regression model
    :param config: Model configuration
    Returns: str: ID model
    """
    X_train, y_train = train["vector"], train["cluster"]
    X_train = vector_transform(X_train)
    y_train = y_train.to_list()
    X_test, y_test = test["vector"], test["cluster"]
    X_test = vector_transform(X_test)
    y_test = y_test.to_list()
    model = OneVsRestClassifier(LogisticRegression(C=config.C, solver=config.solver))
    model.fit(X_train, y_train)
    model_id = str(uuid.uuid4())
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    # save_model(model, f"../../data/nlp/{model_id}.pickle")

    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    report = classification_report(y_test, y_pred)
    statistics = {
        "roc_auc": roc_auc,
        "classification_report": report,
    }
    return model_id


def train_svm_model(train: pd.DataFrame, test: pd.DataFrame, config: LogisticConfig) -> str:
    """ "
    Method to train an SVM model
    :param config: Model configuration
    Returns: str: ID model
    """
    X_train, y_train = train["vector"], train["cluster"]
    X_train = vector_transform(X_train)
    X_test, y_test = test["vector"], test["cluster"]
    X_test = vector_transform(X_test)
    model = SVC(C=config.C, kernel=config.kernel, class_weight=config.class_weight)
    model.fit(X_train, y_train)
    model_id = str(uuid.uuid4())
    model.fit(X_train, y_train)
    model_id = str(uuid.uuid4())
    y_pred = model.predict(X_test)
    # save_model(model, f"../../data/nlp/{model_id}.pickle")

    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    statistics = {
        "roc_auc": roc_auc,
        "classification_report": classification_report(y_test, y_pred),
        "fpr": fpr,
        "tpr": tpr,
    }
    return model_id
