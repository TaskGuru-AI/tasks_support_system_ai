import uuid
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score, roc_auc_score, roc_curve)

from typing import Union
import pandas as pd

from tasks_support_system_ai.api.models.nlp import LogisticConfig, SVMConfig
from tasks_support_system_ai.data.reader import DataFrames, read_data
from tasks_support_system_ai.utils.utils import data_checker, get_correct_data_path
from tasks_support_system_ai.services.nlp.predictor import add_model_statistics

if data_checker.check_data_availability(
        [
            get_correct_data_path("data/nlp_tickets.csv")
        ]
):
    df_train = read_data(DataFrames.NLP_TICKETS_TRAIN)
    df_test = read_data(DataFrames.NLP_TICKETS_TEST)
else:
    df = pd.DataFrame()


def train_model(model: str, config: Union[LogisticConfig, SVMConfig]) -> str:
    if model == 'logistic':
        return train_logistic_model(config)
    elif model == 'svm':
        return train_svm_model(config)
    else:
        raise ValueError('Invalid model name')


def train_logistic_model(config: LogisticConfig) -> str:
    """
     Method to train a Logistic Regression model
     Args:
         config: Конфигурация модели
     Returns:
        str: ID модели
    """
    X_train, y_train = df_train['vector'], df_train['cluster']
    X_test, y_test = df_test['vector'], df_test['cluster']
    model = LogisticRegression(C=config.C, solver=config.solver)
    model.fit(X_train, y_train)

    model_id = str(uuid.uuid4())
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    statistics = {
        'roc_auc': roc_auc,
        'classification_report': classification_report(y_test, y_pred),
        'fpr': fpr,
        'tpr': tpr
    }
    add_model_statistics(model_id, statistics)
    return model_id


def train_svm_model(config: SVMConfig):
    """"
     Method to train an SVM model
     Args:
         config: Конфигурация модели
     Returns:
        str: ID модели
    """
    X_train, y_train = df_train['vector'], df_train['cluster']
    X_test, y_test = df_test['vector'], df_test['cluster']
    model = SVC(C=config.C, kernel=config.kernel, class_weight=config.class_weight)
    model.fit(X_train, y_train)
    model_id = str(uuid.uuid4())
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    statistics = {
        'roc_auc': roc_auc,
        'classification_report': classification_report(y_test, y_pred),
        'fpr': fpr,
        'tpr': tpr
    }
    add_model_statistics(model_id, statistics)
    return model_id
