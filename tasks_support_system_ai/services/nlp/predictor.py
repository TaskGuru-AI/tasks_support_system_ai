import logging
import uuid

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
from tasks_support_system_ai.api.models.nlp import LogisticConfig, SVMConfig
from tasks_support_system_ai.data.nlp.reader import NLPTicketsData
from tasks_support_system_ai.services.nlp.model_service import ModelService
from tasks_support_system_ai.utils.nlp import vector_transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_service = ModelService()


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

    def get_classification_report(self, id: str) -> dict:
        return model_service.get_statistics(id)

    def remove_model(self, model_id: str) -> None:
        model_service.remove_model(model_id)



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

    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    report = classification_report(y_test, y_pred, output_dict=True)
    report["roc_auc"] = roc_auc

    model_service.save(model, model_id)
    model_service.save_stats(model_id, report)

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

    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    report = classification_report(y_test, y_pred, output_dict=True)
    report["roc_auc"] = roc_auc

    model_service.save(model, model_id)
    model_service.save_stats(model_id, report)

    return model_id
