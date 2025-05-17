import uuid

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# from transformers import (
#     AutoTokenizer,
# )
from xgboost import XGBClassifier

# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
from tasks_support_system_ai.api.models.nlp import (
    BertConfig,
    CatBoostConfig,
    LightGBMConfig,
    LogisticConfig,
    SVMConfig,
    XGBoostConfig,
)

# from torch.utils.data import DataLoader
# from torch import nn
from tasks_support_system_ai.core.logger import backend_logger as logger  # noqa: F401
from tasks_support_system_ai.data.nlp.reader import NLPTicketsData
from tasks_support_system_ai.services.nlp.model_service import ModelService
from tasks_support_system_ai.services.nlp.preprocessor import TextPreprocessor
from tasks_support_system_ai.utils.nlp import vector_transform

model_service = ModelService()
text_preprocessor = TextPreprocessor()


class NLPPredictor:
    def __init__(self, data: NLPTicketsData):
        """
        Initialize NLPPredictor with data.
        :param data: The data to be used for predictions.
        """
        self.train_data = data.get_train_data()
        self.test_data = data.get_test_data()
        self.w2v_model = data.get_word2vec_model()

    def predict(self, model_id: str, text: str) -> int:
        """
        Stub for prediction logic.
        """
        if not model_service._model_exists(model_id):
            logger.error(f"Model '{model_id}' does not exist.")
            raise ValueError(f"Model {model_id} not found")

        model = model_service.load_model(model_id)

        tokenized_text = text_preprocessor.preprocess_text(text)
        vector = [get_mean_vector(tokenized_text, self.w2v_model)]
        prediction = model.predict(vector).ravel()
        return prediction.tolist()

    def predict_for_file(self, model_id: str, df: pd.DataFrame) -> pd.DataFrame:
        if not model_service._model_exists(model_id):
            logger.error(f"Model '{model_id}' does not exist.")
            raise ValueError(f"Model {model_id} not found")

        model = model_service.load_model(model_id)

        X = transform_data(df, self.w2v_model)
        prediction = model.predict(X)
        df.drop(columns=["tokenized_text"], inplace=True)
        df["prediction"] = prediction

        return df

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
        elif model == "catboost":
            return train_catboost_model(self.train_data, self.test_data, config)
        elif model == "xgboost":
            return train_xgboost_model(self.train_data, self.test_data, config)
        elif model == "lightgbm":
            return train_lightgbm_model(self.train_data, self.test_data, config)
        elif model == "bert":
            return train_bert_model(self.train_data, self.test_data, config)
        else:
            raise ValueError("Invalid model name")

    def get_classification_report(self, id: str) -> dict:
        return model_service.get_statistics(id)

    def remove_model(self, model_id: str) -> None:
        model_service.remove_model(model_id)

    def get_models(self) -> list:
        return model_service.list_models()


def train_logistic_model(train: pd.DataFrame, test: pd.DataFrame, config: SVMConfig) -> str:
    """
    Method to train a Logistic Regression model
    :param config: Model configuration
    Returns: str: ID model
    """
    X_train, y_train = train["vector"], train["cluster"]
    X_train = vector_transform(X_train)

    X_test, y_test = test["vector"], test["cluster"]
    X_test = vector_transform(X_test)

    model = OneVsRestClassifier(LogisticRegression(C=config.C, solver=config.solver))
    model.fit(X_train, y_train)

    model_id = str(uuid.uuid4())

    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report["roc_auc"] = roc_auc

    model_service.save(model, model_id)
    model_service.save_stats(model_id, report)

    return model_id


def train_svm_model(train: pd.DataFrame, test: pd.DataFrame, config: SVMConfig) -> str:
    """ "
    Method to train an SVM model
    :param config: Model configuration
    Returns: str: ID model
    """
    X_train, y_train = train["vector"], train["cluster"]
    X_train = vector_transform(X_train)

    X_test, y_test = test["vector"], test["cluster"]
    X_test = vector_transform(X_test)

    model = SVC(
        C=config.C,
        kernel=config.kernel,
        class_weight=config.class_weight,
        decision_function_shape="ovr",
        probability=True,
    )
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


def train_catboost_model(train: pd.DataFrame, test: pd.DataFrame, config: CatBoostConfig) -> str:
    """
    Method to train a CatBoost classification model
    :param config: Model configuration
    Returns: str: ID model
    """
    X_train, y_train = train["vector"], train["cluster"]
    X_train = vector_transform(X_train)

    X_test, y_test = test["vector"], test["cluster"]
    X_test = vector_transform(X_test)

    model = CatBoostClassifier(
        iterations=config.iterations,
        depth=config.depth,
        learning_rate=config.learning_rate,
        l2_leaf_reg=config.l2_leaf_reg,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        verbose=0,
    )
    model.fit(X_train, y_train)
    model_id = str(uuid.uuid4())

    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_pred = y_pred + 1

    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report["roc_auc"] = roc_auc

    model_service.save(model, model_id)
    model_service.save_stats(model_id, report)

    return model_id


def train_xgboost_model(train: pd.DataFrame, test: pd.DataFrame, config: XGBoostConfig) -> str:
    """
    Method to train a XGBoost classification model
    :param config: Model configuration
    Returns: str: ID model
    """
    X_train, y_train = train["vector"], train["cluster"]
    X_train = vector_transform(X_train)

    X_test, y_test = test["vector"], test["cluster"]
    X_test = vector_transform(X_test)

    model = XGBClassifier(
        num_class=config.num_class,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        eval_metric="mlogloss",
        objective="multi:softmax",
    )
    y_train = y_train - 1
    y_test = y_test - 1
    model.fit(X_train, y_train)
    model_id = str(uuid.uuid4())

    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report["roc_auc"] = roc_auc

    model_service.save(model, model_id)
    model_service.save_stats(model_id, report)

    return model_id


def train_lightgbm_model(train: pd.DataFrame, test: pd.DataFrame, config: LightGBMConfig) -> str:
    """
    Method to train LightGBM classification model
    :param config: Model configuration
    Returns: str: ID model
    """
    X_train, y_train = train["vector"], train["cluster"]
    X_train = vector_transform(X_train)

    X_test, y_test = test["vector"], test["cluster"]
    X_test = vector_transform(X_test)

    model = LGBMClassifier(
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        num_classes=10,
        metric="multi_logloss",
        objective="multiclass",
        verbose=-1,
    )
    y_train = y_train - 1
    y_test = y_test - 1
    model.fit(X_train, y_train)
    model_id = str(uuid.uuid4())

    y_pred_proba = model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report["roc_auc"] = roc_auc

    model_service.save(model, model_id)
    model_service.save_stats(model_id, report)

    return model_id


def transform_data(df, w2v_model):
    for index, text in enumerate(df["ticket"]):
        row = text_preprocessor.preprocess_text(text)
        df.at[index, "tokenized_text"] = " ".join(row)
    X = np.array([get_mean_vector(text, w2v_model) for text in df["tokenized_text"]])
    return X


def get_mean_vector(text, w2v_model):
    words = [w for w in text if w in w2v_model.wv]
    vector = np.mean(w2v_model.wv[words], axis=0) if words else np.zeros(w2v_model.vector_size)
    return vector


def train_bert_model(train: pd.DataFrame, test: pd.DataFrame, config: BertConfig) -> str:
    """
    Method to train BERT classification model
    :param config: Model configuration
    :return: str: ID of the model
    """
    pass
    # MODEL_NAME = "sberbank-ai/ruBert-base"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # def preprocess_function(examples):
    #     return tokenizer(
    #         examples["text"],
    #         padding="max_length",
    #         truncation=True,
    #         max_length=128,
    #     )
    #
    # model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=10)
    #
    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     evaluation_strategy="epoch",
    #     save_strategy="no",
    #     learning_rate=config.learning_rate,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     num_train_epochs=config.num_epochs,
    #     weight_decay=config.weight_decay,
    #     logging_steps=10,
    #     logging_dir="./logs",
    # )
    #
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=test_dataset,
    #     tokenizer=tokenizer,
    #     data_collator=DataCollatorWithPadding(tokenizer),
    # )
    #
    # trainer.train()
    #
    # predictions = trainer.predict(test_dataset)
    # y_pred_logits = predictions.predictions
    # y_pred = np.argmax(y_pred_logits, axis=1)
    # y_true = predictions.label_ids
    #
    # roc_auc = roc_auc_score(y_true, y_pred_logits, multi_class="ovr")
    # report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # report["roc_auc"] = roc_auc
    #
    # model_id = str(uuid.uuid4())
    # model_path = f"./saved_models/{model_id}"
    # model.save_pretrained(model_path)
    #
    # model_service.save(model, model_id)
    # model_service.save_stats(model_id, report)
    #
    # return model_id
