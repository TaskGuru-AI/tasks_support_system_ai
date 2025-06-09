import uuid

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from xgboost import XGBClassifier

from tasks_support_system_ai.api.models.nlp import (
    BertConfig,
    CatBoostConfig,
    LightGBMConfig,
    LogisticConfig,
    SVMConfig,
    XGBoostConfig,
)
from tasks_support_system_ai.core.logger import backend_logger as logger  # noqa: F401
from tasks_support_system_ai.data.nlp.reader import NLPTicketsData
from tasks_support_system_ai.services.nlp.model_service import ModelService
from tasks_support_system_ai.services.nlp.preprocessor import TextPreprocessor
from tasks_support_system_ai.utils.nlp import tokenize_texts, vector_transform

model_service = ModelService()
text_preprocessor = TextPreprocessor()


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class NLPPredictor:
    def __init__(self, data: NLPTicketsData):
        """
        Initialize NLPPredictor with data.
        :param data: The data to be used for predictions.
        """
        self.train_data = data.get_train_data()
        self.test_data = data.get_test_data()
        self.bert_data = data.get_bert_data()
        self.w2v_model = data.get_word2vec_model()

    def predict(self, model_id: str, text: str) -> int:
        """
        Predict the label for a given text.
        :param model_id: The ID of the model to use for prediction.
        :param text: The text to be predicted.
        :return: The predicted label.
        """
        if not model_service._model_exists(model_id):
            logger.error(f"Model '{model_id}' does not exist.")
            raise ValueError(f"Model {model_id} not found")

        model = model_service.load_model(model_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(model, tuple):
            model, tokenizer = model
            model = model.to(device)
            encoded = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(device)
            with torch.no_grad():
                logits = model(**encoded).logits
                prediction = int(torch.argmax(logits, dim=1).item())
                return [prediction]
        else:
            tokenized_text = text_preprocessor.preprocess_text(text)
            vector = [get_mean_vector(tokenized_text, self.w2v_model)]
            prediction = model.predict(vector).ravel()
            return prediction.tolist()

    def predict_for_file(self, model_id: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the label for a given text.
        :param model_id: The ID of the model to use for prediction.
        :param text: The text to be predicted.
        :return: The predicted label.
        """
        if not model_service._model_exists(model_id):
            logger.error(f"Model '{model_id}' does not exist.")
            raise ValueError(f"Model {model_id} not found")

        model = model_service.load_model(model_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(model, tuple):
            model, tokenizer = model
            model = model.to(device)
            model.eval()

            texts = df["text"].tolist()
            batch_size = 32
            all_preds = []

            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    encoded = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128,
                    ).to(device)
                    logits = model(**encoded).logits
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().tolist())

            df["prediction"] = all_preds

        else:
            # Классическая модель — надо трансформировать данные
            X = transform_data(df, self.w2v_model)
            prediction = model.predict(X)
            df["prediction"] = prediction

        if "tokenized_text" in df.columns:
            df.drop(columns=["tokenized_text"], inplace=True)

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
            return train_bert_model(self.bert_data, config)
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


def train_bert_model(data: pd.DataFrame, config: BertConfig) -> str:
    """
    Method to train BERT classification model
    :param config: Model configuration
    :return: str: ID of the model
    """
    MODEL_NAME = "sberbank-ai/ruBert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=10)

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"].values,
        data["label"].values,
        test_size=0.25,
        stratify=data["label"].values,
        random_state=42,
    )

    train_encodings = tokenize_texts(X_train.tolist(), tokenizer)
    test_encodings = tokenize_texts(X_test.tolist(), tokenizer)

    train_dataset = TextDataset(train_encodings, y_train)
    test_dataset = TextDataset(test_encodings, y_test)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = model.to(device)

    model.train()
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}")
        total_loss = 0
        for batch in train_loader:
            out_batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**out_batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    print("Training completed")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            out_batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**out_batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    print("Inference completed")
    all_probs = torch.nn.functional.softmax(torch.tensor(all_logits), dim=1).numpy()

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    report["roc_auc"] = roc_auc_score(all_labels, all_probs, multi_class="ovr")

    model_id = str(uuid.uuid4())

    model_service.save_pretrained(model, model_id)
    model_service.save_tokenizer(tokenizer, model_id)
    model_service.save_stats(model_id, report)

    return model_id
