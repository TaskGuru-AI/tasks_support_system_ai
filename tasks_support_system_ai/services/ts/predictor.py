from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from darts import TimeSeries
from darts.metrics import mae, mape, rmse
from darts.models import (
    CatBoostModel,
    ExponentialSmoothing,
    LinearRegressionModel,
    NaiveSeasonal,
    Prophet,
)

from tasks_support_system_ai.api.models.ts import ModelConfig
from tasks_support_system_ai.core.logger import backend_logger as logger
from tasks_support_system_ai.data.reader import DataConversion, TSDataIntersection

ModelType = Literal["naive", "es", "prophet", "catboost", "linear"]


class ModelMetrics:
    """Класс для хранения метрик производительности модели"""

    def __init__(self, rmse_val: float, mae_val: float, mape_val: float, model_type: ModelType):
        self.rmse = rmse_val
        self.mae = mae_val
        self.mape = mape_val
        self.model_type = model_type

    def to_dict(self):
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "model_type": self.model_type,
        }


class TSPredictor:
    """Класс для прогнозирования временных рядов"""

    def __init__(self, data_service: TSDataIntersection):
        self.data_service = data_service
        self.trained_models: dict[tuple[int, ModelType], object] = {}

    def create_time_features(self, df, date_col="date"):
        """Создает временные признаки: день недели, месяц, год и другие."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df["dayofweek"] = df[date_col].dt.dayofweek  # 0-6, понедельник-воскресенье
        df["month"] = df[date_col].dt.month  # 1-12
        df["year"] = df[date_col].dt.year  # Год
        df["day"] = df[date_col].dt.day  # День месяца
        df["quarter"] = df[date_col].dt.quarter  # Квартал
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        # Циклические признаки для месяца и дня недели
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

        return df

    def get_historical_data(self, queue_id: int, start_date=None, end_date=None) -> TimeSeries:
        """Получение исторических данных для обучения модели"""
        df_slice = self.data_service.get_df_slice(queue_id, start_date, end_date)
        return DataConversion.pandas_to_darts(df_slice.set_index("date"))

    def _create_model(self, model_type: ModelType, output_chunk_length: int | None = None):
        """Создание экземпляра модели по типу"""
        if model_type == "naive":
            return NaiveSeasonal(K=7)  # Недельная сезонность
        elif model_type == "es":
            return ExponentialSmoothing(seasonal_periods=7)
        elif model_type == "prophet":
            return Prophet(
                yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False
            )
        elif model_type == "catboost":
            return CatBoostModel(
                CatBoostRegressor(
                    iterations=100,
                    learning_rate=0.03,
                    depth=6,
                    l2_leaf_reg=3,
                    random_strength=0.1,
                    verbose=0,
                ),
                lags=[-1, -2, -3, -7, -14, -28],
                add_encoders={
                    "cyclic": {"future": ["dayofweek", "month"]},
                    "datetime_attribute": {"future": ["dayofweek", "month", "year"]},
                },
                output_chunk_length=output_chunk_length,
            )
        elif model_type == "linear":
            return LinearRegressionModel(lags=10, output_chunk_length=30)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train_model(
        self,
        queue_id: int,
        model_type: ModelType = "naive",
        train_test_split: float = 0.8,
        forecast_horizon: int = 30,
    ) -> tuple[object, ModelMetrics]:
        """
        Обучение модели прогнозирования временного ряда - синхронная версия
        """
        try:
            # Получаем данные
            ts = self.get_historical_data(queue_id)

            # Разделяем на обучающую и тестовую выборки
            train_len = int(len(ts) * train_test_split)
            train_ts = ts[:train_len]
            test_ts = ts[train_len:]

            forecast_horizon = min(len(test_ts), forecast_horizon)

            # Выбираем и обучаем модель
            model = self._create_model(model_type, output_chunk_length=forecast_horizon)

            # Обучение модели
            model.fit(train_ts)

            # Прогнозирование и оценка метрик
            if len(test_ts) > 0:
                pred_ts = model.predict(forecast_horizon)

                # Расчет метрик
                rmse_val = rmse(test_ts[:forecast_horizon], pred_ts)
                mae_val = mae(test_ts[:forecast_horizon], pred_ts)
                mape_val = mape(test_ts[:forecast_horizon], pred_ts)

                metrics = ModelMetrics(
                    rmse_val=float(rmse_val),
                    mae_val=float(mae_val),
                    mape_val=float(mape_val),
                    model_type=model_type,
                )
            else:
                # Если тестовых данных нет, возвращаем нулевые метрики
                metrics = ModelMetrics(
                    rmse_val=0.0, mae_val=0.0, mape_val=0.0, model_type=model_type
                )

            # Сохраняем модель
            self.trained_models[(queue_id, model_type)] = model

            return model, metrics

        except Exception as e:
            logger.error(f"Error training model for queue {queue_id}: {str(e)}")
            logger.exception("Error")
            raise

    def predict_ts(
        self, queue_id: int, forecast_horizon: int, model_type: ModelType = "naive"
    ) -> TimeSeries:
        """
        Выполнение прогнозирования временного ряда с заданной даты
        """
        # Проверяем, есть ли обученная модель
        model_key = (queue_id, model_type)
        if model_key not in self.trained_models:
            logger.info(f"Model for queue {queue_id} not found. Training new {model_type} model...")
            self.train_model(queue_id, model_type, forecast_horizon=forecast_horizon)

        # Получаем модель
        model = self.trained_models[model_key]

        # Получаем исторические данные
        ts = self.get_historical_data(queue_id)

        # Переобучаем модель на всех данных
        model.fit(ts)

        # Выполняем прогноз
        forecast_ts = model.predict(forecast_horizon)

        return forecast_ts

    def get_all_models_predictions(
        self, queue_id: int, forecast_horizon: int
    ) -> dict[ModelType, tuple[TimeSeries, ModelMetrics]]:
        """
        Получить прогнозы от всех доступных моделей и их метрики

        Args:
            queue_id: ID очереди
            forecast_horizon: Горизонт прогнозирования

        Returns:
            Словарь с прогнозами и метриками для каждого типа модели
        """
        results = {}

        for model_type in ["naive", "es", "prophet", "catboost", "linear"]:
            try:
                model, metrics = self.train_model(
                    queue_id, model_type, forecast_horizon=forecast_horizon
                )
                forecast_ts = model.predict(forecast_horizon)
                results[model_type] = (forecast_ts, metrics)
            except Exception as e:
                logger.error(f"Error getting prediction for {model_type} model: {str(e)}")
                # Пропускаем модель, если возникла ошибка

        return results

    def clear_models(self, queue_id: int | None = None):
        """
        Очистить сохраненные модели для освобождения памяти

        Args:
            queue_id: ID очереди (если None, то очищаются все модели)
        """
        if queue_id is None:
            # Очистить все модели
            self.trained_models.clear()
        else:
            # Очистить модели только для конкретной очереди
            keys_to_remove = [key for key in self.trained_models if key[0] == queue_id]
            for key in keys_to_remove:
                del self.trained_models[key]


def validate_training_data(ts: TimeSeries) -> tuple[bool, str | None]:
    """Валидация данных для обучения"""
    try:
        if not isinstance(ts, TimeSeries):
            raise ValueError("ts should be darts.TimeSeries format")
        if len(ts) < 10:  # noqa: PLR2004
            raise ValueError("TimeSeries too short (< 10 points)")
        return True, None
    except Exception as e:
        return False, f"Invalid data format: {str(e)}"


def train_model_process(ts: TimeSeries, model_config: ModelConfig, models_dir: str) -> None:
    """Функция для обучения в отдельном процессе"""
    try:
        is_valid, error_message = validate_training_data(ts)
        if not is_valid:
            raise ValueError(error_message)

        model_id = model_config.id
        model_type = model_config.ml_model_type
        hyperparameters = model_config.hyperparameters or {}

        MODEL_TYPES = {
            "linear": LinearRegressionModel,
            "logistic": ExponentialSmoothing,
            "naive": NaiveSeasonal,
            "prophet": Prophet,
            "catboost": CatBoostModel,
        }

        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.info(f"Training model {model_id} with {len(ts)} samples")

        model_class = MODEL_TYPES[model_type]
        model = model_class(**hyperparameters)
        model.fit(ts)

        model_path = Path(models_dir) / f"{model_id}.joblib"
        joblib.dump(model, model_path)

        logger.info(f"Successfully trained and saved model {model_id}")

    except ValueError as e:
        logger.error(f"Validation error in training process: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training process: {str(e)}")
        raise RuntimeError(f"Training failed: {str(e)}")
