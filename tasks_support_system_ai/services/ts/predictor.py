from datetime import datetime, timedelta
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

    def train_model(  # noqa: PLR0913
        self,
        queue_id: int,
        model_type: ModelType = "naive",
        forecast_horizon: int = 30,
        train_start_date: datetime | None = None,
        train_end_date: datetime | None = None,
        forecast_start_date: datetime | None = None,
    ) -> tuple[object, ModelMetrics]:
        """
        Обучение модели прогнозирования временного ряда с явным указанием периодов
        """
        try:
            # Получаем данные для обучения
            train_start_str = train_start_date.strftime("%Y-%m-%d") if train_start_date else None
            train_end_str = train_end_date.strftime("%Y-%m-%d") if train_end_date else None

            # Получаем данные для обучения
            train_ts = self.get_historical_data(queue_id, train_start_str, train_end_str)

            # Получаем данные для валидации
            if forecast_start_date:
                forecast_end_date = forecast_start_date + timedelta(days=forecast_horizon)
                forecast_start_str = forecast_start_date.strftime("%Y-%m-%d")
                forecast_end_str = forecast_end_date.strftime("%Y-%m-%d")

                # Получаем данные для тестирования/валидации
                test_ts = self.get_historical_data(queue_id, forecast_start_str, forecast_end_str)
            else:
                test_ts = None

            # Выбираем и обучаем модель
            model = self._create_model(model_type, output_chunk_length=forecast_horizon)

            # Обучение модели
            model.fit(train_ts)

            # Расчет метрик, если есть тестовые данные
            if test_ts and len(test_ts) > 0:
                pred_ts = model.predict(len(test_ts))

                # Расчет метрик
                rmse_val = rmse(test_ts, pred_ts)
                mae_val = mae(test_ts, pred_ts)
                mape_val = mape(test_ts, pred_ts)

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

            # Сохраняем модель с ключом, который включает информацию о периодах
            key = (queue_id, model_type)
            if forecast_start_date:
                key = (queue_id, model_type, forecast_start_date.strftime("%Y-%m-%d"))

            self.trained_models[key] = model

            return model, metrics

        except Exception as e:
            logger.error(f"Error training model for queue {queue_id}: {str(e)}")
            logger.exception("Error")
            raise

    def predict_ts(
        self,
        queue_id: int,
        forecast_horizon: int,
        model_type: ModelType = "naive",
        forecast_start_date=None,
    ) -> TimeSeries:
        """Выполнение прогнозирования временного ряда с заданной даты"""
        try:
            model_key = (queue_id, model_type)
            if forecast_start_date:
                model_key = (queue_id, model_type, forecast_start_date.strftime("%Y-%m-%d"))

            if model_key not in self.trained_models:
                logger.info(
                    f"Model for queue {queue_id} not found. Training new {model_type} model..."
                )
                self.train_model(
                    queue_id,
                    model_type,
                    forecast_horizon=forecast_horizon,
                    forecast_start_date=forecast_start_date,
                )

            model = self.trained_models[model_key]

            forecast_ts = model.predict(forecast_horizon)
            # для катбуста плывут индексы из-за своей логики
            if model_type == "catboost" and forecast_start_date:
                import pandas as pd
                from darts import TimeSeries

                forecast_values = forecast_ts.values().flatten()

                correct_index = pd.date_range(
                    start=forecast_start_date, periods=forecast_horizon, freq="D"
                )

                df = pd.DataFrame({"new_tickets": forecast_values}, index=correct_index)
                forecast_ts = TimeSeries.from_dataframe(df)
            logger.info(f"Corrected CatBoost forecast to start from {forecast_start_date}")

            return forecast_ts

        except Exception as e:
            logger.error(f"Error in predict_ts: {str(e)}")
            logger.exception("Exception details")
            raise

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
