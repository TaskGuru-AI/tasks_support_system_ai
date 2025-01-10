import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from darts import TimeSeries
from darts.models import ExponentialSmoothing, LinearRegressionModel

from tasks_support_system_ai.api.models.ts import ModelConfig, ModelInfo
from tasks_support_system_ai.core.config import settings
from tasks_support_system_ai.core.logger import backend_logger as logger
from tasks_support_system_ai.data.reader import DataConversion, TSDataIntersection


class TSPredictor:
    def __init__(self, data_service: TSDataIntersection):
        self.data_service = data_service

    def predict_ts(self, queue_id: int, forecast_horizon: int) -> TimeSeries:
        data = self.data_service.get_tickets_load_filter(queue_id=queue_id)
        data = data.set_index("date")
        ts = DataConversion.pandas_to_darts(data)

        model = LinearRegressionModel(lags=10)
        model.fit(ts)

        forecast = model.predict(forecast_horizon)

        return forecast


def validate_training_data(ts) -> tuple[bool, str | None]:
    """Валидация данных для обучения"""
    try:
        if not isinstance(ts, TimeSeries):
            raise ValueError("ts should be darts.TimeSeries format")
        return True, None
    except Exception as e:
        return False, f"Invalid data format: {str(e)}"


def train_model_process(
    X: list[list[float]], y: list[float], model_config: ModelConfig, models_dir: str
) -> None:
    """Функция для обучения в отдельном процессе"""
    try:
        is_valid, error_message = validate_training_data(X, y)
        if not is_valid:
            raise ValueError(error_message)

        model_id = model_config["id"]
        model_type = model_config["ml_model_type"]
        hyperparameters = model_config.get("hyperparameters", {})

        MODEL_TYPES = {"linear": LinearRegressionModel, "logistic": ExponentialSmoothing}

        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}")

        X_array = np.array(X)
        y_array = np.array(y)

        logger.info(f"Training model {model_id} with {len(X_array)} samples")
        logger.info(f"X shape: {X_array.shape}, y shape: {y_array.shape}")

        model = MODEL_TYPES[model_type](**hyperparameters)
        model.fit(X_array, y_array)

        model_path = Path(models_dir) / f"{model_id}.joblib"
        joblib.dump(model, model_path)

        logger.info(f"Successfully trained and saved model {model_id}")

    except ValueError as e:
        logger.error(f"Validation error in training process: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training process: {str(e)}")
        raise RuntimeError(f"Training failed: {str(e)}")


class ModelService:
    """Logic how to train and store models."""

    MODEL_TYPES = {"linear": LinearRegressionModel, "smoothing": ExponentialSmoothing}

    def __init__(self):
        self.models_dir = settings.models_path
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models = {}
        self.current_loaded_model: str | None = None
        self.max_loaded_models = settings.max_loaded_models
        self._executor = ProcessPoolExecutor(max_workers=3)
        self.max_processes = 3
        self.active_processes = 0
        self._process_lock = asyncio.Lock()

    def get_model_id(self, model_class, date_start, date_end):
        # TODO: new hyperparameters create new model
        return model_class.__name__ + str(date_start) + str(date_end) + "_1"

    async def fit_async(self, ts: TimeSeries, model_config: ModelConfig) -> str:
        """Асинхронная функция для обучения. Создает новый процесс, если возможно.

        Мы создаем новый процесс, а потом опрашиваем, закончил ли он работу, с небольшим интервалом.
        """
        try:
            is_valid, error_message = validate_training_data(ts)
            if not is_valid:
                raise ValueError(error_message)

            model_class = ModelConfig.ml_model_type
            date_start, date_end = str(ts.time_index[0].date()), str(ts.time_index[-1].date())
            model_id = self.get_model_id(model_class, date_start, date_end)

            if self._model_exists(model_id):
                raise ValueError(f"Model with id {model_id} already exists")

            async with (
                self._process_lock
            ):  # нужна блокировка, чтобы избежать гонок по обновлению счетчика
                if self.active_processes >= self.max_processes:
                    raise RuntimeError(
                        f"No available processes. Maximum ({self.max_processes})"
                        " processes are already running"
                    )
                self.active_processes += 1

            try:
                process = multiprocessing.Process(
                    target=train_model_process,
                    args=(ts, model_config, str(self.models_dir)),
                )
                process.start()

                while process.is_alive():  # noqa: ASYNC110
                    await asyncio.sleep(0.1)

                if process.exitcode != 0:
                    raise RuntimeError("Training process failed")

                process.join()
                return f"Model '{model_id}' trained and saved"

            finally:
                async with self._process_lock:
                    self.active_processes -= 1

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except RuntimeError as e:
            logger.error(f"Runtime error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")

    def load(self, model_id: str) -> str:
        if not self._model_exists(model_id):
            raise KeyError(f"Model {model_id} not found")

        if len(self.loaded_models) >= self.max_loaded_models and model_id not in self.loaded_models:
            raise RuntimeError(f"Cannot load more than {self.max_loaded_models} models")

        # перезаписать загруженную модель
        if self.current_loaded_model and self.current_loaded_model != model_id:
            self.unload()

        if model_id not in self.loaded_models:
            model_path = self.models_dir / f"{model_id}.joblib"
            self.loaded_models[model_id] = joblib.load(model_path)
            self.current_loaded_model = model_id

        return f"Model '{model_id}' loaded"

    def unload(self) -> str:
        if self.current_loaded_model:
            model_id = self.current_loaded_model
            del self.loaded_models[model_id]
            self.current_loaded_model = None
            return f"Model '{model_id}' unloaded"
        return "No model currently loaded"

    def get_status(self) -> dict:
        return {
            "status": "Model Status Ready",  # что бы это не обозначало
            "loaded_model": self.current_loaded_model,
            "available_cores": multiprocessing.cpu_count(),
        }

    def predict(self, X: list[list[float]], model_id: str) -> list[float]:
        if not self._model_exists(model_id):
            raise ValueError(f"Model {model_id} not found")

        if model_id not in self.loaded_models:
            self.load(model_id)

        model = self.loaded_models[model_id]
        predictions: np.ndarray = model.predict(np.array(X))
        return predictions.tolist()

    def list_models(self):
        models = []
        for model_path in self.models_dir.glob("*.joblib"):
            model = joblib.load(model_path)
            # коряво, лучше бы просто подгружать метаданные
            model_info = ModelInfo(
                id=model_path.stem,
                type="linear" if isinstance(model, LinearRegressionModel) else "logistic",
            )
            models.append(model_info)
        return models

    def remove(self, model_id: str) -> str:
        if not self._model_exists(model_id):
            raise KeyError(f"Model {model_id} not found")

        if self.current_loaded_model == model_id:
            self.unload()

        model_path = self.models_dir / f"{model_id}.joblib"
        model_path.unlink()
        return f"Model '{model_id}' removed"

    def remove_all(self) -> str:
        self.unload()
        for model_path in self.models_dir.glob("*.joblib"):
            model_path.unlink()
        return "All models removed"

    def _model_exists(self, model_id: str) -> bool:
        model_path = self.models_dir / f"{model_id}.joblib"
        return model_path.exists()

    def remove_model(self, model_id: str):
        if not self._model_exists(model_id):
            raise KeyError(f"Model {model_id} not found")

        if model_id == self.current_loaded_model:
            self.unload()

    def __del__(self):
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
