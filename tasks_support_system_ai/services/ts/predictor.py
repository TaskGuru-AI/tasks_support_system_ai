from darts import TimeSeries
from darts.models import (
    LinearRegressionModel,
)

from tasks_support_system_ai.data.reader import DataConversion, TSDataIntersection


class TSPredictor:
    def __init__(self, data_service: TSDataIntersection):
        self.data_service = data_service

    def predict_ts(self, queue_id: int, days_ahead: int) -> TimeSeries:
        data = self.data_service.get_tickets_load_filter(queue_id=queue_id)
        ts = DataConversion.pandas_to_darts(data)

        model = LinearRegressionModel(lags=10)
        model.fit(ts)

        forecast = model.predict(days_ahead)

        return forecast


class ModelService:
    """Logic how to train and store models."""

    ...
