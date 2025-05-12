import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline


def create_time_features(df, date_col='date'):
    """
    Создает временные признаки: день недели, месяц, год.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df['dayofweek'] = df[date_col].dt.dayofweek  # День недели (0-6, понедельник-воскресенье)
    df['month'] = df[date_col].dt.month          # Месяц (1-12)
    df['year'] = df[date_col].dt.year            # Год
    return df

def create_X_y (global_df_top_level, features, target):
    X = global_df_top_level[features]
    y = global_df_top_level[target]
    groups = global_df_top_level['queue_id']
    return X, y, groups

def create_lag_features(df, lags, queue_id_col='queue_id', target_col='new_tickets'):
    """
    Создает лаговые признаки для временного ряда в DataFrame.
    """
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby(queue_id_col)[target_col].shift(lag)
    return df


def create_anomaly_feature(
        df, 
        queue_id_col='queue_id',
        target_col='new_tickets', 
        contamination=0.05, 
        random_state=42
        ):
    """
    Создает бинарный признак 'is_anomaly' на основе детекции аномалий с помощью Isolation Forest.
    Детекция аномалий выполняется отдельно для каждого временного ряда (очереди).
    """

    df['is_anomaly'] = 0  # Инициализируем колонку 'is_anomaly' нулями

    for queue_id in df[queue_id_col].unique():
        df_queue = df[df[queue_id_col] == queue_id].copy() 

        
        model_if = IsolationForest(contamination=contamination, random_state=random_state)
        model_if.fit(df_queue[[target_col]]) 

       
        anomaly_labels = model_if.predict(df_queue[[target_col]])


        is_anomaly_flag = (anomaly_labels == -1).astype(int) 

  
        df.loc[df[queue_id_col] == queue_id, 'is_anomaly'] = is_anomaly_flag 


    return df

def create_rolling_lag_features(
        df, 
        lags, 
        windows, 
        queue_id_col='queue_id', 
        target_col='new_tickets',
        ):
    """
    Создает признаки скользящих статистик для лаговых признаков временного ряда.
    """
    statistics=['mean', 'median', 'std']
    for lag in lags:
        for window in windows:
            for stat in statistics:
                col_name = f'{target_col}_lag_{lag}_rolling_{stat}_{window}'
                if stat == 'mean':
                    df[col_name] = df.groupby(queue_id_col)[f'{target_col}_lag_{lag}'].rolling(
                        window=window, 
                        min_periods=1).mean().reset_index(level=0, drop=True)
                elif stat == 'median':
                    df[col_name] = df.groupby(queue_id_col)[f'{target_col}_lag_{lag}'].rolling(
                        window=window, 
                        min_periods=1).median().reset_index(level=0, drop=True)
                elif stat == 'std':
                    df[col_name] = df.groupby(queue_id_col)[f'{target_col}_lag_{lag}'].rolling(
                        window=window, 
                        min_periods=1).std().reset_index(level=0, drop=True)
                else:
                    raise ValueError(
                        f"Statistic '{stat}' is not supported. Choose from 'mean', 'median', 'std'."
                        )
    return df


class GlobalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Выполняет все этапы предобработки данных, включая создание временных признаков, 
    лаговых признаков,
    детектирование аномалий и создание признаков скользящих статистик.
    """

    def __init__(self, lags, lags_for_rolling, windows_for_rolling):
        self.lags = lags
        self.lags_for_rolling = lags_for_rolling
        self.windows_for_rolling = windows_for_rolling
        self.date_col = 'date'
        self.queue_id_col = 'queue_id'
        self.target_col = 'new_tickets'
        self.contamination = 0.05
        self.random_state = 42
        self.statistics = ['mean', 'median', 'std']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()

        # 1. Создаем временные признаки
        X_ = create_time_features(X_, date_col=self.date_col)

        # 2. Создаем лаговые признаки
        X_ = create_lag_features(
            X_, 
            lags=self.lags, 
            queue_id_col=self.queue_id_col, 
            target_col=self.target_col
            )

        X_ = X_.dropna()

        # 3. Создаем признак аномалии (требует обучения Isolation Forest для каждой очереди)
        X_ = create_anomaly_feature(
            X_, 
            queue_id_col=self.queue_id_col, 
            date_col=self.date_col, 
            target_col=self.target_col,
            contamination=self.contamination, 
            random_state=self.random_state
            )

        # 4. Создаем признаки скользящих статистик
        X_ = create_rolling_lag_features(
            X_, 
            lags=self.lags_for_rolling, 
            windows=self.windows_for_rolling,
            queue_id_col=self.queue_id_col, 
            target_col=self.target_col
            )
        
        X_ = X_.dropna()

        return X_

def create_preprocessing_pipeline(lags_to_create,
                                  lags_for_rolling,
                                  windows_for_rolling,
                                  ):
    """
    Создает и возвращает экземпляр sklearn Pipeline с GlobalPreprocessor.
    """
    global_preprocessor = GlobalPreprocessor(lags=lags_to_create, lags_for_rolling=lags_for_rolling,
                                            windows_for_rolling=windows_for_rolling)

    pipeline = Pipeline(steps=[
        ('feature_preprocessing', global_preprocessor),
    ])

    return pipeline

def create_transformed_df(
        df,
        lags_to_create,
        lags_for_rolling,
        windows_for_rolling,
    ):

    """
    Создает и возвращает df после feature engineering
    """
    preprocess_pipeline = create_preprocessing_pipeline(
        lags_to_create,
        lags_for_rolling,
        windows_for_rolling,
    )

    transformed_df = preprocess_pipeline.fit_transform(df)

    return transformed_df

