from typing import List
import pandas as pd
from tensorflow import keras
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score

from rnn_baseline.data_generators import batches_generator


def train_epoch(model: keras.Model, dataset_train: List[str], batch_size: int = 64, shuffle: bool = True, 
                cur_epoch: int = 0, steps_per_epoch: int = 5000, callbacks: List[keras.callbacks.Callback] = None):
    """
    Делает одну эпоху обучения модели, логируя промежуточные значения функции потерь.

    Параметры:
    -----------
    model: tensorflow.keras.Model
        Обучаемая модель.
    dataset_train: List[str]
        Список путей до файлов с предобработанными последовательностями.
    batch_size: int, default=64
        Размер батча.
    shuffle: bool, default=False
        Перемешивать ли данные перед подачей в модель.
    cur_epoch: int, deafult=0
        Номер текущей эпохи.
    steps_per_epoch: int, default=500
        Число шагов, которое необходимо сделать, чтобы завершить эпоху.
    callbacks: List[tensorflow.keras.callbacks.Callback], default=None
        Список коллбэков, применяемых при обучении модели

    Возвращаемое значение:
    ----------------------
    None
    """
    train_generator = batches_generator(dataset_train, batch_size=batch_size, shuffle=shuffle,
                                        output_format="tf", is_train=True)
    model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=cur_epoch+1,
              initial_epoch=cur_epoch, callbacks=callbacks)


def eval_model(model: keras.Model, dataset_val: List[str], batch_size: int = 32) -> float:
    """
    Скорит выборку моделью и вычисляет метрику ROC AUC.

    Параметры:
    -----------
    model: tensorflow.keras.Model
        Модель, которой необходимо проскорить выборку.
    dataset_val: List[str]
        Список путей до файлов с предобработанными последовательностями.
    batch_size: int, default=32
        Размер батча.

    Возвращаемое значение:
    ----------------------
    auc: float
    """
    val_generator = batches_generator(dataset_val, batch_size=batch_size, shuffle=False,
                                      output_format="tf", is_train=True)
    preds = model.predict(val_generator).flatten()
    val_generator = batches_generator(dataset_val, batch_size=batch_size, shuffle=False,
                                      output_format="tf", is_train=True)
    targets = []
    for _, y in val_generator:
        targets.extend(y)

    return roc_auc_score(targets, preds)


def inference(model: keras.Model, dataset_test: List[str], batch_size: int = 32) -> pd.DataFrame:
    """
    Скорит выборку моделью.

    Параметры:
    -----------
    model: tensorflow.keras.Model
        Модель, которой необходимо проскорить выборку.
    dataset_test: List[str]
        Список путей до файлов с предобработанными последовательностями.
    batch_size: int, default=32
        Размер батча.

    Возвращаемое значение:
    ----------------------
    scores: pandas.DataFrame
        Датафрейм с двумя колонками: "id" - идентификатор заявки и "score" - скор модели.
    """
    test_generator = batches_generator(dataset_test, batch_size=batch_size, shuffle=False,
                                       is_train=False, output_format="tf")

    preds = model.predict(test_generator).flatten()
    
    ids = []
    test_generator = batches_generator(dataset_test, batch_size=batch_size, shuffle=False, 
                                       is_train=False, output_format="tf")
    for _, batch_ids in test_generator:
        ids.extend(batch_ids)
        
    return pd.DataFrame({
        "id": ids,
        "score": preds
    })
