import torch
import numpy as np


class EarlyStopping:
    """
    Останавливает обучение модели, если валидационная метрика не улучшается в течение заданного числа эпох.

    Параметры:
    ----------
    patience: int, default=7
        Допустимое число эпох без улучшения валидационной метрики.
        Валидационная метрика должна улучшаться как минимум каждые ``patience`` эпох, иначе обучение останавливается.
    mode: str, default="min"
        Режим работы. Допустимые значения: "min", "max" - минимизация или максимизация целевой метрики соответственно.
    verbose: bool, default=False
        Печатать ли сообщение при каждом улучшении валидационной метрики.
    delta: int, default=0
        Минимальное изменение контролируемой метрики, которое можно считать улучшением.
    save_path: str, default="checkpoint.hdf5"
        Путь до файла, в который необходимо сохранять лучшую модель.
    metric_name: str, default=None
        Имя метрики.
    save_format: str, default="torch"
        Формат модели. Допустимые значения: "torch", "tf" - для моделей на фреймворках pytorch и tensorflow.keras соответственно.
    """

    def __init__(self, patience=7, mode='min', verbose=False, delta=0, save_path='checkpoint.hdf5', metric_name=None, save_format='torch'):
        if mode not in ["min", "max"]:
            raise ValueError(f"Unrecognized mode: {mode}! Please choose one of the following modes: \"min\", \"max\"")

        if save_format not in ["torch", "tf"]:
            raise ValueError(f"Unrecognized format: {save_format}! Please choose one of the following formats: \"torch\", \"tf\"")

        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_prev_score = np.Inf if mode == "min" else -np.Inf
        self.delta = delta
        self.save_path = save_path
        self.metric_name = "metric" if not metric_name else metric_name
        self.save_format = save_format

    def __call__(self, metric_value, model):

        score = -metric_value if self.mode == "min" else metric_value

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric_value, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f"No imporvement in validation {self.metric_name}. Current: {score:.6f}. Current best: {self.best_score:.6f}")
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric_value, model)
            self.counter = 0

    def save_checkpoint(self, metric_value: float, model: torch.nn.Module or tensorflow.keras.Model):
        """
        Cохраняет модель, если валидационная метрика улучшилась.

        Параметры:
        ----------
        metric_value: float
            Значение валидационной метрики.
        model: torch.nn.Module or tensorflow.keras.Model
            Обучаемая модель.

        Возвращаемое значение:
        ----------------------
        None
        """
        if self.verbose:
            print(
                f"Validation {self.metric_name} improved ({self.best_prev_score:.6f} --> {metric_value:.6f}).  Saving model...")
        if self.save_format == "tf":
            model.save_weights(self.save_path)
        else:
            torch.save(model.state_dict(), self.save_path)

        self.best_prev_score = metric_value
