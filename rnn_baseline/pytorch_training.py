from typing import List
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm_notebook
from sklearn.metrics import roc_auc_score

from rnn_baseline.data_generators import batches_generator


def train_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, dataset_train: List[str],
                batch_size: int = 64, shuffle: bool = True, print_loss_every_n_batches: int = 500,
                device: torch.device = None):
    """
    Делает одну эпоху обучения модели, логируя промежуточные значения функции потерь.

    Параметры:
    -----------
    model: torch.nn.Module
        Обучаемая модель.
    optimizer: torch.optim.Optimizer
        Оптимизатор.
    dataset_train: List[str]
        Список путей до файлов с предобработанными последовательностями.
    batch_size: int, default=64
        Размер батча.
    shuffle: bool, default=False
        Перемешивать ли данные перед подачей в модель.
    print_loss_every_n_batches: int, default=500
        Число батчей.
    device: torch.device, default=None
        Девайс, на который переместить данные.

    Возвращаемое значение:
    ----------------------
    None
    """
    model.train()
    loss_function = nn.BCEWithLogitsLoss(reduction="none")
    losses = torch.LongTensor().to(device)
    samples_counter = 0
    train_generator = batches_generator(dataset_train, batch_size=batch_size, shuffle=shuffle,
                                        device=device, is_train=True, output_format="torch")

    for num_batch, batch in tqdm_notebook(enumerate(train_generator, start=1), desc="Training"):
        output = torch.flatten(model(batch["features"]))
        batch_loss = loss_function(output, batch["label"].float())
        batch_loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()

        samples_counter += batch_loss.size(0)

        losses = torch.cat([losses, batch_loss], dim=0)
        if num_batch % print_loss_every_n_batches == 0:
            print(f"Batches {num_batch - print_loss_every_n_batches + 1} - {num_batch} loss:"
                  f"{losses[-samples_counter:].mean()}", end="\r")
            samples_counter = 0

    print(f"Training loss after epoch: {losses.mean()}", end="\r")


def eval_model(model: torch.nn.Module, dataset_val: List[str], batch_size: int = 32, device: torch.device = None) -> float:
    """
    Скорит выборку моделью и вычисляет метрику ROC AUC.

    Параметры:
    -----------
    model: torch.nn.Module
        Модель, которой необходимо проскорить выборку.
    dataset_val: List[str]
        Список путей до файлов с предобработанными последовательностями.
    batch_size: int, default=32
        Размер батча.
    device: torch.device, default=None
        Девайс, на который переместить данные.

    Возвращаемое значение:
    ----------------------
    auc: float
    """
    model.eval()
    preds = []
    targets = []
    val_generator = batches_generator(dataset_val, batch_size=batch_size, shuffle=False,
                                      device=device, is_train=True, output_format="torch")

    for batch in tqdm_notebook(val_generator, desc="Evaluating model"):
        targets.extend(batch["label"].detach().cpu().numpy().flatten())
        output = model(batch["features"])
        preds.extend(output.detach().cpu().numpy().flatten())

    return roc_auc_score(targets, preds)


def inference(model: torch.nn.Module, dataset_test: List[str], batch_size: int = 32, device: torch.device = None) -> pd.DataFrame:
    """
    Скорит выборку моделью.

    Параметры:
    -----------
    model: torch.nn.Module
        Модель, которой необходимо проскорить выборку.
    dataset_test: List[str]
        Список путей до файлов с предобработанными последовательностями.
    batch_size: int, default=32
        Размер батча.
    device: torch.device, default=None
        Девайс, на который переместить данные.

    Возвращаемое значение:
    ----------------------
    scores: pandas.DataFrame
        Датафрейм с двумя колонками: "id" - идентификатор заявки и "score" - скор модели.
    """
    model.eval()
    preds = []
    ids = []
    test_generator = batches_generator(dataset_test, batch_size=batch_size, shuffle=False,
                                       verbose=False, device=device, is_train=False,
                                       output_format="torch")

    for batch in tqdm_notebook(test_generator, desc="Test predictions"):
        ids.extend(batch["id_"])
        output = model(batch["features"])
        preds.extend(output.detach().cpu().numpy().flatten())

    return pd.DataFrame({
        "id": ids,
        "score": preds
    })
