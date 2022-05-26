from typing import List
import numpy as np
import pickle
import torch


features = ["pre_since_opened", "pre_since_confirmed", "pre_pterm", "pre_fterm", "pre_till_pclose", "pre_till_fclose",
            "pre_loans_credit_limit", "pre_loans_next_pay_summ", "pre_loans_outstanding", "pre_loans_total_overdue",
            "pre_loans_max_overdue_sum", "pre_loans_credit_cost_rate",
            "pre_loans5", "pre_loans530", "pre_loans3060", "pre_loans6090", "pre_loans90",
            "is_zero_loans5", "is_zero_loans530", "is_zero_loans3060", "is_zero_loans6090", "is_zero_loans90",
            "pre_util", "pre_over2limit", "pre_maxover2limit", "is_zero_util", "is_zero_over2limit", "is_zero_maxover2limit",
            "enc_paym_0", "enc_paym_1", "enc_paym_2", "enc_paym_3", "enc_paym_4", "enc_paym_5", "enc_paym_6", "enc_paym_7", "enc_paym_8",
            "enc_paym_9", "enc_paym_10", "enc_paym_11", "enc_paym_12", "enc_paym_13", "enc_paym_14", "enc_paym_15", "enc_paym_16",
            "enc_paym_17", "enc_paym_18", "enc_paym_19", "enc_paym_20", "enc_paym_21", "enc_paym_22", "enc_paym_23", "enc_paym_24",
            "enc_loans_account_holder_type", "enc_loans_credit_status", "enc_loans_credit_type", "enc_loans_account_cur",
            "pclose_flag", "fclose_flag"]


def batches_generator(list_of_paths: List[str], batch_size: int = 32, shuffle: bool = False,
                      is_infinite: bool = False, verbose: bool = False, device: torch.device = None,
                      output_format: str = "torch", is_train: bool = True):
    """
    Создает батчи на вход рекуррентных нейронных сетей, реализованных на фреймворках tensorflow и pytorch.

    Параметры:
    -----------
    list_of_paths: List[str]
        Список путей до файлов с предобработанными последовательностями.
    batch_size: int, default=32
        Размер батча.
    shuffle: bool, default=False
        Перемешивать ли данные перед генерацией батчей.
    is_infinite: bool, default=False
        Должен ли генератор быть бесконечным.
    verbose: bool, default=False
        Печатать ли имя текущего обрабатываемого файла.
    device: torch.device, default=None
        Девайс, на который переместить данные при ``output_format``="torch". Игнорируется, если ``output_format``="tf".
    output_format: str, default="torch"
        Формат возвращаемых данных. Допустимые значения: "torch", "tf".
        Если "torch", то возвращает словарь, с ключами "id_", "features" и "label", если is_train=True,
        и содержащий идентификаторы заявок, признаки и тагрет соответственно.
        Признаки и таргет помещаются на девайс, указанный в ``device``.
        Если "tf", то возращает кортеж (признаки, таргет), если ``is_train``=True, и кортеж (признаки, идентификаторы заявок) иначе.
    is_train: bool, default=True
        Используется ли генератор для обучения модели или для инференса.

    Возвращаемое значение:
    ----------------------
    result: dict or tuple
        Выходной словарь или кортеж в зависимости от параметра ``output_format``.
    """
    if output_format not in ["torch", "tf"]:
        raise ValueError("Unknown format. Please choose one of the following formats: \"torch\", \"tf\"")

    while True:
        if shuffle:
            np.random.shuffle(list_of_paths)

        for path in list_of_paths:
            if verbose:
                print(f"Reading {path}")

            with open(path, "rb") as f:
                data = pickle.load(f)

            ids, padded_sequences, targets = data["id"], data["padded_sequences"], data["target"]
            indices = np.arange(len(ids))
            if shuffle:
                np.random.shuffle(indices)
                ids = ids[indices]
                padded_sequences = padded_sequences[indices]
                if is_train:
                    targets = targets[indices]

            for idx in range(len(ids)):
                bucket_ids = ids[idx]
                bucket = padded_sequences[idx]
                if is_train:
                    bucket_targets = targets[idx]

                for jdx in range(0, len(bucket), batch_size):
                    batch_ids = bucket_ids[jdx: jdx + batch_size]
                    batch_sequences = bucket[jdx: jdx + batch_size]
                    if is_train:
                        batch_targets = bucket_targets[jdx: jdx + batch_size]

                    if output_format == "tf":
                        batch_sequences = [batch_sequences[:, i] for i in range(len(features))]

                        if is_train:
                            yield batch_sequences, batch_targets
                        else:
                            yield batch_sequences, batch_ids
                    else:
                        batch_sequences = [torch.LongTensor(batch_sequences[:, i]).to(device) for i in range(len(features))]
                        if is_train:
                            yield dict(id_=batch_ids,
                                       features=batch_sequences,
                                       label=torch.LongTensor(batch_targets).to(device))
                        else:
                            yield dict(id_=batch_ids,
                                       features=batch_sequences)
        if not is_infinite:
            break
