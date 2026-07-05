# -*- coding: utf-8 -*-
"""
MI4C offline validation using the USTB2026MI4C base8 dataset class.
"""
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from metabci.brainda.algorithms.deep_learning import EEG_Conformer
from metabci.brainda.algorithms.utils.model_selection import (
    model_training_two_stage,
    test_with_cross_validate,
)
from metabci.brainda.datasets.ustb2026mi4c import USTB2026MI4C


warnings.filterwarnings("ignore")


SUBJECTS = [1]
WINDOW_LENGTH = 256
STRIDE = 256
TEST_SIZE = 0.2
RANDOM_STATE = 20250702
KFOLDS = 5

MODEL_NAME = "EEG_Conformer"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(SCRIPT_DIR, "checkpoints", MODEL_NAME + "_ustb2025mi4c_base8")


def set_seed(seed_value):
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True


def split_trials(epoch_data, labels):
    n_epochs, _, n_times = epoch_data.shape
    n_segments = n_times // STRIDE
    segmented_data = []
    segmented_labels = []

    for i in range(n_epochs):
        for j in range(n_segments):
            start = j * STRIDE
            end = start + WINDOW_LENGTH
            if end > n_times:
                continue
            segmented_data.append(epoch_data[i, :, start:end])
            segmented_labels.append(labels[i])

    return np.asarray(segmented_data), np.asarray(segmented_labels)


def get_data_from_dataset(dataset, subject):
    arrays = dataset.load_subject_arrays(subject)
    train_feature_arr, train_label_arr = split_trials(arrays["X"], arrays["y"])

    scaler = StandardScaler()
    data_reshaped = train_feature_arr.reshape(train_feature_arr.shape[0], -1)
    data_normalized = scaler.fit_transform(data_reshaped)
    train_feature_arr = data_normalized.reshape(train_feature_arr.shape)

    train_feature, test_feature, train_label, test_label = train_test_split(
        train_feature_arr,
        train_label_arr,
        test_size=TEST_SIZE,
        shuffle=True,
        random_state=RANDOM_STATE,
        stratify=train_label_arr,
    )

    return train_feature, train_label, test_feature, test_label


def has_kfold_models(model_save_path, kfolds):
    if not os.path.isdir(model_save_path):
        return False
    files = [name for name in os.listdir(model_save_path) if name.endswith(".pth")]
    return all(any("fold{}_".format(kfold) in name for name in files) for kfold in range(kfolds))


def build_model(X, y, device):
    model = EEG_Conformer(
        n_channels=X.shape[1],
        n_samples=X.shape[2],
        n_classes=np.unique(y).size,
    )
    return model.module.to(dtype=torch.float32).to(device)


def offline_validation_subject(X, y, subject, X_test, y_test):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    device_name = torch.cuda.get_device_name(device) if use_cuda else "cpu"
    print("current device:", device_name)

    set_seed(RANDOM_STATE)

    subject_arg = [subject]
    model_save_path = os.path.join(MODEL_ROOT, "Subject_0{}".format(subject))
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    model = build_model(X, y, device)

    if not has_kfold_models(model_save_path, KFOLDS):
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            betas=(0.7, 0.999),
            weight_decay=0.001,
        )
        model_training_two_stage(
            model,
            criterion,
            optimizer,
            None,
            1000,
            100,
            100,
            64,
            X,
            y,
            KFOLDS,
            device,
            MODEL_NAME,
            subject_arg,
            model_save_path,
        )

    features_path = os.path.join(
        MODEL_ROOT,
        "Subject_0{}".format(subject),
        "visualization",
        "tsne_feature",
    )
    if not os.path.exists(features_path):
        os.makedirs(features_path)

    test_with_cross_validate(
        model,
        device,
        X_test,
        y_test,
        model_save_path,
        KFOLDS,
        subject_arg,
        visual=False,
        features_path=features_path,
    )


if __name__ == "__main__":
    dataset = USTB2026MI4C()
    for subject in SUBJECTS:
        X, y, test_feature_arr, test_label_arr = get_data_from_dataset(dataset, subject)
        offline_validation_subject(
            X,
            y,
            subject=subject,
            X_test=test_feature_arr,
            y_test=test_label_arr,
        )
