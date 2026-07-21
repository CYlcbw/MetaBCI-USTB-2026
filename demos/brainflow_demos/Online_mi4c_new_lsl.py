# -*- coding: utf-8 -*-
# License: MIT License
"""
MI online analysis.

This demo keeps the original four-class online flow and only changes the
online EEG input to an 8-channel LSL stream.
"""
import os
import sys
import time
import warnings

import numpy as np
import pylsl
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import softmax


BASE_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from metabci.brainda.algorithms.deep_learning import EEG_Conformer
from metabci.brainda.algorithms.utils.model_selection import (
    model_training_two_stage,
    test_with_cross_validate,
)
from metabci.brainda.datasets import USTB2026MI4C
from metabci.brainflow.workers import command_output

import keyboard

warnings.filterwarnings("ignore")


# ===== 参数配置 =====
LSL_EEG_STREAM_NAME = "CURRYStream"
SOCKET_HOST = "192.168.3.17"
SOCKET_PORT = 12345

MODEL_SAVE_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "EEG_Conformer_ustb2025mi4c_base8",
    "Subject_01",
)
MODEL_PATH = ""
MODEL_FOLD = 2

USED_CHANNELS = 8
EEG_SAMPLING_RATE = 256
WINDOW_LENGTH_SEC = 1
WINDOW_LENGTH_SAMPLES = EEG_SAMPLING_RATE * WINDOW_LENGTH_SEC
CLASSES_NUMBER = 4
KFOLDS = 5
PREDICTION_TO_LABEL = {0: "left", 1: "right", 2: "retreat", 3: "forward"}
PREDICTION_TO_COMMAND = {idx: str(idx + 1) for idx in PREDICTION_TO_LABEL}
FIRST_RUN = True
MAX_RUNS = 7
OUTPUT_INTERVAL_SEC = 5


def set_random_seed(seed_value=20250702):
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True


def get_data_new(subjects):
    subject = int(subjects[0] if isinstance(subjects, (list, tuple)) else subjects)
    arrays = USTB2026MI4C().load_subject_arrays(subject)
    epoch_data = arrays["X"].astype(np.float32)
    labels = arrays["y"].astype(np.int64)

    n_epochs, _, n_times = epoch_data.shape
    window_length = WINDOW_LENGTH_SAMPLES
    stride = WINDOW_LENGTH_SAMPLES
    n_segments = n_times // stride
    segmented_data = []
    label_run_expanded = []
    for i in range(n_epochs):
        for j in range(n_segments):
            start = j * stride
            end = start + window_length
            segmented_data.append(epoch_data[i, :, start:end])
            label_run_expanded.append(labels[i])

    train_feature_arr = np.asarray(segmented_data, dtype=np.float32)
    train_label_arr = np.asarray(label_run_expanded, dtype=np.int64)
    scaler = StandardScaler()
    data_reshaped = train_feature_arr.reshape(train_feature_arr.shape[0], -1)
    data_normalized = scaler.fit_transform(data_reshaped)
    train_feature_arr = data_normalized.reshape(train_feature_arr.shape).astype(np.float32)

    train_feature, test_feature, train_label, test_label = train_test_split(
        train_feature_arr,
        train_label_arr,
        test_size=0.2,
        shuffle=True,
        random_state=20250702,
        stratify=train_label_arr,
    )

    return train_feature, train_label, test_feature, test_label


def find_fold_model(model_save_path, fold):
    if not os.path.exists(model_save_path):
        return None
    for filename in sorted(os.listdir(model_save_path)):
        if "fold{}_".format(fold) in filename and filename.endswith(".pth"):
            return os.path.join(model_save_path, filename)
    return None


def has_all_fold_models(model_save_path, kfolds):
    return all(find_fold_model(model_save_path, fold) is not None for fold in range(kfolds))


def train_fold_models_if_needed(X, y, subject, device):
    if has_all_fold_models(MODEL_SAVE_PATH, KFOLDS):
        return

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model = EEG_Conformer(
        n_channels=X.shape[1],
        n_samples=X.shape[2],
        n_classes=np.unique(y).size,
    )
    model = model.module.to(dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.7, 0.999),
        weight_decay=0.001,
    )
    lr_scheduler = None
    frist_epochs = 1000
    eary_stop_epoch = 100
    second_epochs = 100
    batch_size = 64
    model_training_two_stage(
        model,
        criterion,
        optimizer,
        lr_scheduler,
        frist_epochs,
        eary_stop_epoch,
        second_epochs,
        batch_size,
        X,
        y,
        KFOLDS,
        device,
        "EEG_Conformer",
        subject,
        MODEL_SAVE_PATH,
    )


def offline_validation_new(X, y, subject, X_test=None, y_test=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("current device:", device)
    set_random_seed()

    train_fold_models_if_needed(X, y, subject, device)

    model = EEG_Conformer(
        n_channels=X.shape[1],
        n_samples=X.shape[2],
        n_classes=np.unique(y).size,
    )
    model = model.module.to(dtype=torch.float32).to(device)
    features_path = os.path.join(MODEL_SAVE_PATH, "visualization", "tsne_feature")
    os.makedirs(features_path, exist_ok=True)
    test_with_cross_validate(
        model,
        device,
        X_test,
        y_test,
        MODEL_SAVE_PATH,
        KFOLDS,
        subject,
        visual=False,
        features_path=features_path,
    )


def get_online_model_path():
    if MODEL_PATH:
        return MODEL_PATH
    model_path = find_fold_model(MODEL_SAVE_PATH, MODEL_FOLD)
    if model_path is None:
        raise FileNotFoundError(
            "未找到 fold{} 在线模型，请先确认五折模型已训练完成: {}".format(
                MODEL_FOLD,
                MODEL_SAVE_PATH,
            )
        )
    return model_path


def preprocess_samples(samples):
    samples = np.asarray(samples, dtype=np.float32).T
    if samples.shape[0] < USED_CHANNELS:
        raise ValueError(
            "LSL channel count mismatch: expected at least {}, got {}".format(
                USED_CHANNELS,
                samples.shape[0],
            )
        )
    return np.ascontiguousarray(samples[:USED_CHANNELS, :], dtype=np.float32)


def resolve_lsl_stream_by_name(stream_name, timeout=5.0):
    resolver = getattr(pylsl, "resolve_byprop", None)
    if resolver is not None:
        return resolver("name", stream_name, minimum=1, timeout=timeout)
    return [
        stream for stream in pylsl.resolve_streams(wait_time=timeout)
        if stream.name() == stream_name
    ]


def update_lsl_buffer(eeg_inlet, eeg_buffer):
    samples, timestamps = eeg_inlet.pull_chunk(timeout=0.1)
    if not samples:
        return eeg_buffer
    samples_cleaned = preprocess_samples(samples)
    new_samples = samples_cleaned.shape[1]
    if new_samples >= WINDOW_LENGTH_SAMPLES:
        return samples_cleaned[:, -WINDOW_LENGTH_SAMPLES:]
    eeg_buffer = np.roll(eeg_buffer, -new_samples, axis=1)
    eeg_buffer[:, -new_samples:] = samples_cleaned
    return eeg_buffer


def wait_and_update_lsl_buffer(eeg_inlet, eeg_buffer, wait_seconds):
    wait_start = time.time()
    while time.time() - wait_start < wait_seconds:
        eeg_buffer = update_lsl_buffer(eeg_inlet, eeg_buffer)
        print("eeg_buffer shape: ", eeg_buffer.shape)
        time.sleep(0.01)
    return eeg_buffer


if __name__ == "__main__":
    subjects = [1]

    # 1. 初始化 Socket
    server_socket, client_socket = command_output(SOCKET_HOST, SOCKET_PORT)

    # 2. 初始化 LSL EEG 数据流
    print(f"正在搜索 LSL EEG 流: {LSL_EEG_STREAM_NAME}...")
    streams = resolve_lsl_stream_by_name(LSL_EEG_STREAM_NAME)
    if not streams:
        raise RuntimeError(f"未找到 EEG 流: {LSL_EEG_STREAM_NAME}")
    eeg_inlet = pylsl.StreamInlet(streams[0])
    print(f"已连接到 EEG 流: {eeg_inlet.info().name()}")

    # 3. 数据缓存队列（用于存储最近的 EEG 数据）
    eeg_buffer = np.zeros((USED_CHANNELS, WINDOW_LENGTH_SAMPLES), dtype=np.float32)
    scaler = StandardScaler()

    # 4. 加载离线数据，计算离线准确率
    X, y, test_feature_arr, test_label_arr = get_data_new(subjects)
    offline_validation_new(
        X,
        y,
        subject=subjects,
        X_test=test_feature_arr,
        y_test=test_label_arr,
    )

    # 5. 加载预训练模型
    model = EEG_Conformer(
        n_channels=USED_CHANNELS,
        n_samples=EEG_SAMPLING_RATE,
        n_classes=CLASSES_NUMBER,
    )
    model = model.module.to(dtype=torch.float32)
    state_dict = torch.load(get_online_model_path(), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 6. 开始实时解码
    print("按空格键开始实时解码...")
    keyboard.wait("space")

    first_run = FIRST_RUN
    run_times = 0
    socket_data = None

    try:
        print("开始实时解码（按 Ctrl+C 停止）...")
        while True:
            eeg_buffer = update_lsl_buffer(eeg_inlet, eeg_buffer)
            print("eeg_buffer shape: ", eeg_buffer.shape)

            # 检查是否满足解码条件
            decode_allowed = first_run or (socket_data == "arrived")
            if decode_allowed and eeg_buffer.shape[1] == WINDOW_LENGTH_SAMPLES:
                if first_run:
                    first_run = False

                # 执行解码（等待5秒期间持续更新数据）
                eeg_buffer = wait_and_update_lsl_buffer(
                    eeg_inlet,
                    eeg_buffer,
                    OUTPUT_INTERVAL_SEC,
                )
                print("eeg_buffer shape: ", eeg_buffer.shape)

                # 5秒后调用模型并发送结果
                eeg_window = scaler.fit_transform(eeg_buffer)
                input_data = eeg_window.reshape(1, *eeg_window.shape)
                if isinstance(input_data, torch.Tensor):
                    X_test = input_data.to(torch.float32)
                else:
                    X_test = torch.from_numpy(input_data).to(torch.float32)
                print("X_test shape: ", X_test.shape)
                prediction = model(X_test)
                prediction = int(softmax(prediction, dim=-1).argmax(dim=-1).numpy()[0])
                label = PREDICTION_TO_LABEL[prediction]
                command = PREDICTION_TO_COMMAND[prediction]
                message = command.encode("ascii")
                client_socket.send(message)
                print(f"解码结果: {label}")
                try:
                    socket_data = client_socket.recv(1024).decode("ascii")
                    if socket_data:
                        print(f"Received from client: {socket_data}")
                except Exception:
                    print("Error receiving data")
                run_times += 1
                if MAX_RUNS and run_times >= MAX_RUNS:
                    break

    except KeyboardInterrupt:
        print("用户终止程序...")
    finally:
        client_socket.close()
        server_socket.close()
        print("Server disconnected.")
