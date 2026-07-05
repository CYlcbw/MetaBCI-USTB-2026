# -*- coding: utf-8 -*-
# License: MIT License
"""
MI online analysis.

This demo keeps the original four-class online flow and only changes the
online EEG input to the MetaBCI OpenBCI amplifier.
"""
import os
import sys
import time
import warnings

import numpy as np
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
from metabci.brainflow.amplifiers import OpenBCI
from metabci.brainflow.workers import command_output

import keyboard

warnings.filterwarnings("ignore")


# ===== 参数配置 =====
OPENBCI_STREAM_NAME = "obci_eeg1"
OPENBCI_MARKER_STREAM_NAME = "obci_eeg2"
SOCKET_HOST = "0.0.0.0"
SOCKET_PORT = 12345

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "ustb2025mi4c-main",
    "Dataset",
    "base8_s01_s04_final",
    "S01_base8_all_runs.npz",
)
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
OPENBCI_INPUT_SAMPLING_RATE = 0
CLASSES_NUMBER = 4
KFOLDS = 5
PREDICTION_TO_LABEL = {0: "left", 1: "right", 2: "retreat", 3: "forward"}
PREDICTION_TO_COMMAND = {idx: str(idx + 1) for idx in PREDICTION_TO_LABEL}
FIRST_RUN = True
MAX_RUNS = 7


def set_random_seed(seed_value=20250702):
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True


def get_data_new(subjects):
    with np.load(DATA_PATH, allow_pickle=True) as npz:
        epoch_data = npz["X"].astype(np.float32)
        labels = npz["y"].astype(np.int64)

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


def get_openbci_input_srate(device):
    if OPENBCI_INPUT_SAMPLING_RATE:
        return float(OPENBCI_INPUT_SAMPLING_RATE)
    try:
        nominal_srate = float(device.stream_metadata.get("nominal_srate", 0))
    except (TypeError, ValueError):
        nominal_srate = 0
    if nominal_srate > 0:
        return nominal_srate
    return float(EEG_SAMPLING_RATE)


def resample_to_model_rate(samples):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.shape[1] == WINDOW_LENGTH_SAMPLES:
        return np.ascontiguousarray(samples, dtype=np.float32)
    old_x = np.linspace(0, 1, samples.shape[1], endpoint=False)
    new_x = np.linspace(0, 1, WINDOW_LENGTH_SAMPLES, endpoint=False)
    out = np.empty((samples.shape[0], WINDOW_LENGTH_SAMPLES), dtype=np.float32)
    for ch in range(samples.shape[0]):
        out[ch] = np.interp(new_x, old_x, samples[ch]).astype(np.float32)
    return out


def update_openbci_buffer(eeg_buffer, rows, input_window_samples):
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or rows.size == 0:
        return eeg_buffer
    samples = rows[:, :USED_CHANNELS].T
    new_samples = samples.shape[1]
    if new_samples >= input_window_samples:
        eeg_buffer = samples[:, -input_window_samples:]
    else:
        eeg_buffer = np.roll(eeg_buffer, -new_samples, axis=1)
        eeg_buffer[:, -new_samples:] = samples
    return eeg_buffer


if __name__ == "__main__":
    subjects = [1]
    device = None

    # 1. 初始化 Socket
    server_socket, client_socket = command_output(SOCKET_HOST, SOCKET_PORT)

    # 2. 初始化 OpenBCI GUI LSL 数据流
    print(f"正在搜索 LSL EEG 流: {OPENBCI_STREAM_NAME}...")
    device = OpenBCI(
        stream_name=OPENBCI_STREAM_NAME,
        stream_type="EEG",
        source_id="openbcigui",
        timeout=5.0,
        max_buflen=3,
        pull_timeout=0.1,
        max_samples=256,
        eeg_channel_indices=list(range(USED_CHANNELS)),
        expected_channels=USED_CHANNELS,
        strict_channels=False,
        marker_stream_name=OPENBCI_MARKER_STREAM_NAME,
        ignore_zero_markers=True,
    )
    device.start_acq()
    print(f"已连接到 EEG 流: {OPENBCI_STREAM_NAME}")

    # 3. 数据缓存队列（用于存储最近的 EEG 数据）
    input_srate = get_openbci_input_srate(device)
    input_window_samples = int(round(input_srate * WINDOW_LENGTH_SEC))
    device.max_samples = max(1, int(round(input_srate * 0.1)))
    eeg_buffer = np.zeros((USED_CHANNELS, input_window_samples), dtype=np.float32)
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
    socket_data = None
    run_times = 0

    try:
        print("开始实时解码（按 Ctrl+C 停止）...")
        while True:
            rows = device.recv()
            if rows:
                eeg_buffer = update_openbci_buffer(eeg_buffer, rows, input_window_samples)
                eeg_window = resample_to_model_rate(eeg_buffer)

            decode_allowed = first_run or (socket_data == "arrived")
            if decode_allowed and eeg_buffer.shape[1] == input_window_samples:
                if first_run:
                    first_run = False

                start_time = time.time()
                while time.time() - start_time < 5:
                    rows = device.recv()
                    if rows:
                        eeg_buffer = update_openbci_buffer(eeg_buffer, rows, input_window_samples)
                    eeg_window = resample_to_model_rate(eeg_buffer)
                    time.sleep(0.01)

                eeg_window = scaler.fit_transform(eeg_window)
                input_data = eeg_window.reshape(1, *eeg_window.shape)
                if isinstance(input_data, torch.Tensor):
                    X_test = input_data.to(torch.float32)
                else:
                    X_test = torch.from_numpy(input_data).to(torch.float32)
                prediction = model(X_test)
                prediction = int(softmax(prediction, dim=-1).argmax(dim=-1).numpy()[0])
                label = PREDICTION_TO_LABEL[prediction]
                command = PREDICTION_TO_COMMAND[prediction]
                client_socket.sendall(command.encode("ascii"))
                print(f"解码结果: {label}, command={command}")
                try:
                    socket_data = client_socket.recv(1024).decode("ascii")
                    if socket_data:
                        print(f"Received from client: {socket_data}")
                        if "arrived" in socket_data:
                            socket_data = "arrived"
                except:
                    print("Error receiving data")
                run_times += 1
                if MAX_RUNS and run_times >= MAX_RUNS:
                    break

    except KeyboardInterrupt:
        print("用户终止程序...")
    finally:
        if device is not None:
            device.stop_acq()
        client_socket.close()
        server_socket.close()
        print("Server disconnected.")
