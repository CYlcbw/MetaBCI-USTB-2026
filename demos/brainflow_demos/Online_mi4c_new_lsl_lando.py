# -*- coding: utf-8 -*-
"""Simulated full-chain runner for Online_mi4c_new_lsl.py.

This script starts a local simulated LSL EEG stream and a local socket client,
then runs the same real dataset validation, model loading, and seven-step
online decoding flow used by the NeuroScan Curry LSL online demo.
"""

import os
import socket
import sys
import threading
import time

import numpy as np
import pylsl
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import softmax

from metabci.brainda.algorithms.deep_learning import EEG_Conformer
from metabci.brainflow.workers import command_output

from Online_mi4c_new_lsl import (
    CLASSES_NUMBER,
    EEG_SAMPLING_RATE,
    LSL_EEG_STREAM_NAME,
    MAX_RUNS,
    OUTPUT_INTERVAL_SEC,
    PREDICTION_TO_COMMAND,
    PREDICTION_TO_LABEL,
    SOCKET_PORT,
    USED_CHANNELS,
    WINDOW_LENGTH_SAMPLES,
    get_data_new,
    get_online_model_path,
    offline_validation_new,
    resolve_lsl_stream_by_name,
    update_lsl_buffer,
    wait_and_update_lsl_buffer,
)
from simulate_online_lsl_streams import create_eeg_outlet, make_eeg_chunk


CLIENT_CONNECT_HOST = "127.0.0.1"
LOCAL_SOCKET_HOST = "127.0.0.1"
SIMULATED_LSL_CHUNK_SIZE = 16


def start_simulated_lsl_stream(stop_event):
    rng = np.random.default_rng(20260714)
    outlet = create_eeg_outlet(
        LSL_EEG_STREAM_NAME,
        "lando_simulated_curry",
        USED_CHANNELS,
        EEG_SAMPLING_RATE,
    )
    sample_index = 0
    chunk_sleep = SIMULATED_LSL_CHUNK_SIZE / float(EEG_SAMPLING_RATE)
    print(f"[lando-lsl] 模拟 LSL EEG 流已启动: {LSL_EEG_STREAM_NAME}")

    while not stop_event.is_set():
        chunk = make_eeg_chunk(
            sample_index,
            SIMULATED_LSL_CHUNK_SIZE,
            USED_CHANNELS,
            EEG_SAMPLING_RATE,
            rng,
        )
        outlet.push_chunk(chunk.tolist())
        sample_index += SIMULATED_LSL_CHUNK_SIZE
        time.sleep(chunk_sleep)

    print("[lando-lsl] 模拟 LSL EEG 流已停止")


def start_simulated_socket_client(stop_event):
    client_socket = None
    while not stop_event.is_set():
        try:
            client_socket = socket.create_connection(
                (CLIENT_CONNECT_HOST, SOCKET_PORT),
                timeout=1.0,
            )
            break
        except OSError:
            time.sleep(0.2)

    if client_socket is None:
        return

    client_socket.settimeout(None)
    print(f"[lando-client] 已连接模拟客户端: {CLIENT_CONNECT_HOST}:{SOCKET_PORT}")
    with client_socket:
        receive_count = 0
        while not stop_event.is_set() and receive_count < MAX_RUNS:
            command_bytes = client_socket.recv(1024)
            if not command_bytes:
                break
            command = command_bytes.decode("ascii", errors="ignore")
            receive_count += 1
            print(f"[lando-client] 收到控制指令({receive_count}/{MAX_RUNS}): {command}")
            client_socket.sendall(b"arrived")

    print("[lando-client] 模拟客户端已结束")


def load_online_model():
    model = EEG_Conformer(
        n_channels=USED_CHANNELS,
        n_samples=EEG_SAMPLING_RATE,
        n_classes=CLASSES_NUMBER,
    )
    model = model.module.to(dtype=torch.float32)
    model_path = get_online_model_path()
    print(f"加载在线模型: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_lando_online_flow():
    subjects = [1]
    stop_event = threading.Event()
    lsl_thread = threading.Thread(
        target=start_simulated_lsl_stream,
        args=(stop_event,),
        daemon=True,
    )
    client_thread = threading.Thread(
        target=start_simulated_socket_client,
        args=(stop_event,),
        daemon=True,
    )

    lsl_thread.start()
    client_thread.start()

    server_socket = None
    client_socket = None

    try:
        # 1. 初始化 Socket，并由本脚本内部模拟客户端连接。
        server_socket, client_socket = command_output(LOCAL_SOCKET_HOST, SOCKET_PORT)

        # 2. 初始化 LSL EEG 数据流。
        print(f"正在搜索 LSL EEG 流: {LSL_EEG_STREAM_NAME}...")
        streams = resolve_lsl_stream_by_name(LSL_EEG_STREAM_NAME)
        if not streams:
            raise RuntimeError(f"未找到 EEG 流: {LSL_EEG_STREAM_NAME}")
        eeg_inlet = pylsl.StreamInlet(streams[0])
        print(f"已连接到 EEG 流: {eeg_inlet.info().name()}")

        # 3. 加载离线数据并执行真实离线验证。
        X, y, test_feature_arr, test_label_arr = get_data_new(subjects)
        offline_validation_new(
            X,
            y,
            subject=subjects,
            X_test=test_feature_arr,
            y_test=test_label_arr,
        )

        # 4. 加载真实在线模型。
        model = load_online_model()

        # 5. 使用模拟 LSL 数据执行 7 次在线解码输出。
        eeg_buffer = np.zeros((USED_CHANNELS, WINDOW_LENGTH_SAMPLES), dtype=np.float32)
        scaler = StandardScaler()
        first_run = True
        run_times = 0
        socket_data = None

        print("lando 模拟模式：无需按空格，开始实时解码...")

        while run_times < MAX_RUNS:
            eeg_buffer = update_lsl_buffer(eeg_inlet, eeg_buffer)
            print("eeg_buffer shape: ", eeg_buffer.shape)

            decode_allowed = first_run or (socket_data == "arrived")
            if not decode_allowed:
                continue
            if first_run:
                first_run = False

            eeg_buffer = wait_and_update_lsl_buffer(
                eeg_inlet,
                eeg_buffer,
                OUTPUT_INTERVAL_SEC,
            )
            print("eeg_buffer shape: ", eeg_buffer.shape)

            eeg_window = scaler.fit_transform(eeg_buffer)
            input_data = eeg_window.reshape(1, *eeg_window.shape)
            X_test = torch.from_numpy(input_data).to(torch.float32)
            print("X_test shape: ", X_test.shape)

            prediction = model(X_test)
            prediction = int(softmax(prediction, dim=-1).argmax(dim=-1).numpy()[0])
            label = PREDICTION_TO_LABEL[prediction]
            command = PREDICTION_TO_COMMAND[prediction]
            client_socket.send(command.encode("ascii"))

            run_times += 1
            print(f"解码结果: {label}")

            try:
                socket_data = client_socket.recv(1024).decode("ascii")
                if socket_data:
                    print(f"Received from client: {socket_data}")
            except Exception:
                print("Error receiving data")

    finally:
        stop_event.set()
        if client_socket is not None:
            client_socket.close()
        if server_socket is not None:
            server_socket.close()
        client_thread.join(timeout=2.0)
        lsl_thread.join(timeout=2.0)
        print("Server disconnected.")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_lando_online_flow()
