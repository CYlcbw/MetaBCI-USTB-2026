# -*- coding: utf-8 -*-
"""Receive and print OpenBCI GUI LSL data.

This demo verifies that the MetaBCI OpenBCI adapter can connect to the
OpenBCI GUI EEG LSL stream and convert incoming samples into MetaBCI rows.
"""

import argparse
import os
import sys
import time

import numpy as np


BASE_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from metabci.brainflow.amplifiers import OpenBCI


def build_parser():
    parser = argparse.ArgumentParser(
        description="Receive OpenBCI GUI EEG LSL data and print MetaBCI rows."
    )
    parser.add_argument("--stream-name", default="obci_eeg1")
    parser.add_argument("--marker-stream-name", default="")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--max-chunks", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--pull-timeout", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=64)
    return parser


def main():
    args = build_parser().parse_args()
    device = OpenBCI(
        stream_name=args.stream_name,
        stream_type="EEG",
        source_id="openbcigui",
        timeout=args.timeout,
        max_buflen=3,
        pull_timeout=args.pull_timeout,
        max_samples=args.max_samples,
        eeg_channel_indices=list(range(args.channels)),
        expected_channels=args.channels,
        strict_channels=False,
        marker_stream_name=args.marker_stream_name or None,
        ignore_zero_markers=True,
    )

    try:
        print(f"正在搜索 OpenBCI GUI LSL EEG 流: {args.stream_name}...")
        device.start_acq()
        print("已连接 OpenBCI GUI LSL EEG 流")
        print("stream info:", device.get_stream_info())
        print("channel labels:", device.get_channel_labels())
        print("开始接收数据，按 Ctrl+C 可提前停止...")

        received_chunks = 0
        while received_chunks < args.max_chunks:
            rows = device.recv()
            if not rows:
                time.sleep(0.05)
                continue

            data = np.asarray(rows, dtype=np.float64)
            received_chunks += 1
            eeg = data[:, :args.channels]
            triggers = data[:, -1]
            trigger_values = sorted(set(triggers.astype(int).tolist()))
            print(
                "chunk {}/{}: samples={}, columns={}, eeg_shape={}, "
                "trigger_values={}, first_row_eeg={}".format(
                    received_chunks,
                    args.max_chunks,
                    data.shape[0],
                    data.shape[1],
                    eeg.shape,
                    trigger_values,
                    np.round(eeg[0], 3).tolist(),
                )
            )

        print("OpenBCI GUI LSL 数据接收测试完成")
    finally:
        device.stop_acq()


if __name__ == "__main__":
    main()
