# -*- coding: utf-8 -*-
"""Publish simulated LSL streams for online MI4C demos.

This script is a lightweight replacement for Curry/OpenBCI GUI during local
debugging. It publishes:

* CURRYStream: 8-channel EEG stream for Online_mi4c_new_lsl.py
* obci_eeg1: 8-channel EEG stream for Online_mi4c_new_OpenBCI.py
* obci_eeg2: marker stream compatible with the OpenBCI marker inlet
"""

import argparse
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock


CHANNEL_NAMES = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Publish simulated LSL EEG/marker streams for MI4C online demos."
    )
    parser.add_argument(
        "--streams",
        default="all",
        choices=["all", "curry", "openbci"],
        help="Which EEG streams to publish.",
    )
    parser.add_argument("--srate", type=float, default=256.0, help="EEG sample rate.")
    parser.add_argument("--channels", type=int, default=8, help="EEG channel count.")
    parser.add_argument("--chunk-size", type=int, default=16, help="Samples per chunk.")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run duration in seconds; 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--marker-interval",
        type=float,
        default=2.0,
        help="Seconds between simulated OpenBCI marker samples.",
    )
    return parser


def create_eeg_outlet(name, source_id, channel_count, srate):
    info = StreamInfo(
        name=name,
        type="EEG",
        channel_count=channel_count,
        nominal_srate=srate,
        channel_format="float32",
        source_id=source_id,
    )
    channels = info.desc().append_child("channels")
    for index in range(channel_count):
        label = CHANNEL_NAMES[index] if index < len(CHANNEL_NAMES) else f"Ch{index + 1}"
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", "uV")
        channel.append_child_value("type", "EEG")
    return StreamOutlet(info)


def create_marker_outlet():
    info = StreamInfo(
        name="obci_eeg2",
        type="Marker",
        channel_count=1,
        nominal_srate=0,
        channel_format="float32",
        source_id="openbcigui_marker",
    )
    return StreamOutlet(info)


def make_eeg_chunk(sample_index, chunk_size, channel_count, srate, rng):
    sample_ids = sample_index + np.arange(chunk_size, dtype=np.float32)
    t = sample_ids / float(srate)
    chunk = np.empty((chunk_size, channel_count), dtype=np.float32)
    for channel in range(channel_count):
        alpha = 12.0 * np.sin(2 * np.pi * (8.0 + channel * 0.25) * t)
        beta = 4.0 * np.sin(2 * np.pi * (18.0 + channel * 0.1) * t)
        noise = rng.normal(0.0, 2.0, size=chunk_size)
        chunk[:, channel] = alpha + beta + noise
    return chunk


def main():
    args = build_parser().parse_args()
    rng = np.random.default_rng(20260714)
    outlets = []

    if args.streams in ("all", "curry"):
        outlets.append(("CURRYStream", create_eeg_outlet(
            "CURRYStream", "simulated_curry", args.channels, args.srate)))
    if args.streams in ("all", "openbci"):
        outlets.append(("obci_eeg1", create_eeg_outlet(
            "obci_eeg1", "openbcigui", args.channels, args.srate)))
    marker_outlet = create_marker_outlet() if args.streams in ("all", "openbci") else None

    stream_names = ", ".join(name for name, _ in outlets)
    if marker_outlet is not None:
        stream_names += ", obci_eeg2"
    print(f"Publishing simulated LSL streams: {stream_names}")
    print("Press Ctrl+C to stop.")

    sample_index = 0
    marker_value = 1
    start_time = time.time()
    last_marker_time = start_time
    chunk_sleep = args.chunk_size / float(args.srate)

    try:
        while args.duration <= 0 or time.time() - start_time < args.duration:
            chunk = make_eeg_chunk(
                sample_index,
                args.chunk_size,
                args.channels,
                args.srate,
                rng,
            )
            for _, outlet in outlets:
                outlet.push_chunk(chunk.tolist())
            sample_index += args.chunk_size

            now = time.time()
            if marker_outlet is not None and now - last_marker_time >= args.marker_interval:
                marker_outlet.push_sample([float(marker_value)], local_clock())
                print(f"sent marker={marker_value}")
                marker_value = marker_value % 4 + 1
                last_marker_time = now

            time.sleep(chunk_sleep)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
