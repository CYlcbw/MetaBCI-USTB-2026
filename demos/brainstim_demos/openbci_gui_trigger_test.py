# -*- coding: utf-8 -*-
"""Send test markers to OpenBCI GUI Marker Widget over UDP.

OpenBCI GUI's Marker Widget expects each UDP payload to be one network-order
float32. This script intentionally uses only Python standard-library modules so
it can test the GUI marker path without OpenBCI, BrainFlow, or PsychoPy.
"""

import argparse
import socket
import struct
import time


def parse_labels(value):
    labels = []
    for item in value.split(","):
        item = item.strip()
        if item:
            labels.append(float(item))
    if not labels:
        raise argparse.ArgumentTypeError("at least one label is required")
    return labels


def build_parser():
    parser = argparse.ArgumentParser(
        description="Send UDP trigger markers to OpenBCI GUI Marker Widget."
    )
    parser.add_argument("--host", default="127.0.0.1", help="OpenBCI GUI marker host")
    parser.add_argument("--port", type=int, default=12350, help="OpenBCI GUI marker UDP port")
    parser.add_argument(
        "--labels",
        type=parse_labels,
        default=parse_labels("1,2,3,4"),
        help="Comma-separated marker labels, e.g. 1,2,3,4",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between markers")
    parser.add_argument("--repeat", type=int, default=0, help="Repeat count; 0 means forever")
    return parser


def main():
    args = build_parser().parse_args()
    address = (args.host, args.port)
    sent = 0

    print(
        "Sending OpenBCI GUI markers to "
        f"{args.host}:{args.port}; labels={args.labels}; interval={args.interval}s"
    )
    print("Press Ctrl+C to stop.")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            while args.repeat <= 0 or sent < args.repeat * len(args.labels):
                label = args.labels[sent % len(args.labels)]
                sock.sendto(struct.pack("!f", float(label)), address)
                sent += 1
                print(f"sent marker={label:g} count={sent}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()
