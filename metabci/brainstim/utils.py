import serial
from psychopy import parallel
import numpy as np
import json
import queue
import socket
import struct
import threading
import time

from pylsl import StreamInfo, StreamOutlet, local_clock


class NeuroScanPort:
    """
    Send tag communication Using parallel port or serial port.

    author: Lichao Xu

    Created on: 2020-07-30

    update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        port_addr: ndarray
            The port address, hexadecimal or decimal.
        use_serial: bool
            If False, send the tags using parallel port, otherwise using serial port.
        baudrate: int
            The serial port baud rate.

    Attributes
    ----------
        port_addr: ndarray
            The port address, hexadecimal or decimal.
        use_serial: bool
            If False, send the tags using parallel port, otherwise using serial port.
        baudrate: int
            The serial port baud rate.
        port:
            Send tag communication Using parallel port or serial port.

    Tip
    ----
    .. code-block:: python
       :caption: An example of using port to send tags

        from brainstim.utils import NeuroScanPort
        port = NeuroScanPort(port_addr, use_serial=False) if port_addr else None
        VSObject.win.callOnFlip(port.setData, 1)
        port.setData(0)

    """

    def __init__(self, port_addr, use_serial=False, baudrate=115200):
        self.use_serial = use_serial
        if use_serial:
            self.port = serial.Serial(port=port_addr, baudrate=baudrate)
            self.port.write([0])
        else:
            self.port = parallel.ParallelPort(address=port_addr)

    def setData(self, label):
        """Send event labels

        Parameters
        ----------
            label:
                The label sent.

        """
        if self.use_serial:
            self.port.write([int(label)])
        else:
            self.port.setData(int(label))


class NeuraclePort:
    """
    Send trigger to Neuracle device.The Neuracle device uses serial
    port for writing trigger, so it does not need to write a 0 trigger
    before a int trigger. This class is writen under the Trigger box instruction.

    author: Jie Mei

    Created on: 2022-12-05

    update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        port_addr: ndarray
            The port address, hexadecimal or decimal.
        baudrate: int
            The serial port baud rate.

    """

    def __init__(self, port_addr, baudrate=115200) -> None:
        # The only choice for neuracle is using serial for writting trigger
        self.port = serial.Serial(port=port_addr, baudrate=baudrate)

    def setData(self, label):
        # Neuracle doesn't need 0 trigger before a int trigger.
        if str(label) != '0':
            head_string = '01E10100'
            hex_label = str(hex(label))
            if len(hex_label) == 3:
                hex_value = hex_label[2]
                hex_label = '0'+hex_value.upper()
            else:
                hex_label = hex_label[2:].upper()
            send_string = head_string+hex_label
            send_string_byte = [int(send_string[i:i+2], 16)
                                for i in range(0, len(send_string), 2)]
            self.port.write(send_string_byte)


class LsLPort:
    """
    Creating a lab streaming layer marker, which could align with the
    stream which retriving stream from devices.

    """

    def __init__(self) -> None:
        self.info = StreamInfo(
            name='LSLMarkerStream',
            type='Marker',
            channel_count=1,
            nominal_srate=0,
            channel_format='cf_int16')
        self.outlet = StreamOutlet(self.info)

    def setData(self, label):
        # We don't need 0 trigger before a int trigger
        if str(label) != '0':
            self.outlet.push_sample(str(label))


class OpenBCIPort:
    """Send OpenBCI markers to GUI, local LSL, or both.

    Parameters
    ----------
    target:
        ``"gui"`` sends a network-order float32 UDP marker to OpenBCI GUI.
        ``"lsl"`` publishes a local one-channel LSL marker stream for MetaBCI.
        ``"both"`` does both with one ``setData`` call.
    """

    NAME = "OpenBCI Port"
    TARGET_GUI = "gui"
    TARGET_LSL = "lsl"
    TARGET_BOTH = "both"

    def __init__(
        self,
        port_addr=None,
        target="both",
        stream_name="obci_eeg2",
        stream_type="Marker",
        source_id="metabci_openbci_marker",
        send_zero=False,
    ):
        self.target = self._normalize_target(target)
        self.address = self._normalize_address(port_addr)
        self.stream_name = stream_name
        self.stream_type = stream_type
        self.source_id = source_id
        self.send_zero = send_zero

        self.sock = None
        if self.target in (self.TARGET_GUI, self.TARGET_BOTH):
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.info = None
        self.outlet = None
        if self.target in (self.TARGET_LSL, self.TARGET_BOTH):
            self.info = StreamInfo(
                name=stream_name,
                type=stream_type,
                channel_count=1,
                nominal_srate=0,
                channel_format="float32",
                source_id=source_id,
            )
            self.outlet = StreamOutlet(self.info)

    @staticmethod
    def _normalize_address(port_addr):
        if port_addr is None:
            return ("127.0.0.1", 12350)
        if isinstance(port_addr, tuple):
            return (port_addr[0], int(port_addr[1]))
        if isinstance(port_addr, int):
            return ("127.0.0.1", int(port_addr))
        if isinstance(port_addr, str):
            if ":" in port_addr:
                host, port = port_addr.rsplit(":", 1)
                return (host, int(port))
            if port_addr.isdigit():
                return ("127.0.0.1", int(port_addr))
            return (port_addr, 12350)
        raise ValueError("port_addr must be a tuple, int, str, or None")

    @classmethod
    def _normalize_target(cls, target):
        if target is None:
            return cls.TARGET_BOTH
        normalized = str(target).strip().lower().replace("-", "_")
        aliases = {
            "gui": cls.TARGET_GUI,
            "openbci_gui": cls.TARGET_GUI,
            "udp": cls.TARGET_GUI,
            "lsl": cls.TARGET_LSL,
            "local_lsl": cls.TARGET_LSL,
            "metabci": cls.TARGET_LSL,
            "metabci_lsl": cls.TARGET_LSL,
            "both": cls.TARGET_BOTH,
            "all": cls.TARGET_BOTH,
            "gui_lsl": cls.TARGET_BOTH,
            "openbci_gui_lsl": cls.TARGET_BOTH,
        }
        if normalized not in aliases:
            raise ValueError("Unsupported OpenBCIPort target: {}".format(target))
        return aliases[normalized]

    def setData(self, label):
        label = float(label)
        if label == 0 and not self.send_zero:
            return
        if self.sock is not None:
            self.sock.sendto(struct.pack("!f", label), self.address)
        if self.outlet is not None:
            self.outlet.push_sample([label], local_clock())

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        self.outlet = None


def _check_array_like(value, length=None):
    """
    Check array dimensions.

    -author: Lichao Xu

    -Created on: 2020-07-30

    -update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        value: ndarray,
            The array to check.
        length: int,
            The array dimension.

    """

    flag = isinstance(value, (list, tuple, np.ndarray))
    return flag and (len(value) == length if length is not None else True)


def _clean_dict(old_dict, includes=[]):
    """
    Clear dictionary.

    -author: Lichao Xu

    -Created on: 2020-07-30

    -update log:
        2023-12-09 by Lixia Lin <1582063370@qq.com> Add code annotation

    Parameters
    ----------
        old_dict: dict,
            The dict to clear.
        includes: list,
            Key-value indexes that need to be preserved.

    """

    names = list(old_dict.keys())
    for name in names:
        if name not in includes:
            old_dict[name] = None
            del old_dict[name]
    return old_dict
