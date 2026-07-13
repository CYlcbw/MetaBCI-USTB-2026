# -*- coding: utf-8 -*-
"""USTB2026 four-class motor imagery datasets."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import mne
import numpy as np
from mne.channels import make_standard_montage
from mne.io import Raw, RawArray

from .base import BaseDataset
from ..utils.channels import upper_ch_names
from ..utils.download import mne_data_path


class USTB2026MI4C(BaseDataset):
    """USTB2026MI4C processed base8 dataset.

    Local data are expected as compressed ``.npz`` files with ``X`` as
    trials x channels x samples and ``y`` as zero-based four-class MI labels.
    """

    _EVENTS = {
        "left_hand": (1, (0, 4)),
        "right_hand": (2, (0, 4)),
        "feet": (3, (0, 4)),
        "both_hands": (4, (0, 4)),
    }

    _CHANNELS = [
        "FC3",
        "FCZ",
        "FC4",
        "C3",
        "CZ",
        "C4",
        "CP3",
        "CP4",
    ]

    _DATASET_DIR = (
        Path(__file__).resolve().parents[3]
        / "demos"
        / "brainflow_demos"
        / "data"
        / "ustb2025mi4c-main"
        / "Dataset"
    )

    _REMOTE_BASE_URL = (
        "https://raw.githubusercontent.com/CYlcbw/MetaBCI-USTB-2026/master/"
        "demos/brainflow_demos/data/ustb2025mi4c-main/Dataset"
    )

    _SUBJECT_FILES = {
        1: [("base8_s01_s04_final", "S01_base8_all_runs.npz")],
        2: [("base8_s01_s04_final", "S02_base8_all_runs.npz")],
        3: [("base8_s01_s04_final", "S03_base8_all_runs.npz")],
        4: [("base8_s01_s04_final", "S04_base8_all_runs.npz")],
        5: [("base8_s01_s05_final", "S05_base8_all_runs.npz")],
    }

    _REMOTE_SUBJECT_FILES = {
        1: ("base8_s01_s04_final", "S01_base8_all_runs.npz"),
        2: ("base8_s01_s04_final", "S02_base8_all_runs.npz"),
        3: ("base8_s01_s04_final", "S03_base8_all_runs.npz"),
        4: ("base8_s01_s04_final", "S04_base8_all_runs.npz"),
        5: ("base8_s01_s05_final", "S05_base8_all_runs.npz"),
    }

    def __init__(self):
        super().__init__(
            dataset_code="ustb2026mi4c_base8",
            subjects=list(range(1, 6)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=256,
            paradigm="imagery",
        )

    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[Dict[str, str]] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        subject = int(subject)
        if subject not in self.subjects:
            raise ValueError("Invalid subject id")

        data_root = Path(path).expanduser() if path is not None else self._DATASET_DIR
        local_path = self._find_local_subject_file(subject, data_root)
        if local_path is not None and not force_update:
            return [[local_path]]

        folder, file_name = self._REMOTE_SUBJECT_FILES[subject]
        url = "{}/{}/{}".format(self._REMOTE_BASE_URL, folder, file_name)
        file_dest = mne_data_path(
            url,
            self.dataset_code,
            path=path,
            proxies=proxies,
            force_update=force_update,
            update_path=update_path,
            verbose=verbose,
        )
        return [[Path(file_dest)]]

    def _find_local_subject_file(self, subject: int, data_root: Path) -> Optional[Path]:
        candidates = []
        for folder, file_name in self._SUBJECT_FILES[subject]:
            candidates.append(data_root / folder / file_name)
            candidates.append(data_root / file_name)
        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def load_subject_arrays(self, subject: Union[str, int]):
        paths = self.data_path(subject)
        npz_path = Path(paths[0][0])

        with np.load(npz_path, allow_pickle=True) as npz:
            X = np.asarray(npz["X"], dtype=np.float32)
            y = np.asarray(npz["y"], dtype=np.int64)
            ch_names = [str(ch) for ch in npz.get("ch_names", self._CHANNELS)]
            run_ids = np.asarray(
                npz["run_ids"] if "run_ids" in npz.files else np.ones(len(y)),
                dtype=np.int64,
            )

        if X.ndim != 3:
            raise ValueError("X must be trials x channels x samples, got {}".format(X.shape))
        if X.shape[1] != len(self._CHANNELS):
            raise ValueError(
                "USTB2026MI4C base8 expects {:d} channels, got {:d}".format(
                    len(self._CHANNELS), X.shape[1]
                )
            )
        if len(y) != X.shape[0] or len(run_ids) != X.shape[0]:
            raise ValueError("X, y, and run_ids must have the same trial count")
        if y.min() == 1:
            y = y - 1
        if not set(np.unique(y).astype(int).tolist()).issubset({0, 1, 2, 3}):
            raise ValueError("y must contain MI4C labels 0..3")

        ch_names = [ch.upper() for ch in ch_names]
        if ch_names != self._CHANNELS:
            raise ValueError(
                "Unexpected channel order: {}. Expected: {}".format(
                    ch_names, self._CHANNELS
                )
            )

        return {
            "X": X,
            "y": y,
            "run_ids": run_ids,
            "ch_names": ch_names,
            "path": str(npz_path),
        }

    def _get_single_subject_data(
        self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        arrays = self.load_subject_arrays(subject)
        montage = make_standard_montage("standard_1005")
        montage.rename_channels({ch_name: ch_name.upper() for ch_name in montage.ch_names})

        runs = {}
        run_ids = arrays["run_ids"]
        for run_id in sorted(np.unique(run_ids).astype(int).tolist()):
            trial_index = np.flatnonzero(run_ids == run_id)
            raw = self._trials_to_raw(
                arrays["X"][trial_index],
                arrays["y"][trial_index],
                float(self.srate),
                montage,
            )
            runs["run_{:d}".format(run_id - 1)] = raw

        return {"session_0": runs}

    def _trials_to_raw(self, X, y, srate, montage):
        n_trials, n_channels, n_samples = X.shape
        eeg = np.transpose(X, (1, 0, 2)).reshape(n_channels, n_trials * n_samples)

        stim = np.zeros((1, n_trials * n_samples), dtype=np.float64)
        stim[0, np.arange(n_trials) * n_samples] = y.astype(np.int64) + 1

        ch_names = self._CHANNELS + ["STI 014"]
        ch_types = ["eeg"] * len(self._CHANNELS) + ["stim"]
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=srate)

        raw = RawArray(data=np.vstack([eeg, stim]), info=info, verbose=False)
        raw = upper_ch_names(raw)
        raw.set_montage(montage)
        return raw


class USTB2026MI4CBase8(USTB2026MI4C):
    """Name for the processed base8 MI4C dataset."""


USTB2025MI4C = USTB2026MI4C
USTB2025MI4CBase8 = USTB2026MI4CBase8
