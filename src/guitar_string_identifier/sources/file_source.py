"""Fontes de áudio baseadas em arquivos locais (WAV, FLAC, OGG, etc.)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from guitar_string_identifier.core.interfaces import AudioSource


class FileAudioSource(AudioSource):
    """Lê áudio de um arquivo usando soundfile e converte para mono float64."""

    def __init__(self, path: Path | str, channel: int | None = None) -> None:
        self.path = Path(path)
        self.channel = channel

    def read(self) -> tuple[np.ndarray, int]:
        data, sr = sf.read(str(self.path), dtype="float64", always_2d=True)
        if self.channel is not None and 0 <= self.channel < data.shape[1]:
            mono = data[:, self.channel]
        else:
            mono = np.mean(data, axis=1)
        return np.asarray(mono, dtype=np.float64), int(sr)
