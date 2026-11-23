"""Fontes de áudio ao vivo: buffer WebSocket (navegador) e sounddevice opcional."""

from __future__ import annotations

import numpy as np

from guitar_string_identifier.core.interfaces import AudioSource


class BufferAudioSource(AudioSource):
    """Fonte de áudio em memória alimentada por blocos PCM float32 (ex: Web Audio API)."""

    def __init__(self, samples: np.ndarray, sr: int) -> None:
        self._samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        self._sr = int(sr)

    def read(self) -> tuple[np.ndarray, int]:
        return self._samples.copy(), self._sr


class SoundDeviceAudioSource(AudioSource):
    """Fonte opcional de captura direta no servidor via biblioteca `sounddevice`.

    Nota: Em ambientes WSL2 (Windows Subsystem for Linux), o acesso direto às
    interfaces de áudio do Windows via ALSA/PortAudio não é confiável. Para uso
    em WSL2, utilize a captura via navegador (Web Audio API + WebSocket).
    """

    def __init__(
        self,
        device: int | str | None = None,
        sr: int = 48000,
        duration: float = 1.5,
        channel: int = 0,
    ) -> None:
        self.device = device
        self.sr = sr
        self.duration = duration
        self.channel = channel

    def read(self) -> tuple[np.ndarray, int]:
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "A biblioteca opcional 'sounddevice' não está instalada. "
                "Em WSL2, use a captura ao vivo pelo navegador (Web Audio API)."
            ) from exc

        n_samples = int(round(self.duration * self.sr))
        channels = self.channel + 1
        recording = sd.rec(
            n_samples,
            samplerate=self.sr,
            channels=channels,
            dtype="float32",
            device=self.device,
        )
        sd.wait()
        mono = np.asarray(recording[:, self.channel], dtype=np.float64)
        return mono, self.sr
