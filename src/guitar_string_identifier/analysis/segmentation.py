"""Segmentação de notas por detecção de onset (fluxo espectral e energia)."""

from __future__ import annotations

import numpy as np
from scipy.signal import stft

from guitar_string_identifier.core.interfaces import NoteSegmenter
from guitar_string_identifier.core.types import Note


class WholeFileSegmenter(NoteSegmenter):
    """Trata o arquivo inteiro como uma única nota (ideal para datasets recortados).

    Localiza o início real do ataque (acima de um limiar relativo ao pico) para
    definir `t0` com precisão antes do descarte do transiente.
    """

    def __init__(self, silence_threshold_db: float = -40.0) -> None:
        self.silence_threshold_db = silence_threshold_db

    def segment(self, audio: np.ndarray, sr: int) -> list[Note]:
        if audio.size == 0:
            return []
        mono = np.asarray(audio, dtype=np.float64)
        if mono.ndim > 1:
            mono = np.mean(mono, axis=1)

        peak = float(np.max(np.abs(mono)))
        if peak < 1e-8:
            return []

        thresh = peak * (10.0 ** (self.silence_threshold_db / 20.0))
        above = np.flatnonzero(np.abs(mono) >= thresh)
        start_idx = int(above[0]) if above.size > 0 else 0
        # Recua um pequeno trecho (5 ms) para não cortar o ataque
        margin = int(0.005 * sr)
        start_idx = max(0, start_idx - margin)

        return [Note(audio=mono[start_idx:], sr=sr, t0=start_idx / sr)]


class OnsetNoteSegmenter(NoteSegmenter):
    """Segmentador baseado em fluxo espectral e energia RMS para áudio contínuo."""

    def __init__(
        self,
        frame_size: int = 2048,
        hop_size: int = 512,
        flux_threshold_sigma: float = 2.5,
        min_note_duration: float = 0.25,
        min_rms_db: float = -45.0,
    ) -> None:
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.flux_threshold_sigma = flux_threshold_sigma
        self.min_note_duration = min_note_duration
        self.min_rms_db = min_rms_db

    def segment(self, audio: np.ndarray, sr: int) -> list[Note]:
        if audio.size < self.frame_size:
            return []
        mono = np.asarray(audio, dtype=np.float64)
        if mono.ndim > 1:
            mono = np.mean(mono, axis=1)

        peak = float(np.max(np.abs(mono)))
        if peak < 1e-8:
            return []

        _, _, zxx = stft(
            mono,
            fs=sr,
            nperseg=self.frame_size,
            noverlap=self.frame_size - self.hop_size,
            window="hann",
        )
        mag = np.abs(zxx)
        log_mag = np.log1p(10.0 * mag)

        # Fluxo espectral retificado (apenas incrementos positivos)
        diff = np.diff(log_mag, axis=1, prepend=log_mag[:, :1])
        flux = np.sum(np.maximum(0.0, diff), axis=0)

        # Energia RMS por frame
        rms = np.sqrt(np.mean(mag**2, axis=0) + 1e-18)
        rms_db = 20.0 * np.log10(rms / (np.max(rms) + 1e-18) + 1e-18)

        median_flux = float(np.median(flux))
        mad_flux = float(np.median(np.abs(flux - median_flux))) + 1e-9
        threshold = median_flux + self.flux_threshold_sigma * 1.4826 * mad_flux

        min_frames = max(1, int(self.min_note_duration * sr / self.hop_size))
        onsets: list[int] = []
        n_frames = flux.size
        for i in range(1, n_frames - 1):
            if (
                flux[i] > threshold
                and flux[i] >= flux[i - 1]
                and flux[i] >= flux[i + 1]
                and rms_db[i] > self.min_rms_db
                and (not onsets or (i - onsets[-1]) >= min_frames)
            ):
                onsets.append(i)

        if not onsets:
            # Fallback para nota única se houver energia suficiente
            return WholeFileSegmenter().segment(mono, sr)

        notes: list[Note] = []
        for idx, onset_frame in enumerate(onsets):
            start_sample = max(0, onset_frame * self.hop_size)
            end_sample = (
                onsets[idx + 1] * self.hop_size
                if idx + 1 < len(onsets)
                else mono.size
            )
            seg = mono[start_sample:end_sample]
            if seg.size >= int(self.min_note_duration * sr):
                notes.append(Note(audio=seg, sr=sr, t0=start_sample / sr))
        return notes
