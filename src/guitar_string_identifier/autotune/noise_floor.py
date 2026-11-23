"""Estimativa do piso de ruído (silêncio) e derivação do limiar de onset."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseFloorResult:
    """Resultado da medição de ~2 s de silêncio."""

    rms_db: float
    peak_db: float
    suggested_onset_threshold_db: float
    estimated_snr_db_at_minus12: float
    noise_spectrum_freqs: list[float]
    noise_spectrum_db: list[float]
    status: str  # "good" | "noisy" | "too_loud"
    message: str


class NoiseFloorEstimator:
    """Mede RMS, pico e espectro de ruído a partir de uma captura de silêncio."""

    def __init__(self, onset_margin_db: float = 16.0) -> None:
        self.onset_margin_db = onset_margin_db

    def estimate(self, audio: np.ndarray, sr: int) -> NoiseFloorResult:
        sig = np.asarray(audio, dtype=np.float64).reshape(-1)
        if sig.size < 256:
            return NoiseFloorResult(
                rms_db=-90.0,
                peak_db=-90.0,
                suggested_onset_threshold_db=-45.0,
                estimated_snr_db_at_minus12=78.0,
                noise_spectrum_freqs=[],
                noise_spectrum_db=[],
                status="noisy",
                message="Trecho de silêncio muito curto para estimar piso de ruído.",
            )

        sig = sig - np.mean(sig)
        rms = float(np.sqrt(np.mean(sig**2) + 1e-18))
        peak = float(np.max(np.abs(sig)) + 1e-18)
        rms_db = 20.0 * math.log10(rms)
        peak_db = 20.0 * math.log10(peak)

        # Limiar de onset: margem acima do RMS de silêncio, respeitando o pico de silêncio + 6 dB
        onset_db = max(rms_db + self.onset_margin_db, peak_db + 6.0)
        onset_db = float(np.clip(onset_db, -60.0, -15.0))

        # SNR estimado assumindo uma nota típica em -12 dBFS
        est_snr = max(0.0, -12.0 - rms_db)

        # Espectro de ruído subamostrado (Welch simplificado em janela única)
        n_fft = 1 << int(np.ceil(np.log2(min(max(4096, sig.size), 65536))))
        seg = sig[: min(sig.size, n_fft)]
        win = np.hanning(seg.size)
        spec = np.abs(np.fft.rfft(seg * win, n=n_fft)) / (np.sum(win) * 0.5 + 1e-18)
        spec_db = 20.0 * np.log10(spec + 1e-12)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        max_idx = int(np.searchsorted(freqs, min(4000.0, 0.48 * sr)))
        step = max(1, max_idx // 200)
        sub_f = [round(float(freqs[i]), 1) for i in range(0, max_idx, step)]
        sub_db = [round(float(spec_db[i]), 1) for i in range(0, max_idx, step)]

        if rms_db > -30.0:
            status = "too_loud"
            msg = (
                f"Ruído de fundo muito alto ({rms_db:.1f} dBFS). Verifique cabo, "
                "aterramento ou se alguma corda ficou vibrando."
            )
        elif rms_db > -48.0:
            status = "noisy"
            msg = (
                f"Ruído de fundo moderado ({rms_db:.1f} dBFS). "
                "Limiar de detecção ajustado automaticamente."
            )
        else:
            status = "good"
            msg = f"Piso de ruído limpo ({rms_db:.1f} dBFS, SNR estimado ~{est_snr:.0f} dB)."

        return NoiseFloorResult(
            rms_db=round(rms_db, 2),
            peak_db=round(peak_db, 2),
            suggested_onset_threshold_db=round(onset_db, 1),
            estimated_snr_db_at_minus12=round(est_snr, 1),
            noise_spectrum_freqs=sub_f,
            noise_spectrum_db=sub_db,
            status=status,
            message=msg,
        )
