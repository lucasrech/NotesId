"""Características complementares: decaimento por parcial, posição de palhetada e espectro."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import stft

from guitar_string_identifier.core.interfaces import FeatureExtractor
from guitar_string_identifier.core.types import Note, PartialInfo


class ComplementaryFeatureExtractor(FeatureExtractor):
    """Extrai taxas de decaimento por parcial, posição de palhetada, centroide e roll-off."""

    def __init__(
        self,
        transient_ms: float = 50.0,
        max_decay_partials: int = 8,
        stft_size: int = 4096,
        stft_hop: int = 512,
        rolloff_percent: float = 0.85,
    ) -> None:
        self.transient_ms = transient_ms
        self.max_decay_partials = max_decay_partials
        self.stft_size = stft_size
        self.stft_hop = stft_hop
        self.rolloff_percent = rolloff_percent

    def extract(self, note: Note, partials: list[PartialInfo] | None = None) -> dict[str, Any]:
        audio = np.asarray(note.audio, dtype=np.float64)
        sr = note.sr
        f0 = note.f0

        start = int(round((self.transient_ms / 1000.0) * sr))
        if start >= audio.size - 512:
            start = 0
        steady = audio[start:]
        steady = steady - np.mean(steady)

        # 1. Centroide e roll-off espectral (apoio de baixo peso)
        n_fft_spec = 1 << int(np.ceil(np.log2(max(2048, min(steady.size, int(0.5 * sr))))))
        seg_spec = steady[: min(steady.size, n_fft_spec)]
        win = np.hanning(seg_spec.size)
        mag = np.abs(np.fft.rfft(seg_spec * win, n=n_fft_spec))
        freqs = np.fft.rfftfreq(n_fft_spec, d=1.0 / sr)
        mag_sum = float(np.sum(mag)) + 1e-18

        spectral_centroid = float(np.sum(freqs * mag) / mag_sum)
        # Normalizado por f0 (adimensional, mais comparável entre notas)
        centroid_ratio = spectral_centroid / f0 if f0 and f0 > 0 else 0.0

        power = mag**2
        cum_power = np.cumsum(power)
        total_power = float(cum_power[-1]) + 1e-18
        rolloff_idx = int(np.searchsorted(cum_power, self.rolloff_percent * total_power))
        rolloff_idx = min(rolloff_idx, freqs.size - 1)
        spectral_rolloff = float(freqs[rolloff_idx])

        # 2. Taxa de decaimento por parcial via STFT (regressão log-linear em dB/s)
        decay_rates: dict[int, float] = {}
        decay_mean: float | None = None
        decay_ratio_high_low: float | None = None

        if f0 is not None and f0 > 0 and steady.size >= self.stft_size:
            nperseg = min(self.stft_size, steady.size)
            noverlap = max(0, nperseg - self.stft_hop)
            f_stft, t_stft, zxx = stft(
                steady,
                fs=sr,
                nperseg=nperseg,
                noverlap=noverlap,
                window="hann",
            )
            mag_stft = np.abs(zxx) + 1e-12
            db_stft = 20.0 * np.log10(mag_stft)
            df_stft = f_stft[1] - f_stft[0]

            # Usa frequências medidas dos parciais quando disponíveis
            partial_map = {p.n: p.measured_freq for p in (partials or [])}

            for n in range(1, self.max_decay_partials + 1):
                target_f = partial_map.get(n, n * f0)
                if target_f >= 0.45 * sr:
                    break
                bin_c = int(round(target_f / df_stft))
                b0 = max(0, bin_c - 2)
                b1 = min(db_stft.shape[0], bin_c + 3)
                env_db = np.max(db_stft[b0:b1, :], axis=0)

                if env_db.size < 4:
                    continue

                # Regressão linear apenas enquanto o parcial está acima de (pico - 28 dB)
                peak_db = float(env_db[0])
                valid = env_db >= (peak_db - 28.0)
                # Limita a janela a até 1,2 s para evitar ruído de cauda
                valid &= t_stft <= 1.2
                if int(np.sum(valid)) < 4:
                    continue

                t_v = t_stft[valid]
                y_v = env_db[valid]
                slope, _ = np.polyfit(t_v, y_v, 1)
                # Taxa de decaimento positiva em dB/s
                decay_rate = max(0.0, float(-slope))
                decay_rates[n] = decay_rate

                if partials:
                    for p in partials:
                        if p.n == n:
                            p.decay_rate = round(decay_rate, 3)

            if decay_rates:
                decay_mean = float(np.mean(list(decay_rates.values())))
                low_vals = [v for k, v in decay_rates.items() if k <= 3]
                high_vals = [v for k, v in decay_rates.items() if k >= 4]
                if low_vals and high_vals:
                    low_m = float(np.mean(low_vals))
                    high_m = float(np.mean(high_vals))
                    decay_ratio_high_low = high_m / (low_m + 0.5)

        # 3. Estimativa da posição de palhetada pelos nulos do pente espectral
        pluck_position_rel = self._estimate_pluck_position(partials)

        return {
            "decay_rates": decay_rates,
            "decay_mean_db_s": decay_mean,
            "decay_ratio_high_low": decay_ratio_high_low,
            "pluck_position_rel": pluck_position_rel,
            "spectral_centroid_hz": spectral_centroid,
            "spectral_centroid_ratio": centroid_ratio,
            "spectral_rolloff_hz": spectral_rolloff,
        }

    @staticmethod
    def _estimate_pluck_position(partials: list[PartialInfo] | None) -> float | None:
        """Estima fração p in [0.06, 0.50] onde |sin(n * pi * p)| melhor explica os nulos."""
        if not partials or len(partials) < 4:
            return None

        # Para uma corda ideal pinçada em p, a amplitude do parcial n é proporcional a
        # |sin(n * pi * p)| / n^2. Logo amp_n * n^2 ~ |sin(n * pi * p)|.
        ns = np.array([p.n for p in partials], dtype=np.float64)
        amps = np.array([10.0 ** (p.magnitude_db / 20.0) for p in partials], dtype=np.float64)
        compensated = amps * (ns**1.5)
        comp_norm = compensated / (np.linalg.norm(compensated) + 1e-12)

        best_p = 0.2
        best_corr = -1.0
        for p_cand in np.linspace(0.06, 0.50, 89):
            model = np.abs(np.sin(ns * math.pi * p_cand))
            norm = float(np.linalg.norm(model))
            if norm < 1e-9:
                continue
            corr = float(np.dot(comp_norm, model / norm))
            if corr > best_corr:
                best_corr = corr
                best_p = float(p_cand)
        return round(best_p, 3)
