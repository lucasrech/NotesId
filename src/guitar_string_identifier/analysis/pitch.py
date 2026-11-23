"""Estimativa de frequência fundamental (f0) via algoritmo YIN com precisão sub-cent."""

from __future__ import annotations

import numpy as np

from guitar_string_identifier.core.interfaces import PitchEstimator
from guitar_string_identifier.core.types import Note


def _yin_frame(
    frame: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    threshold: float,
) -> tuple[float | None, float]:
    """Executa YIN em uma única janela, retornando (f0_hz, aperiodicity)."""
    n = frame.size
    tau_min = max(2, int(np.floor(sr / fmax)))
    tau_max = min(n // 2, int(np.ceil(sr / fmin)))
    if tau_max <= tau_min + 2:
        return None, 1.0

    # Função diferença via FFT (O(N log N)):
    # d(tau) = sum_{j=0}^{W-1} (x[j] - x[j+tau])^2
    #        = r0 + r_tau - 2 * acf(tau)
    w = n - tau_max
    x = frame - np.mean(frame)
    if np.sum(x * x) < 1e-12:
        return None, 1.0

    n_fft = 1 << int(np.ceil(np.log2(2 * n)))
    fft_x = np.fft.rfft(x[:w], n=n_fft)
    fft_full = np.fft.rfft(x, n=n_fft)
    # Correlação cruzada sum_{j=0}^{w-1} x[j] * x[j + tau]
    corr = np.fft.irfft(np.conj(fft_x) * fft_full, n=n_fft)[: tau_max + 1]

    # Somas cumulativas de energia
    sq = x * x
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    e0 = csum[w]
    e_tau = csum[w : w + tau_max + 1] - csum[: tau_max + 1]

    d = e0 + e_tau - 2.0 * corr
    d = np.maximum(d, 0.0)

    # Função diferença normalizada pela média cumulativa (CMNDF)
    cmndf = np.ones(tau_max + 1, dtype=np.float64)
    running_sum = np.cumsum(d[1:])
    idx = np.arange(1, tau_max + 1, dtype=np.float64)
    nonzero = running_sum > 1e-18
    cmndf[1:][nonzero] = d[1:][nonzero] * idx[nonzero] / running_sum[nonzero]

    # Busca o primeiro mínimo local abaixo do limiar
    tau_est: int | None = None
    for tau in range(tau_min, tau_max):
        if cmndf[tau] < threshold:
            while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                tau += 1
            tau_est = tau
            break

    # Fallback: mínimo global na faixa [tau_min, tau_max)
    if tau_est is None:
        rel_idx = int(np.argmin(cmndf[tau_min:tau_max]))
        tau_est = tau_min + rel_idx

    aperiodicity = float(cmndf[tau_est])

    # Interpolação parabólica para precisão sub-amostra (sub-cent)
    if 1 <= tau_est < tau_max:
        y_prev = cmndf[tau_est - 1]
        y_curr = cmndf[tau_est]
        y_next = cmndf[tau_est + 1]
        denom = y_prev - 2.0 * y_curr + y_next
        if abs(denom) > 1e-12:
            delta = 0.5 * (y_prev - y_next) / denom
            delta = float(np.clip(delta, -0.5, 0.5))
            tau_refined = tau_est + delta
        else:
            tau_refined = float(tau_est)
    else:
        tau_refined = float(tau_est)

    if tau_refined <= 0:
        return None, 1.0
    return sr / tau_refined, aperiodicity


class YinPitchEstimator(PitchEstimator):
    """Estimador de f0 usando YIN multi-frame no regime estacionário."""

    def __init__(
        self,
        fmin: float = 65.0,
        fmax: float = 1400.0,
        threshold: float = 0.12,
        transient_ms: float = 50.0,
        frame_ms: float = 80.0,
        n_frames: int = 5,
    ) -> None:
        self.fmin = fmin
        self.fmax = fmax
        self.threshold = threshold
        self.transient_ms = transient_ms
        self.frame_ms = frame_ms
        self.n_frames = n_frames

    def estimate(self, note: Note) -> float | None:
        audio = np.asarray(note.audio, dtype=np.float64)
        sr = note.sr
        if audio.size < 256:
            return None

        start = int(round((self.transient_ms / 1000.0) * sr))
        if start >= audio.size - 256:
            start = 0

        steady = audio[start:]
        frame_len = max(int(round((self.frame_ms / 1000.0) * sr)), int(2.5 * sr / self.fmin))
        if steady.size < frame_len:
            frame_len = steady.size

        # Seleciona múltiplas janelas distribuídas nos primeiros ~0.5 s de regime estacionário
        max_span = min(steady.size - frame_len, int(0.5 * sr))
        if max_span <= 0 or self.n_frames <= 1:
            starts = [0]
        else:
            starts = np.linspace(0, max_span, self.n_frames, dtype=int).tolist()

        all_estimates: list[tuple[float, float]] = []
        for s in starts:
            frame = steady[s : s + frame_len]
            f0, aper = _yin_frame(frame, sr, self.fmin, self.fmax, self.threshold)
            if f0 is not None:
                all_estimates.append((f0, aper))

        estimates = [pair for pair in all_estimates if pair[1] < 0.35]
        if not estimates:
            relaxed = [pair for pair in all_estimates if pair[1] < 0.75]
            if not relaxed:
                return None
            relaxed.sort(key=lambda pair: pair[1])
            return relaxed[0][0]

        # Mediana das estimativas com melhor periodicidade (evita outliers de oitava)
        estimates.sort(key=lambda pair: pair[1])
        best = estimates[: max(1, (len(estimates) + 1) // 2)]
        freqs = np.array([pair[0] for pair in best], dtype=np.float64)
        return float(np.median(freqs))
