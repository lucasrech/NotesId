"""Extração de parciais e ajuste do coeficiente de inharmonicidade B.

Modelo físico de corda rígida:
    f_n = n * f0 * sqrt(1 + B * n^2)

Linearização exata:
    (f_n / n)^2 = f0^2 + (f0^2 * B) * n^2
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from guitar_string_identifier.core.interfaces import FeatureExtractor
from guitar_string_identifier.core.music import freq_to_cents
from guitar_string_identifier.core.types import Note, PartialInfo


def _parabolic_peak_freq(
    mag_db: np.ndarray,
    peak_idx: int,
    df: float,
) -> tuple[float, float]:
    """Refina posição e magnitude de um pico espectral por interpolação parabólica."""
    if peak_idx <= 0 or peak_idx >= mag_db.size - 1:
        return peak_idx * df, float(mag_db[peak_idx])

    y_prev = float(mag_db[peak_idx - 1])
    y_curr = float(mag_db[peak_idx])
    y_next = float(mag_db[peak_idx + 1])
    denom = y_prev - 2.0 * y_curr + y_next
    if abs(denom) < 1e-12:
        return peak_idx * df, y_curr

    delta = 0.5 * (y_prev - y_next) / denom
    delta = float(np.clip(delta, -0.5, 0.5))
    freq = (peak_idx + delta) * df
    mag = y_curr - 0.25 * (y_prev - y_next) * delta
    return freq, float(mag)


def _fit_inharmonicity_irls(
    ns: np.ndarray,
    freqs: np.ndarray,
    weights: np.ndarray,
    n_iter: int = 8,
) -> tuple[float, float, float, float, float]:
    """Ajusta f_n = n * f0 * sqrt(1 + B * n^2) por mínimos quadrados robustos (IRLS).

    Retorna (f0_fit, b_fit, b_std, residual_cents, r2).
    """
    x = (ns**2).astype(np.float64)
    y = (freqs / ns) ** 2

    w = weights.astype(np.float64).copy()
    w = w / (np.sum(w) + 1e-18)

    c0 = float(y[0])
    c1 = 0.0

    for _ in range(n_iter):
        # Design matrix A = [1, x]
        sw = np.sqrt(w)
        a_mat = np.column_stack([sw, sw * x])
        b_vec = sw * y
        params, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        c0, c1 = float(params[0]), float(params[1])

        # Resíduos relativos em frequência (cents aproximado)
        y_pred = c0 + c1 * x
        if np.any(y_pred <= 0):
            break
        f_pred = ns * np.sqrt(np.maximum(y_pred, 1e-12))
        res_cents = 1200.0 * np.log2(freqs / f_pred)

        # Pesos Huber (delta = 4 cents)
        scale = max(2.0, 1.4826 * float(np.median(np.abs(res_cents))))
        delta = 1.5 * scale
        abs_r = np.abs(res_cents)
        huber = np.where(abs_r <= delta, 1.0, delta / (abs_r + 1e-12))
        w = weights * huber
        w = w / (np.sum(w) + 1e-18)

    if c0 <= 0:
        return float(freqs[0] / ns[0]), 0.0, float("inf"), float("inf"), 0.0

    f0_fit = math.sqrt(c0)
    b_fit = c1 / c0

    # Se o ajuste der ligeiramente negativo por ruído, registra valor real mas limita para predição
    b_clamped = max(0.0, b_fit)
    f_pred = ns * f0_fit * np.sqrt(1.0 + b_clamped * (ns**2))
    res_cents_arr = 1200.0 * np.log2(freqs / f_pred)
    residual_cents = float(np.sqrt(np.average(res_cents_arr**2, weights=w)))

    # R^2 ponderado no espaço linearizado (f_n / n)^2 vs n^2
    y_pred_lin = c0 + c1 * x
    y_mean = float(np.average(y, weights=w))
    ss_res = float(np.sum(w * (y - y_pred_lin) ** 2))
    ss_tot = float(np.sum(w * (y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 0.0

    # Incerteza estimada de B (desvio padrão da inclinação / c0)
    x_mean = float(np.average(x, weights=w))
    sxx = float(np.sum(w * (x - x_mean) ** 2))
    if sxx > 1e-18 and ns.size > 2:
        # Variância residual escalada pelo número efetivo de amostras
        n_eff = (np.sum(w) ** 2) / (np.sum(w**2) + 1e-18)
        sigma2 = ss_res / max(1.0, n_eff - 2.0) * n_eff
        var_c1 = sigma2 / (sxx * n_eff) if n_eff > 0 else float("inf")
        b_std = math.sqrt(max(0.0, var_c1)) / c0
    else:
        b_std = float("inf")

    return f0_fit, b_fit, b_std, residual_cents, r2


class InharmonicityExtractor(FeatureExtractor):
    """Extrai parciais via FFT longa com zero-padding e ajusta o coeficiente B."""

    def __init__(
        self,
        window_sec: float = 0.6,
        transient_ms: float = 50.0,
        max_partials: int = 12,
        peak_tolerance_cents: float = 40.0,
        min_peak_snr_db: float = 12.0,
        min_rel_mag_db: float = -55.0,
        zero_pad_factor: int = 4,
        hum_notch_freqs_hz: list[float] | None = None,
        hum_notch_half_width_hz: float = 3.5,
        iterative_narrowing: bool = False,
    ) -> None:
        self.window_sec = window_sec
        self.transient_ms = transient_ms
        self.max_partials = max_partials
        self.peak_tolerance_cents = peak_tolerance_cents
        self.min_peak_snr_db = min_peak_snr_db
        self.min_rel_mag_db = min_rel_mag_db
        self.zero_pad_factor = zero_pad_factor
        self.hum_notch_freqs_hz = list(hum_notch_freqs_hz or [])
        self.hum_notch_half_width_hz = hum_notch_half_width_hz
        self.iterative_narrowing = iterative_narrowing

    def _is_in_hum_band(self, freq_hz: float) -> bool:
        """Retorna True se a frequência cair em uma faixa excluída de ronco (50/60 Hz)."""
        return any(
            abs(freq_hz - hum_f) <= self.hum_notch_half_width_hz
            for hum_f in self.hum_notch_freqs_hz
        )

    def _search_partials(
        self,
        mag_db_rel: np.ndarray,
        df: float,
        nyquist: float,
        f0_init: float,
        max_partials: int,
        base_tol_cents: float,
        f0_seed: float,
        b_seed: float,
    ) -> list[PartialInfo]:
        """Busca progressiva dos parciais n = 1..max_partials excluindo faixas de ronco."""
        partials: list[PartialInfo] = []
        f0_running = f0_seed
        b_running = b_seed

        for n in range(1, max_partials + 1):
            ideal_freq = n * f0_init
            pred_freq = n * f0_running * math.sqrt(1.0 + max(0.0, b_running) * (n**2))
            if pred_freq >= nyquist:
                break

            tol_cents = base_tol_cents
            if n >= 4 and len(partials) < 4 and b_seed == 0.0:
                tol_cents = max(tol_cents, 80.0)
            f_low = pred_freq * (2.0 ** (-tol_cents / 1200.0))
            f_high = pred_freq * (2.0 ** (tol_cents / 1200.0))

            idx_low = max(1, int(math.floor(f_low / df)))
            idx_high = min(mag_db_rel.size - 2, int(math.ceil(f_high / df)))
            if idx_high <= idx_low:
                continue

            region = mag_db_rel[idx_low : idx_high + 1]
            local_max_rel = int(np.argmax(region))
            peak_idx = idx_low + local_max_rel
            peak_val = float(mag_db_rel[peak_idx])

            if (
                mag_db_rel[peak_idx] < mag_db_rel[peak_idx - 1]
                or mag_db_rel[peak_idx] < mag_db_rel[peak_idx + 1]
            ):
                continue

            bg_low = max(1, int(math.floor((pred_freq - 0.35 * f0_init) / df)))
            bg_high = min(mag_db_rel.size - 1, int(math.ceil((pred_freq + 0.35 * f0_init) / df)))
            noise_floor = float(np.median(mag_db_rel[bg_low:bg_high]))
            snr_db = peak_val - noise_floor

            if peak_val < self.min_rel_mag_db or snr_db < self.min_peak_snr_db:
                continue

            meas_freq, meas_db = _parabolic_peak_freq(mag_db_rel, peak_idx, df)
            # Exclui picos contaminados por ronco da rede elétrica (50/60 Hz e harmônicos)
            if self._is_in_hum_band(meas_freq):
                continue

            dev_cents = freq_to_cents(meas_freq, ideal_freq)
            partials.append(
                PartialInfo(
                    n=n,
                    ideal_freq=ideal_freq,
                    measured_freq=meas_freq,
                    magnitude_db=meas_db,
                    deviation_cents=dev_cents,
                )
            )

            if len(partials) >= 3:
                ns_tmp = np.array([p.n for p in partials], dtype=np.float64)
                fs_tmp = np.array([p.measured_freq for p in partials], dtype=np.float64)
                w_tmp = np.array(
                    [10.0 ** (p.magnitude_db / 20.0) for p in partials],
                    dtype=np.float64,
                )
                f0_tmp, b_tmp, _, _, _ = _fit_inharmonicity_irls(ns_tmp, fs_tmp, w_tmp, n_iter=3)
                if 0.0 <= b_tmp <= 0.05 and abs(freq_to_cents(f0_tmp, f0_init)) < 35.0:
                    f0_running = f0_tmp
                    b_running = b_tmp

        return partials

    def extract(
        self,
        note: Note,
        *,
        window_sec: float | None = None,
        transient_ms: float | None = None,
        max_partials: int | None = None,
        peak_tolerance_cents: float | None = None,
        iterative_narrowing: bool | None = None,
    ) -> dict[str, Any]:
        if note.f0 is None or note.f0 <= 0:
            return {
                "B": None,
                "B_raw": None,
                "B_std": None,
                "B_residual_cents": float("inf"),
                "B_r2": 0.0,
                "f0_refined": note.f0,
                "n_partials_used": 0,
                "partials": [],
                "spectrum_freqs": [],
                "spectrum_db": [],
            }

        eff_window = window_sec if window_sec is not None else self.window_sec
        eff_transient = transient_ms if transient_ms is not None else self.transient_ms
        eff_max_partials = max_partials if max_partials is not None else self.max_partials
        eff_tol = (
            peak_tolerance_cents
            if peak_tolerance_cents is not None
            else self.peak_tolerance_cents
        )
        eff_iter = (
            iterative_narrowing
            if iterative_narrowing is not None
            else self.iterative_narrowing
        )

        audio = np.asarray(note.audio, dtype=np.float64)
        sr = note.sr
        f0_init = float(note.f0)

        # Descarta o transiente inicial
        start = int(round((eff_transient / 1000.0) * sr))
        if start >= audio.size - 1024:
            start = 0
        steady = audio[start:]

        # Janela longa
        win_len = min(steady.size, max(int(round(eff_window * sr)), int(0.25 * sr)))
        if win_len > steady.size:
            win_len = steady.size
        seg = steady[:win_len]
        seg = seg - np.mean(seg)

        # Janela Blackman-Harris (lóbulos laterais < -92 dB, excelente para parciais fracos)
        n_idx = np.arange(win_len, dtype=np.float64)
        window = (
            0.35875
            - 0.48829 * np.cos(2.0 * np.pi * n_idx / max(1, win_len - 1))
            + 0.14128 * np.cos(4.0 * np.pi * n_idx / max(1, win_len - 1))
            - 0.01168 * np.cos(6.0 * np.pi * n_idx / max(1, win_len - 1))
        )
        windowed = seg * window

        # Zero-padding para alta resolução espectral
        n_fft = 1 << int(np.ceil(np.log2(max(4096, win_len * self.zero_pad_factor))))
        spec = np.fft.rfft(windowed, n=n_fft)
        mag = np.abs(spec) + 1e-18
        mag_db = 20.0 * np.log10(mag)
        peak_global_db = float(np.max(mag_db))
        mag_db_rel = mag_db - peak_global_db

        df = sr / n_fft
        nyquist = 0.48 * sr

        # Passo 1: busca progressiva inicial
        partials = self._search_partials(
            mag_db_rel, df, nyquist, f0_init, eff_max_partials, eff_tol, f0_init, 0.0
        )

        # Passos iterativos opcionais (modo auto): estreita a janela de busca conforme B converge
        if eff_iter and len(partials) >= 3:
            for narrow_tol in (min(eff_tol, 24.0), min(eff_tol, 15.0)):
                ns_tmp = np.array([p.n for p in partials], dtype=np.float64)
                fs_tmp = np.array([p.measured_freq for p in partials], dtype=np.float64)
                w_tmp = np.array(
                    [max(0.05, 10.0 ** (p.magnitude_db / 40.0)) for p in partials],
                    dtype=np.float64,
                )
                f0_seed, b_seed, _, _, _ = _fit_inharmonicity_irls(ns_tmp, fs_tmp, w_tmp)
                if not (0.0 <= b_seed <= 0.05):
                    break
                refined = self._search_partials(
                    mag_db_rel,
                    df,
                    nyquist,
                    f0_init,
                    eff_max_partials,
                    narrow_tol,
                    f0_seed,
                    b_seed,
                )
                if len(refined) >= 3:
                    partials = refined

        # Ajuste final de B
        if len(partials) >= 3:
            ns_arr = np.array([p.n for p in partials], dtype=np.float64)
            fs_arr = np.array([p.measured_freq for p in partials], dtype=np.float64)
            # Ponderação combina SNR/amplitude (em escala suave) e ordem n (parciais mais altos
            # têm maior alavancagem para B, mas menor SNR)
            amp_w = np.array(
                [max(0.05, 10.0 ** (p.magnitude_db / 40.0)) for p in partials],
                dtype=np.float64,
            )
            f0_fit, b_raw, b_std, residual_cents, r2 = _fit_inharmonicity_irls(
                ns_arr, fs_arr, amp_w
            )
            b_val = max(1e-7, b_raw) if b_raw > -5e-4 else None
            # Atualiza ideal_freq e deviation_cents usando f0_fit refinada
            for p in partials:
                p.ideal_freq = p.n * f0_fit
                p.deviation_cents = freq_to_cents(p.measured_freq, p.ideal_freq)
        else:
            f0_fit = f0_init
            b_raw = None
            b_val = None
            b_std = float("inf")
            residual_cents = float("inf")
            r2 = 0.0

        # Prepara espectro subamostrado para visualização na interface web
        max_plot_freq = min(nyquist, (self.max_partials + 1.5) * f0_init)
        max_plot_idx = min(mag_db_rel.size, int(math.ceil(max_plot_freq / df)))
        step = max(1, max_plot_idx // 600)
        plot_indices = np.arange(0, max_plot_idx, step)
        # Mantém máximo local em cada bloco para não "achatar" picos finos na visualização
        spec_freqs: list[float] = []
        spec_db: list[float] = []
        for idx0 in plot_indices:
            idx1 = min(max_plot_idx, idx0 + step)
            sub = mag_db_rel[idx0:idx1]
            best_local = int(np.argmax(sub))
            spec_freqs.append(round(float((idx0 + best_local) * df), 2))
            spec_db.append(round(float(sub[best_local]), 2))

        return {
            "B": b_val,
            "B_raw": b_raw,
            "B_std": b_std,
            "B_residual_cents": residual_cents,
            "B_r2": r2,
            "f0_refined": f0_fit,
            "n_partials_used": len(partials),
            "partials": partials,
            "spectrum_freqs": spec_freqs,
            "spectrum_db": spec_db,
        }
