"""Gerador sintético de corda rígida (Karplus-Strong com dispersão all-pass).

Permite sintetizar notas com frequência fundamental f0 e coeficiente de
inharmonicidade B conhecidos, para validar os algoritmos de extração de f0 e B
antes de avaliar em áudio real.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.signal import lfilter

from guitar_string_identifier.core.music import midi_to_hz, string_fret_to_midi


@dataclass
class SynthParams:
    """Parâmetros físicos para síntese de uma corda rígida."""

    f0: float
    b_coeff: float = 1e-4
    sr: int = 48000
    duration: float = 2.0
    pluck_position: float = 0.22  # fração do comprimento da corda (0 < p < 1)
    decay_base: float = 0.996  # fator de perda por período
    brightness: float = 0.5  # peso do filtro passa-baixas no loop (0..1)
    num_allpass: int = 6
    seed: int = 42


def _allpass_phase_delay(omega: float, a: float) -> float:
    """Atraso de fase (em amostras) de H(z) = (a + z^-1) / (1 + a z^-1)."""
    if abs(omega) < 1e-12:
        return (1.0 - a) / (1.0 + a)
    # Fase de H(e^{jw}) = -w - 2 * atan2(a * sin(w), 1 + a * cos(w))
    # Como H(e^{jw}) = e^{-jw} * (1 + a e^{jw}) / (1 + a e^{-jw}):
    # arg(1 + a e^{jw}) = atan2(a sin w, 1 + a cos w)
    # Logo arg(H) = -w + 2 * atan2(a sin w, 1 + a cos w)
    phase = -omega + 2.0 * math.atan2(a * math.sin(omega), 1.0 + a * math.cos(omega))
    return -phase / omega


def _solve_allpass_coeff(f0: float, b_coeff: float, sr: int, num_allpass: int = 4) -> float:
    """Resolve numericamente o coeficiente 'a' do filtro all-pass para atingir B.

    Queremos que a redução do atraso de fase entre f0 e f_ref (ex: n=6)
    reproduza o desvio de frequência f_n = n * f0 * sqrt(1 + B * n^2).
    """
    if b_coeff <= 1e-9:
        return 0.0

    n_ref = 6
    f_ref = n_ref * f0 * math.sqrt(1.0 + b_coeff * (n_ref**2))
    if f_ref >= 0.42 * sr:
        n_ref = max(2, int(0.40 * sr / f0))
        f_ref = n_ref * f0 * math.sqrt(1.0 + b_coeff * (n_ref**2))

    w0 = 2.0 * math.pi * f0 / sr
    wn = 2.0 * math.pi * f_ref / sr

    # No loop de atraso total T(w) = D + M * tau_ap(w):
    # Em w0: T(w0) = sr / f0
    # Em wn: T(wn) = n_ref * sr / f_ref = (sr / f0) / sqrt(1 + B * n_ref^2)
    # Logo precisamos de: M * (tau_ap(w0) - tau_ap(wn)) = (sr / f0) * (1 - 1/sqrt(1 + B * n^2))
    target_delta = (sr / f0) * (1.0 - 1.0 / math.sqrt(1.0 + b_coeff * (n_ref**2)))

    def residual(a: float) -> float:
        delta = num_allpass * (_allpass_phase_delay(w0, a) - _allpass_phase_delay(wn, a))
        return delta - target_delta

    # Para -1 < a < 0, tau_ap(w0) > tau_ap(wn), então delta > 0 e cresce quando a -> -1
    a_low = -0.995
    a_high = -1e-7
    if residual(a_low) < 0:
        return a_low
    if residual(a_high) > 0:
        return 0.0
    return float(brentq(residual, a_low, a_high, xtol=1e-9))


def synthesize_stiff_string(params: SynthParams) -> np.ndarray:
    """Gera áudio de uma nota de corda rígida via Karplus-Strong com dispersão.

    Retorna sinal mono float64 normalizado em [-0.95, 0.95].
    """
    sr = params.sr
    n_samples = int(round(params.duration * sr))
    f0 = params.f0
    b_coeff = max(0.0, params.b_coeff)
    m_ap = params.num_allpass

    # 1. Coeficiente all-pass de dispersão
    a_disp = _solve_allpass_coeff(f0, b_coeff, sr, num_allpass=m_ap)

    # 2. Atraso total necessário em f0
    w0 = 2.0 * math.pi * f0 / sr
    total_delay = sr / f0

    # Atraso do filtro passa-baixas de 1º ordem no loop:
    # H_lp(z) = (1 - g) + g * z^-1, atraso de fase em w0 ~ g (para w0 pequeno)
    g_lp = float(np.clip(1.0 - params.brightness, 0.05, 0.5))
    lp_phase = -math.atan2(g_lp * math.sin(w0), (1.0 - g_lp) + g_lp * math.cos(w0))
    lp_delay = -lp_phase / w0 if w0 > 1e-12 else g_lp

    disp_delay = m_ap * _allpass_phase_delay(w0, a_disp) if a_disp != 0.0 else 0.0
    rem_delay = total_delay - lp_delay - disp_delay

    # Separa parte inteira e parte fracionária (com filtro all-pass Thiran 1ª ordem)
    int_delay = int(math.floor(rem_delay))
    frac = rem_delay - int_delay
    if frac < 0.15 and int_delay > 4:
        int_delay -= 1
        frac += 1.0
    if int_delay < 2:
        int_delay = 2
        frac = 0.5

    # Thiran 1ª ordem para atraso fracionário frac: a_frac = (1 - frac) / (1 + frac)
    a_frac = (1.0 - frac) / (1.0 + frac)

    # 3. Excitação inicial (ruído filtrado com pente de posição de palhetada)
    rng = np.random.default_rng(params.seed)
    exc_len = max(int_delay, 16)
    exc = rng.standard_normal(exc_len)
    exc -= np.mean(exc)

    # Filtro pente da posição de palhetada: y[n] = x[n] - x[n - round(p * L)]
    pluck_delay = max(1, int(round(params.pluck_position * total_delay)))
    if pluck_delay < exc_len:
        exc_comb = exc.copy()
        exc_comb[pluck_delay:] -= exc[:-pluck_delay]
        exc = exc_comb

    # Suavização leve na excitação para simular palheta
    exc = lfilter([0.5, 0.5], [1.0], exc)

    # 4. Loop Karplus-Strong (implementado em buffer circular)
    ring = np.zeros(int_delay, dtype=np.float64)
    ring[:] = exc[:int_delay]

    out = np.zeros(n_samples, dtype=np.float64)
    ptr = 0

    # Estados dos filtros no loop
    lp_prev = 0.0
    frac_x_prev = 0.0
    frac_y_prev = 0.0
    ap_x_prev = np.zeros(m_ap, dtype=np.float64)
    ap_y_prev = np.zeros(m_ap, dtype=np.float64)
    loss = params.decay_base

    one_minus_g = 1.0 - g_lp

    for i in range(n_samples):
        x = ring[ptr]
        out[i] = x

        # Passa-baixas de perda
        y = loss * (one_minus_g * x + g_lp * lp_prev)
        lp_prev = x

        # Atraso fracionário all-pass: y_f[n] = a_frac * y[n] + y[n-1] - a_frac * y_f[n-1]
        yf = a_frac * y + frac_x_prev - a_frac * frac_y_prev
        frac_x_prev = y
        frac_y_prev = yf
        sig = yf

        # Cascata de filtros all-pass de dispersão
        if a_disp != 0.0:
            for k in range(m_ap):
                yk = a_disp * sig + ap_x_prev[k] - a_disp * ap_y_prev[k]
                ap_x_prev[k] = sig
                ap_y_prev[k] = yk
                sig = yk

        ring[ptr] = sig
        ptr += 1
        if ptr == int_delay:
            ptr = 0

    # Remove DC e normaliza
    out -= np.mean(out)
    peak = float(np.max(np.abs(out)))
    if peak > 1e-12:
        out = 0.95 * (out / peak)
    return out


def predict_b_from_open(b0: float, fret: int) -> float:
    """Prevê B na casa `fret` a partir de B0 da corda solta: B(fret) = B0 * 2^(fret / 6)."""
    return b0 * (2.0 ** (fret / 6.0))


def synthesize_from_guitar_position(
    string: int,
    fret: int,
    b0_by_string: dict[int, float],
    tuning: dict[int, str] | None = None,
    sr: int = 48000,
    duration: float = 2.0,
    pluck_position: float = 0.22,
    seed: int = 42,
) -> tuple[np.ndarray, float, float]:
    """Sintetiza uma nota para (corda, casa) usando a lei física B(fret) = B0 * 2^(fret/6).

    Retorna (audio, f0_hz, b_imposed).
    """
    midi = string_fret_to_midi(string, fret, tuning=tuning)
    f0 = midi_to_hz(midi)
    b0 = b0_by_string[string]
    b_val = predict_b_from_open(b0, fret)
    # Perda por período ajustada à frequência para sustain realista (~3 dB/s na fundamental)
    decay_per_period = float(np.clip(10.0 ** (-3.0 / (20.0 * max(f0, 50.0))), 0.990, 0.9995))
    params = SynthParams(
        f0=f0,
        b_coeff=b_val,
        sr=sr,
        duration=duration,
        pluck_position=pluck_position,
        decay_base=decay_per_period,
        seed=seed,
    )
    audio = synthesize_stiff_string(params)
    return audio, f0, b_val
