"""Resolução adaptativa (modo 'auto') dos parâmetros de análise em tempo de execução."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ResolvedAnalysisParams:
    """Parâmetros efetivos escolhidos para uma nota específica."""

    window_sec: float
    transient_ms: float
    max_partials: int
    peak_tolerance_cents: float
    iterative_narrowing: bool


class AdaptiveParams:
    """Calcula janela FFT, descarte do transiente, nº de parciais e tolerância de picos."""

    def estimate_transient_ms(
        self,
        audio: np.ndarray,
        sr: int,
        min_ms: float = 15.0,
        max_ms: float = 120.0,
    ) -> float:
        """Detecta onde a energia do ataque estabiliza (em vez de 50 ms fixos)."""
        sig = np.asarray(audio, dtype=np.float64).reshape(-1)
        hop = max(64, int(0.005 * sr))  # passo de 5 ms
        frame_len = max(128, int(0.015 * sr))
        max_scan = min(sig.size, int(0.25 * sr))
        if max_scan <= frame_len + hop:
            return 35.0

        n_frames = (max_scan - frame_len) // hop
        if n_frames < 5:
            return 35.0

        rms_env = np.empty(n_frames, dtype=np.float64)
        for i in range(n_frames):
            seg = sig[i * hop : i * hop + frame_len]
            rms_env[i] = math.sqrt(float(np.mean(seg**2)) + 1e-18)

        peak_idx = int(np.argmax(rms_env))
        # Após o pico, encontra onde a taxa de variação relativa da envoltória estabiliza
        # (queda suave < 18% por passo de 5 ms por 2 passos seguidos)
        settle_idx = peak_idx + 2
        for i in range(peak_idx + 1, n_frames - 2):
            r0 = rms_env[i] + 1e-12
            rel_drop1 = abs(rms_env[i + 1] - r0) / r0
            rel_drop2 = abs(rms_env[i + 2] - rms_env[i + 1]) / (rms_env[i + 1] + 1e-12)
            if rel_drop1 < 0.18 and rel_drop2 < 0.18:
                settle_idx = i
                break

        transient_ms = (settle_idx * hop / sr) * 1000.0
        return round(float(np.clip(transient_ms, min_ms, max_ms)), 1)

    def estimate_sustain_sec(
        self,
        audio: np.ndarray,
        sr: int,
        transient_ms: float,
        drop_db: float = 28.0,
    ) -> float:
        """Estima a duração útil de sustain após o transiente (até cair `drop_db`)."""
        sig = np.asarray(audio, dtype=np.float64).reshape(-1)
        start = min(sig.size - 1, int(round((transient_ms / 1000.0) * sr)))
        steady = sig[start:]
        if steady.size < int(0.15 * sr):
            return steady.size / sr

        hop = max(256, int(0.02 * sr))
        n_frames = max(1, steady.size // hop)
        rms = np.array(
            [
                math.sqrt(float(np.mean(steady[i * hop : (i + 1) * hop] ** 2)) + 1e-18)
                for i in range(n_frames)
            ],
            dtype=np.float64,
        )
        rms_db = 20.0 * np.log10(rms + 1e-12)
        ref_db = float(np.max(rms_db[: min(5, rms_db.size)]))
        above = np.flatnonzero(rms_db >= ref_db - drop_db)
        if above.size == 0:
            return min(0.5, steady.size / sr)
        return float((int(above[-1]) + 1) * hop / sr)

    def resolve_window_sec(
        self,
        f0: float,
        sustain_sec: float,
        min_sec: float = 0.55,
        max_sec: float = 1.0,
    ) -> float:
        """Escolhe janela FFT a partir da f0 e do sustain disponível.

        Garante janela longa (>= 0,55 s quando há sustain) para alta resolução
        na estimativa de B, estendendo até 1,0 s em notas graves (E2/A2).
        """
        cycles_target = 80.0 / max(f0, 65.0)
        ideal = float(np.clip(cycles_target, min_sec, max_sec))
        # Não excede o sustain útil disponível (mas garante ao menos 0,30 s se existir)
        usable = max(0.30, min(ideal, sustain_sec))
        return round(usable, 3)

    def resolve_max_partials(
        self,
        f0: float,
        sr: int,
        cap: int = 12,
    ) -> int:
        """Determina o limite superior de parciais abaixo de Nyquist (0,45 * sr)."""
        if f0 <= 0:
            return 12
        nyq_limit = int(math.floor((0.45 * sr) / f0))
        return int(np.clip(nyq_limit, 4, cap))

    def resolve_peak_tolerance_cents(self, expected_b: float | None = None) -> float:
        """Deriva tolerância inicial de busca de picos a partir do B esperado."""
        if expected_b is None or expected_b <= 0:
            return 45.0
        # Deslocamento esperado em n=4: ~1200 * log2(sqrt(1 + B*16))
        shift_n4 = 600.0 * math.log2(1.0 + expected_b * 16.0)
        return round(float(np.clip(30.0 + 1.5 * shift_n4, 30.0, 75.0)), 1)

    def resolve_for_note(
        self,
        audio: np.ndarray,
        sr: int,
        f0: float,
        param_modes: dict[str, str],
        manual_window_sec: float,
        manual_transient_ms: float,
        manual_max_partials: int,
        manual_peak_tol: float,
        expected_b: float | None = None,
    ) -> ResolvedAnalysisParams:
        """Resolve todos os parâmetros de análise respeitando o modo auto/manual de cada um."""
        if param_modes.get("transient_ms") == "auto":
            transient_ms = self.estimate_transient_ms(audio, sr)
        else:
            transient_ms = manual_transient_ms

        if param_modes.get("window_sec") == "auto":
            sustain = self.estimate_sustain_sec(audio, sr, transient_ms)
            window_sec = self.resolve_window_sec(f0, sustain)
        else:
            window_sec = manual_window_sec

        if param_modes.get("max_partials") == "auto":
            max_partials = self.resolve_max_partials(f0, sr)
        else:
            max_partials = manual_max_partials

        iter_narrow = param_modes.get("peak_tolerance_cents") == "auto"
        peak_tol = self.resolve_peak_tolerance_cents(expected_b) if iter_narrow else manual_peak_tol

        return ResolvedAnalysisParams(
            window_sec=window_sec,
            transient_ms=transient_ms,
            max_partials=max_partials,
            peak_tolerance_cents=peak_tol,
            iterative_narrowing=iter_narrow,
        )
