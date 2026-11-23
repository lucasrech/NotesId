"""Detector de ronco da rede elétrica (60 Hz / 50 Hz e harmônicos) no silêncio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HumDetectionResult:
    """Resultado da detecção de ronco de rede elétrica."""

    detected: bool
    fundamental_hz: float | None
    harmonics_hz: list[float]
    peak_prominence_db: float
    message: str


class HumDetector:
    """Detecta picos estreitos em 60 Hz (prioridade no Brasil) ou 50 Hz e harmônicos."""

    def __init__(
        self,
        min_prominence_db: float = 10.0,
        max_harmonics: int = 8,
        priority_hz: float = 60.0,
    ) -> None:
        self.min_prominence_db = min_prominence_db
        self.max_harmonics = max_harmonics
        self.priority_hz = priority_hz

    def _measure_grid_score(
        self,
        mag_db: np.ndarray,
        df: float,
        base_hz: float,
        sr: int,
    ) -> tuple[float, list[float]]:
        """Mede proeminência média dos harmônicos de base_hz acima do piso local."""
        prominences: list[float] = []
        detected_harmonics: list[float] = []

        for k in range(1, self.max_harmonics + 1):
            freq = k * base_hz
            if freq >= 0.45 * sr:
                break
            idx_c = int(round(freq / df))
            # Pico estreito em ±2.5 Hz
            half_peak = max(1, int(round(2.5 / df)))
            p0 = max(1, idx_c - half_peak)
            p1 = min(mag_db.size - 1, idx_c + half_peak + 1)
            peak_val = float(np.max(mag_db[p0:p1]))

            # Piso local em [freq - 14 Hz, freq - 4 Hz] U [freq + 4 Hz, freq + 14 Hz]
            b_lo0 = max(1, int(round((freq - 14.0) / df)))
            b_lo1 = max(1, int(round((freq - 4.0) / df)))
            b_hi0 = min(mag_db.size - 1, int(round((freq + 4.0) / df)))
            b_hi1 = min(mag_db.size - 1, int(round((freq + 14.0) / df)))
            bg = np.concatenate([mag_db[b_lo0:b_lo1], mag_db[b_hi0:b_hi1]])
            if bg.size < 4:
                continue
            noise_floor = float(np.median(bg))
            prom = peak_val - noise_floor
            if prom >= self.min_prominence_db:
                detected_harmonics.append(round(freq, 1))
                prominences.append(prom)

        if not prominences:
            return 0.0, []
        # Pontuação privilegia ter fundamental + harmônicos consistentes
        max_prom = float(np.max(prominences))
        return max_prom, detected_harmonics

    def detect(self, silence_audio: np.ndarray, sr: int) -> HumDetectionResult:
        """Detecta se há ronco estreito em 60 Hz (prioridade Brasil) ou 50 Hz e harmônicos.

        Args:
            silence_audio: Trecho de silêncio capturado (~2 s) com volume da guitarra aberto.
            sr: Taxa de amostragem em Hz.

        Returns:
            `HumDetectionResult` indicando se houve detecção e quais faixas excluir.
        """
        sig = np.asarray(silence_audio, dtype=np.float64).reshape(-1)
        if sig.size < int(0.4 * sr):
            return HumDetectionResult(
                detected=False,
                fundamental_hz=None,
                harmonics_hz=[],
                peak_prominence_db=0.0,
                message="Áudio de silêncio curto demais para detectar ronco de 50/60 Hz.",
            )

        sig = sig - np.mean(sig)
        # Se o sinal for silêncio digital absoluto, não há ronco
        if float(np.max(np.abs(sig))) < 1e-7:
            return HumDetectionResult(
                detected=False,
                fundamental_hz=None,
                harmonics_hz=[],
                peak_prominence_db=0.0,
                message="Nenhum ronco de rede detectado (silêncio limpo).",
            )

        n_fft = 1 << int(np.ceil(np.log2(max(sig.size * 2, 16384))))
        win = np.hanning(sig.size)
        spec = np.abs(np.fft.rfft(sig * win, n=n_fft)) + 1e-18
        mag_db = 20.0 * np.log10(spec)
        df = sr / n_fft

        score_60, harm_60 = self._measure_grid_score(mag_db, df, 60.0, sr)
        score_50, harm_50 = self._measure_grid_score(mag_db, df, 50.0, sr)

        # Prioriza 60 Hz (Brasil) em caso de empate próximo (diferença < 3 dB)
        chosen_hz: float | None = None
        chosen_harms: list[float] = []
        chosen_prom = 0.0

        if harm_60 and (not harm_50 or score_60 + 3.0 >= score_50):
            chosen_hz = 60.0
            chosen_harms = harm_60
            chosen_prom = score_60
        elif harm_50:
            chosen_hz = 50.0
            chosen_harms = harm_50
            chosen_prom = score_50

        if chosen_hz is None:
            return HumDetectionResult(
                detected=False,
                fundamental_hz=None,
                harmonics_hz=[],
                peak_prominence_db=0.0,
                message="Nenhum ronco significativo de 50/60 Hz detectado.",
            )

        return HumDetectionResult(
            detected=True,
            fundamental_hz=chosen_hz,
            harmonics_hz=chosen_harms,
            peak_prominence_db=round(chosen_prom, 1),
            message=(
                f"Ronco de rede detectado em {int(chosen_hz)} Hz "
                f"({len(chosen_harms)} faixas até {int(chosen_harms[-1])} Hz, "
                f"+{chosen_prom:.1f} dB). Essas faixas serão excluídas da busca de parciais."
            ),
        )
