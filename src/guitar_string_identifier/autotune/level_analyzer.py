"""Análise de nível, clipping e rejeição de entrada ruim (acorde / saturação / silêncio)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LevelAnalysisResult:
    """Resultado da análise de nível de uma nota forte."""

    peak: float
    peak_db: float
    rms_db: float
    clipped_samples: int
    clipped_ratio: float
    status: str  # "too_low" | "good" | "clipping"
    suggested_reference_gain_db: float
    message: str


@dataclass
class InputQualityResult:
    """Validação rápida de qualidade da entrada (rejeita silêncio, clipping e acordes)."""

    ok: bool
    is_silence: bool = False
    is_clipping: bool = False
    is_polyphonic: bool = False
    warning: str | None = None
    non_harmonic_energy_ratio: float = 0.0


class LevelAnalyzer:
    """Avalia pico/clipping de notas e rejeita entradas inadequadas (acorde/saturação)."""

    def __init__(
        self,
        clip_threshold: float = 0.985,
        max_clipped_ratio: float = 0.002,
        min_peak_db: float = -24.0,
    ) -> None:
        self.clip_threshold = clip_threshold
        self.max_clipped_ratio = max_clipped_ratio
        self.min_peak_db = min_peak_db

    def analyze_level(
        self,
        audio: np.ndarray,
        noise_floor_db: float | None = None,
    ) -> LevelAnalysisResult:
        """Mede pico (dBFS), RMS e proporção de clipping de uma nota forte.

        Args:
            audio: Vetor de amostras da nota tocada com força.
            noise_floor_db: Piso de ruído prévio em dBFS (opcional, para contexto).

        Returns:
            `LevelAnalysisResult` com status (`too_low`, `good` ou `clipping`) e
            orientação em linguagem simples para ajustar o ganho físico da interface.
        """
        sig = np.asarray(audio, dtype=np.float64).reshape(-1)
        if sig.size == 0:
            return LevelAnalysisResult(
                peak=0.0,
                peak_db=-120.0,
                rms_db=-120.0,
                clipped_samples=0,
                clipped_ratio=0.0,
                status="too_low",
                suggested_reference_gain_db=0.0,
                message="Nenhum áudio recebido.",
            )

        abs_sig = np.abs(sig)
        peak = float(np.max(abs_sig))
        peak_db = 20.0 * math.log10(peak + 1e-12)
        rms = float(np.sqrt(np.mean(sig**2) + 1e-18))
        rms_db = 20.0 * math.log10(rms + 1e-12)

        clipped_samples = int(np.sum(abs_sig >= self.clip_threshold))
        clipped_ratio = clipped_samples / max(1, sig.size)

        # Ganho de referência sugerido: diferença até o alvo ideal de -6 dBFS
        suggested_gain = round(float(np.clip(-6.0 - peak_db, -24.0, 24.0)), 1)

        if clipped_ratio > self.max_clipped_ratio or peak >= 0.995:
            status = "clipping"
            msg = (
                f"Sinal saturando (pico {peak_db:.1f} dBFS, {clipped_samples} amostras ceifadas). "
                "Reduza o ganho físico (knob de Gain) na sua placa de captura."
            )
        elif peak_db < self.min_peak_db:
            status = "too_low"
            extra = (
                f" (apenas {peak_db - noise_floor_db:.1f} dB acima do piso de ruído)"
                if noise_floor_db is not None
                else ""
            )
            msg = (
                f"Nível baixo demais (pico {peak_db:.1f} dBFS{extra}). "
                "Aumente o ganho físico na placa ou verifique a chave Hi-Z/Instrumento."
            )
        else:
            status = "good"
            msg = f"Nível excelente (pico {peak_db:.1f} dBFS, RMS {rms_db:.1f} dBFS, sem clipping)."

        return LevelAnalysisResult(
            peak=round(peak, 4),
            peak_db=round(peak_db, 2),
            rms_db=round(rms_db, 2),
            clipped_samples=clipped_samples,
            clipped_ratio=round(clipped_ratio, 6),
            status=status,
            suggested_reference_gain_db=suggested_gain,
            message=msg,
        )

    def check_input_quality(
        self,
        audio: np.ndarray,
        sr: int,
        f0: float | None = None,
        noise_floor_db: float | None = None,
    ) -> InputQualityResult:
        """Verifica se o áudio é uma nota isolada limpa (rejeita silêncio, clipping e acordes)."""
        sig = np.asarray(audio, dtype=np.float64).reshape(-1)
        if sig.size < 512:
            return InputQualityResult(
                ok=False,
                is_silence=True,
                warning="Áudio muito curto ou vazio.",
            )

        abs_sig = np.abs(sig)
        peak = float(np.max(abs_sig))
        rms = float(np.sqrt(np.mean(sig**2) + 1e-18))
        rms_db = 20.0 * math.log10(rms + 1e-12)

        # 1. Silêncio
        silence_thresh = (noise_floor_db + 8.0) if noise_floor_db is not None else -55.0
        if peak < 1e-4 or rms_db < silence_thresh:
            return InputQualityResult(
                ok=False,
                is_silence=True,
                warning="Sinal muito fraco ou silêncio.",
            )

        # 2. Clipping severo
        clipped_ratio = float(np.sum(abs_sig >= self.clip_threshold)) / sig.size
        if clipped_ratio > self.max_clipped_ratio or (clipped_ratio > 0.0005 and peak >= 0.995):
            return InputQualityResult(
                ok=False,
                is_clipping=True,
                warning=(
                    f"Sinal saturado ({clipped_ratio * 100:.2f}% de clipping). "
                    "Abaixe o ganho na interface de áudio e toque novamente."
                ),
            )

        # 3. Detecção clássica de acorde (múltiplas f0):
        # Analisa picos espectrais fortes na faixa de fundamentais de guitarra [70, 1200] Hz
        # que NÃO pertencem à série harmônica de f0 (nem a subharmônico f0/2).
        if f0 is not None and f0 > 50.0:
            start = min(int(0.05 * sr), max(0, sig.size // 4))
            seg = sig[start : min(sig.size, start + int(0.45 * sr))]
            if seg.size >= 2048:
                seg = seg - np.mean(seg)
                n_fft = 1 << int(np.ceil(np.log2(max(8192, seg.size * 2))))
                win = np.hanning(seg.size)
                spec = np.abs(np.fft.rfft(seg * win, n=n_fft))
                freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
                df = sr / n_fft

                # Região de interesse para detectar notas simultâneas
                lo_idx = max(1, int(math.floor(70.0 / df)))
                hi_idx = min(spec.size - 2, int(math.ceil(min(1800.0, 6.5 * f0) / df)))
                sub_spec = spec[lo_idx:hi_idx]
                if sub_spec.size > 10:
                    max_mag = float(np.max(sub_spec)) + 1e-18
                    # Encontra picos locais acima de -14 dB (~20% do pico máximo)
                    thresh = max_mag * 0.20
                    harm_energy = 0.0
                    non_harm_energy = 0.0
                    strong_non_harm_peaks = 0

                    for i in range(1, sub_spec.size - 1):
                        val = float(sub_spec[i])
                        if val >= thresh and val >= sub_spec[i - 1] and val >= sub_spec[i + 1]:
                            f_pk = float(freqs[lo_idx + i])
                            # Verifica se é múltiplo de f0 (margem para inharmonicidade)
                            ratio = f_pk / f0
                            nearest_n = max(1, round(ratio))
                            # Considera também f0/2 caso YIN tenha pego a oitava
                            cents_harm = abs(1200.0 * math.log2(f_pk / (nearest_n * f0)))
                            cents_sub = abs(1200.0 * math.log2(f_pk / (0.5 * f0)))
                            if cents_harm <= 65.0 or cents_sub <= 55.0:
                                harm_energy += val**2
                            else:
                                non_harm_energy += val**2
                                strong_non_harm_peaks += 1

                    total_e = harm_energy + non_harm_energy + 1e-18
                    non_harm_ratio = non_harm_energy / total_e
                    if strong_non_harm_peaks >= 2 and non_harm_ratio >= 0.22:
                        return InputQualityResult(
                            ok=False,
                            is_polyphonic=True,
                            non_harmonic_energy_ratio=round(non_harm_ratio, 3),
                            warning=(
                                "Múltiplas notas (acorde) detectadas. "
                                "Este modo exige uma nota isolada por vez."
                            ),
                        )

        return InputQualityResult(ok=True)
