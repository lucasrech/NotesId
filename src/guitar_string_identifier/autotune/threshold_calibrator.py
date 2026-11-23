"""Calibração automática de limiares (confiança mínima e margem) para taxa de erro alvo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThresholdCalibrationResult:
    """Resultado da busca de limiares de decisão na curva confiança × erro."""

    min_confidence: float
    min_margin: float
    achieved_error_rate: float
    achieved_coverage: float
    n_validation_samples: int
    note: str


class ThresholdCalibrator:
    """Escolhe (min_confidence, min_margin) para atingir `target_error_rate` com maior cobertura.

    Deve receber APENAS predições de validação geradas dentro do conjunto de calibração
    (ex.: validação cruzada por amostra/captador), nunca o conjunto de teste!
    """

    def calibrate(
        self,
        validation_records: list[dict],
        target_error_rate: float = 0.10,
    ) -> ThresholdCalibrationResult:
        """Recebe registros com {confidence, margin, correct, n_candidates}."""
        ambig = [r for r in validation_records if r.get("n_candidates", 2) >= 2]
        if len(ambig) < 8:
            return ThresholdCalibrationResult(
                min_confidence=0.45,
                min_margin=0.10,
                achieved_error_rate=0.0,
                achieved_coverage=1.0,
                n_validation_samples=len(ambig),
                note="Amostras de validação insuficientes (< 8); mantidos limiares padrão.",
            )

        best_conf = 0.45
        best_margin = 0.10
        best_cov = -1.0
        best_err = 1.0
        found_target = False

        # Grade clássica e determinística sobre (min_confidence, min_margin)
        conf_grid = np.linspace(0.40, 0.95, 23)
        margin_grid = np.linspace(0.05, 0.50, 10)

        for c_th in conf_grid:
            for m_th in margin_grid:
                decided = [r for r in ambig if r["confidence"] >= c_th and r["margin"] >= m_th]
                if len(decided) < max(3, int(0.15 * len(ambig))):
                    continue
                err_rate = 1.0 - (sum(1 for r in decided if r["correct"]) / len(decided))
                cov = len(decided) / len(ambig)

                if err_rate <= target_error_rate:
                    if (
                        not found_target
                        or cov > best_cov
                        or (cov == best_cov and err_rate < best_err)
                    ):
                        found_target = True
                        best_conf = float(c_th)
                        best_margin = float(m_th)
                        best_cov = cov
                        best_err = err_rate
                elif not found_target and (
                    err_rate < best_err or (err_rate == best_err and cov > best_cov)
                ):
                    best_conf = float(c_th)
                    best_margin = float(m_th)
                    best_cov = cov
                    best_err = err_rate

        note = (
            f"Calibrado na validação interna ({len(ambig)} notas ambíguas) para erro alvo "
            f"≤ {target_error_rate * 100:.0f}%: erro estimado {best_err * 100:.1f}% "
            f"com cobertura {best_cov * 100:.1f}%."
        )
        return ThresholdCalibrationResult(
            min_confidence=round(best_conf, 2),
            min_margin=round(best_margin, 2),
            achieved_error_rate=round(best_err, 4),
            achieved_coverage=round(best_cov, 4),
            n_validation_samples=len(ambig),
            note=note,
        )
