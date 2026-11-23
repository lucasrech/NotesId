"""Estimativa clássica de pesos das características por critério de Fisher / variância inversa."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from guitar_string_identifier.calibration.calibrator import CalibrationSampleResult


@dataclass
class WeightEstimateResult:
    """Pesos estimados exclusivamente a partir dos dados de calibração."""

    weight_inharmonicity: float
    weight_decay: float
    weight_spectral: float
    fisher_b: float
    fisher_decay: float
    fisher_spectral: float
    method: str
    note: str


class WeightEstimator:
    """Estima pesos das características (B, decaimento, espectral) sem ML.

    Usa o critério discriminante de Fisher (variância entre cordas / variância
    intra-corda) calculado exclusivamente sobre as amostras de calibração.
    """

    @staticmethod
    def _fisher_ratio(grouped_values: dict[int, list[float]]) -> float:
        """Razão de Fisher clássica: Var(médias entre classes) / Média(Var intra-classe)."""
        valid_groups = [
            np.asarray(vals, dtype=np.float64) for vals in grouped_values.values() if len(vals) >= 2
        ]
        if len(valid_groups) < 2:
            return 0.0

        means = np.array([float(np.mean(g)) for g in valid_groups], dtype=np.float64)
        within_vars = np.array([float(np.var(g, ddof=1)) for g in valid_groups], dtype=np.float64)
        var_between = float(np.var(means, ddof=1))
        var_within = float(np.mean(within_vars)) + 1e-12
        return max(0.0, var_between / var_within)

    def estimate(self, samples: list[CalibrationSampleResult]) -> WeightEstimateResult:
        """Calcula pesos relativos normalizados com weight_inharmonicity = 1.0."""
        accepted = [s for s in samples if s.accepted and s.b0_implied is not None]
        if len(accepted) < 6:
            return WeightEstimateResult(
                weight_inharmonicity=1.0,
                weight_decay=0.25,
                weight_spectral=0.05,
                fisher_b=0.0,
                fisher_decay=0.0,
                fisher_spectral=0.0,
                method="padrão (poucas amostras)",
                note="Menos de 6 notas aceitas; mantidos os pesos físicos padrão.",
            )

        log_b0_by_s: dict[int, list[float]] = {}
        decay_by_s: dict[int, list[float]] = {}
        cent_by_s: dict[int, list[float]] = {}

        for item in accepted:
            if item.b0_implied and item.b0_implied > 0:
                log_b0_by_s.setdefault(item.string, []).append(math.log(item.b0_implied))
            if item.decay_mean_db_s is not None and math.isfinite(item.decay_mean_db_s):
                # Remove efeito grosseiro de casa (aproximação linear local)
                decay_by_s.setdefault(item.string, []).append(float(item.decay_mean_db_s))
            if item.centroid_ratio is not None and item.centroid_ratio > 0:
                cent_by_s.setdefault(item.string, []).append(float(item.centroid_ratio))

        f_b = self._fisher_ratio(log_b0_by_s)
        f_dec = self._fisher_ratio(decay_by_s)
        f_spec = self._fisher_ratio(cent_by_s)

        if f_b <= 1e-6:
            return WeightEstimateResult(
                weight_inharmonicity=1.0,
                weight_decay=0.25,
                weight_spectral=0.05,
                fisher_b=round(f_b, 3),
                fisher_decay=round(f_dec, 3),
                fisher_spectral=round(f_spec, 3),
                method="padrão (Fisher B nulo)",
                note="Variância entre cordas insuficiente; mantidos pesos padrão.",
            )

        # Normaliza em relação a B (= 1.0) e limita características auxiliares
        # para que centroide/roll-off permaneçam de baixo peso (apoio)
        w_dec = float(np.clip(f_dec / f_b, 0.05, 0.85))
        w_spec = float(np.clip(f_spec / f_b, 0.0, 0.15))

        return WeightEstimateResult(
            weight_inharmonicity=1.0,
            weight_decay=round(w_dec, 3),
            weight_spectral=round(w_spec, 3),
            fisher_b=round(f_b, 3),
            fisher_decay=round(f_dec, 3),
            fisher_spectral=round(f_spec, 3),
            method="critério discriminante de Fisher (dados de calibração)",
            note=(
                f"Razões de Fisher na calibração: B={f_b:.2f}, decaimento={f_dec:.2f}, "
                f"centroide={f_spec:.2f}."
            ),
        )
