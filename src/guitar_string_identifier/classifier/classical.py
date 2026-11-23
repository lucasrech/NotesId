"""Classificador clássico por distância de Mahalanobis / verossimilhança gaussiana."""

from __future__ import annotations

import math

import numpy as np

from guitar_string_identifier.core.interfaces import StringClassifier
from guitar_string_identifier.core.music import hz_to_midi, midi_to_note_name
from guitar_string_identifier.core.types import Candidate, Features, Note, Result
from guitar_string_identifier.profiles.schema import AnalysisConfig, CalibrationData

# Prior físico genérico de B0 para guitarra elétrica 6 cordas (caso não haja calibração)
DEFAULT_PRIOR_B0: dict[int, float] = {
    1: 2.5e-5,   # E4 lisa
    2: 4.0e-5,   # B3 lisa
    3: 9.5e-5,   # G3 lisa (mais grossa das lisas)
    4: 6.5e-5,   # D3 encapada (núcleo fino)
    5: 1.05e-4,  # A2 encapada
    6: 1.55e-4,  # E2 encapada
}


class GaussianStringClassifier(StringClassifier):
    """Classifica entre candidatos (corda, casa) via verossimilhança gaussiana clássica.

    Característica principal: log(B) comparado ao modelo físico B(s, f) = B0[s] * 2^(f/6).
    Características complementares: taxa de decaimento por parcial e centroide (baixo peso).
    """

    def __init__(
        self,
        calibration: CalibrationData | None = None,
        analysis_config: AnalysisConfig | None = None,
    ) -> None:
        self.calibration = calibration or CalibrationData()
        self.config = analysis_config or AnalysisConfig()

    def _predict_b(self, string: int, fret: int) -> float:
        b0_map = self.calibration.b0_by_string
        b0 = b0_map.get(string, DEFAULT_PRIOR_B0.get(string, 8.0e-5))
        return b0 * (2.0 ** (fret / 6.0))

    def classify(
        self,
        note: Note,
        features: Features,
        candidates: list[Candidate],
    ) -> Result:
        f0 = features.get("f0_refined") or note.f0
        note_name = (
            candidates[0].note
            if candidates
            else (midi_to_note_name(round(hz_to_midi(f0))) if f0 and f0 > 0 else "?")
        )

        if not candidates:
            return Result(
                string=None,
                fret=None,
                note=note_name,
                confidence=0.0,
                candidates=[],
                f0=f0,
                indeterminate=True,
                features=features,
            )

        # Caso trivial: apenas 1 posição possível na guitarra para esta nota
        if len(candidates) == 1:
            only = candidates[0]
            only.expected_b = self._predict_b(only.string, only.fret)
            only.distance = 0.0
            only.score = 1.0
            return Result(
                string=only.string,
                fret=only.fret,
                note=only.note,
                confidence=1.0,
                candidates=[only],
                f0=f0,
                indeterminate=False,
                features=features,
            )

        b_meas = features.get("B")
        res_cents = float(features.get("B_residual_cents", float("inf")))
        decay_meas = features.get("decay_mean_db_s")
        centroid_ratio = features.get("spectral_centroid_ratio")

        # Se o resíduo do ajuste de B for alto, amplia sigma efetivo para refletir menor certeza
        sigma_b = self.calibration.log_b_sigma
        if math.isfinite(res_cents) and res_cents > 2.5:
            sigma_b *= 1.0 + 0.15 * (res_cents - 2.5)

        distances: list[float] = []
        for cand in candidates:
            b_pred = self._predict_b(cand.string, cand.fret)
            cand.expected_b = b_pred

            d2 = 0.0
            # 1. Inharmonicidade B (característica principal)
            if b_meas is not None and b_meas > 0 and self.config.weight_inharmonicity > 0:
                z_b = (math.log(b_meas) - math.log(b_pred)) / sigma_b
                d2 += self.config.weight_inharmonicity * (z_b**2)

            # 2. Taxa de decaimento por parcial (complementar)
            if (
                decay_meas is not None
                and self.config.weight_decay > 0
                and cand.string in self.calibration.decay_intercept_by_string
            ):
                d_int = self.calibration.decay_intercept_by_string[cand.string]
                d_slp = self.calibration.decay_slope_by_string.get(cand.string, 0.0)
                decay_pred = d_int + d_slp * cand.fret
                z_d = (decay_meas - decay_pred) / self.calibration.decay_sigma
                d2 += self.config.weight_decay * (z_d**2)

            # 3. Centroide normalizado (apoio de baixo peso)
            if (
                centroid_ratio is not None
                and centroid_ratio > 0
                and self.config.weight_spectral > 0
                and cand.string in self.calibration.centroid_ratio_by_string
            ):
                c_pred = self.calibration.centroid_ratio_by_string[cand.string]
                z_c = (centroid_ratio - c_pred) / self.calibration.centroid_sigma
                d2 += self.config.weight_spectral * (z_c**2)

            cand.distance = math.sqrt(max(0.0, d2))
            distances.append(d2)

        # Converte distâncias quadradas em probabilidades posteriores (softmax de -0.5 * d^2)
        d2_arr = np.array(distances, dtype=np.float64)
        log_likes = -0.5 * d2_arr
        log_likes -= float(np.max(log_likes))
        probs = np.exp(log_likes)
        probs /= float(np.sum(probs)) + 1e-18

        for cand, prob in zip(candidates, probs, strict=True):
            cand.score = float(prob)

        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        best = ranked[0]
        second = ranked[1]

        confidence = best.score
        # Penaliza confiança se B não pôde ser medido
        if b_meas is None:
            confidence *= 0.4

        margin = best.score - second.score
        is_indeterminate = (
            confidence < self.config.min_confidence or margin < self.config.min_margin
        )

        return Result(
            string=None if is_indeterminate else best.string,
            fret=None if is_indeterminate else best.fret,
            note=best.note,
            confidence=round(float(confidence), 4),
            candidates=ranked,
            f0=f0,
            indeterminate=is_indeterminate,
            features=features,
        )
