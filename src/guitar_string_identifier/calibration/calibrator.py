"""Calibração por guitarra: estima B0 e modelos complementares a partir de notas rotuladas."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from guitar_string_identifier.analysis.complementary import ComplementaryFeatureExtractor
from guitar_string_identifier.analysis.inharmonicity import InharmonicityExtractor
from guitar_string_identifier.analysis.pitch import YinPitchEstimator
from guitar_string_identifier.core.types import Note
from guitar_string_identifier.profiles.schema import AnalysisConfig, CalibrationData


@dataclass
class CalibrationSampleResult:
    """Resultado da medição de uma única nota durante a calibração."""

    string: int
    fret: int
    accepted: bool
    reason: str
    f0: float | None = None
    b_measured: float | None = None
    b0_implied: float | None = None
    residual_cents: float = float("inf")
    n_partials: int = 0
    decay_mean_db_s: float | None = None
    centroid_ratio: float | None = None


@dataclass
class CalibrationReport:
    """Relatório consolidado de uma execução de calibração."""

    calibration: CalibrationData
    samples: list[CalibrationSampleResult] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        """Quantidade de notas aceitas após filtragem por resíduo e nº de parciais."""
        return sum(1 for s in self.samples if s.accepted)

    @property
    def rejected_count(self) -> int:
        """Quantidade de notas descartadas por baixa qualidade de ajuste."""
        return sum(1 for s in self.samples if not s.accepted)


class GuitarCalibrator:
    """Estima constantes físicas por corda (B0, decaimento e centroide) a partir de notas."""

    def __init__(
        self,
        analysis_config: AnalysisConfig | None = None,
        max_residual_cents: float = 6.0,
        min_partials: int = 4,
    ) -> None:
        self.config = analysis_config or AnalysisConfig()
        self.max_residual_cents = max_residual_cents
        self.min_partials = min_partials
        self.pitch_estimator = YinPitchEstimator(transient_ms=self.config.transient_ms)
        self.inharm_extractor = InharmonicityExtractor(
            window_sec=self.config.window_sec,
            transient_ms=self.config.transient_ms,
            max_partials=self.config.max_partials,
            peak_tolerance_cents=self.config.peak_tolerance_cents,
            min_peak_snr_db=self.config.min_peak_snr_db,
        )
        self.comp_extractor = ComplementaryFeatureExtractor(
            transient_ms=self.config.transient_ms,
        )

    def measure_sample(self, note: Note, string: int, fret: int) -> CalibrationSampleResult:
        """Analisa uma nota rotulada e decide se ela é válida para calibração."""
        f0 = note.f0 if note.f0 is not None else self.pitch_estimator.estimate(note)
        if f0 is None or f0 <= 0:
            return CalibrationSampleResult(
                string=string,
                fret=fret,
                accepted=False,
                reason="Não foi possível estimar f0 com segurança.",
            )

        note.f0 = f0
        inh = self.inharm_extractor.extract(note)
        b_val = inh.get("B")
        res_cents = float(inh.get("B_residual_cents", float("inf")))
        n_part = int(inh.get("n_partials_used", 0))

        if b_val is None or b_val <= 1e-7 or b_val > 0.05:
            return CalibrationSampleResult(
                string=string,
                fret=fret,
                accepted=False,
                reason="Coeficiente B fora da faixa física válida.",
                f0=f0,
                b_measured=b_val,
                residual_cents=res_cents,
                n_partials=n_part,
            )

        if n_part < self.min_partials:
            return CalibrationSampleResult(
                string=string,
                fret=fret,
                accepted=False,
                reason=f"Poucos parciais detectados ({n_part} < {self.min_partials}).",
                f0=f0,
                b_measured=b_val,
                residual_cents=res_cents,
                n_partials=n_part,
            )

        if res_cents > self.max_residual_cents:
            return CalibrationSampleResult(
                string=string,
                fret=fret,
                accepted=False,
                reason=(
                    f"Resíduo do ajuste muito alto ({res_cents:.2f} cents > "
                    f"{self.max_residual_cents:.1f} cents)."
                ),
                f0=f0,
                b_measured=b_val,
                residual_cents=res_cents,
                n_partials=n_part,
            )

        # Lei física de corda rígida com tensão constante: B(fret) = B0 * 2^(fret / 6)
        b0_implied = b_val / (2.0 ** (fret / 6.0))
        comp = self.comp_extractor.extract(note, partials=inh.get("partials"))

        return CalibrationSampleResult(
            string=string,
            fret=fret,
            accepted=True,
            reason="Aceita.",
            f0=float(inh.get("f0_refined") or f0),
            b_measured=float(b_val),
            b0_implied=float(b0_implied),
            residual_cents=res_cents,
            n_partials=n_part,
            decay_mean_db_s=comp.get("decay_mean_db_s"),
            centroid_ratio=comp.get("spectral_centroid_ratio"),
        )

    def calibrate_from_samples(
        self,
        samples: list[CalibrationSampleResult],
        notes_label: str = "",
    ) -> CalibrationReport:
        """Consolida amostras já medidas em um objeto CalibrationData."""
        accepted = [s for s in samples if s.accepted and s.b0_implied is not None]

        by_string: dict[int, list[CalibrationSampleResult]] = {}
        for item in accepted:
            by_string.setdefault(item.string, []).append(item)

        b0_by_string: dict[int, float] = {}
        decay_intercept: dict[int, float] = {}
        decay_slope: dict[int, float] = {}
        centroid_by_string: dict[int, float] = {}

        log_b_errors: list[float] = []
        decay_errors: list[float] = []
        centroid_errors: list[float] = []

        for string_num, items in sorted(by_string.items()):
            # Mediana em escala logarítmica para robustez contra outliers
            log_b0s = np.array([math.log(i.b0_implied) for i in items if i.b0_implied], dtype=float)
            b0_est = float(math.exp(float(np.median(log_b0s))))
            b0_by_string[string_num] = b0_est

            for i in items:
                if i.b_measured:
                    b_pred = b0_est * (2.0 ** (i.fret / 6.0))
                    log_b_errors.append(math.log(i.b_measured) - math.log(b_pred))

            # Modelo linear de decaimento por corda em função da casa
            decay_pairs = [
                (float(i.fret), float(i.decay_mean_db_s))
                for i in items
                if i.decay_mean_db_s is not None and math.isfinite(i.decay_mean_db_s)
            ]
            if len(decay_pairs) >= 2:
                frets_arr = np.array([p[0] for p in decay_pairs], dtype=float)
                dec_arr = np.array([p[1] for p in decay_pairs], dtype=float)
                if np.ptp(frets_arr) > 0:
                    slope, intercept = np.polyfit(frets_arr, dec_arr, 1)
                    decay_slope[string_num] = float(slope)
                    decay_intercept[string_num] = float(intercept)
                    for f_v, d_v in decay_pairs:
                        decay_errors.append(d_v - (intercept + slope * f_v))
                else:
                    mean_d = float(np.mean(dec_arr))
                    decay_intercept[string_num] = mean_d
                    decay_slope[string_num] = 0.0
            elif len(decay_pairs) == 1:
                decay_intercept[string_num] = decay_pairs[0][1]
                decay_slope[string_num] = 0.0

            # Centroide normalizado médio por corda
            c_vals = [
                float(i.centroid_ratio)
                for i in items
                if i.centroid_ratio is not None and i.centroid_ratio > 0
            ]
            if c_vals:
                c_med = float(np.median(c_vals))
                centroid_by_string[string_num] = c_med
                for cv in c_vals:
                    centroid_errors.append(cv - c_med)

        if len(log_b_errors) >= 4:
            # Estimador robusto (MAD escalado) limitado à faixa física [0.08, 0.50]
            mad = float(np.median(np.abs(log_b_errors))) * 1.4826
            std = float(np.std(log_b_errors))
            log_b_sigma = float(np.clip(max(mad, 0.7 * std), 0.08, 0.50))
        else:
            log_b_sigma = 0.22

        if len(decay_errors) >= 4:
            decay_sigma = float(np.clip(float(np.std(decay_errors)), 1.0, 20.0))
        else:
            decay_sigma = 3.5

        if len(centroid_errors) >= 4:
            centroid_sigma = float(np.clip(float(np.std(centroid_errors)), 0.5, 10.0))
        else:
            centroid_sigma = 2.0

        cal = CalibrationData(
            b0_by_string=b0_by_string,
            log_b_sigma=round(log_b_sigma, 4),
            decay_intercept_by_string={k: round(v, 4) for k, v in decay_intercept.items()},
            decay_slope_by_string={k: round(v, 5) for k, v in decay_slope.items()},
            decay_sigma=round(decay_sigma, 4),
            centroid_ratio_by_string={k: round(v, 4) for k, v in centroid_by_string.items()},
            centroid_sigma=round(centroid_sigma, 4),
            notes_used=len(accepted),
            calibration_notes=notes_label,
        )
        return CalibrationReport(calibration=cal, samples=samples)

    def calibrate(
        self,
        labeled_notes: list[tuple[Note, int, int]],
        notes_label: str = "",
    ) -> CalibrationReport:
        """Executa calibração completa a partir de uma lista de (Note, corda, casa)."""
        samples = [self.measure_sample(note, s, f) for note, s, f in labeled_notes]
        return self.calibrate_from_samples(samples, notes_label=notes_label)
