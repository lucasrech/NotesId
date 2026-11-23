"""Sessão stateful do assistente de Auto Tuning (ao vivo e arquivo/dataset).

Nunca sobrescreve o perfil ativo automaticamente: acumula medições e gera uma
proposta estruturada (valor atual vs valor proposto + justificativa) para
confirmação explícita do usuário.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from guitar_string_identifier.autotune.device_probe import DeviceProbe
from guitar_string_identifier.autotune.hum_detector import HumDetector
from guitar_string_identifier.autotune.level_analyzer import LevelAnalyzer
from guitar_string_identifier.autotune.noise_floor import NoiseFloorEstimator
from guitar_string_identifier.autotune.threshold_calibrator import ThresholdCalibrator
from guitar_string_identifier.autotune.tuning_detector import (
    StringTuningMeasurement,
    TuningDetector,
)
from guitar_string_identifier.autotune.weight_estimator import WeightEstimator
from guitar_string_identifier.calibration.calibrator import (
    CalibrationSampleResult,
    GuitarCalibrator,
)
from guitar_string_identifier.core.types import Note
from guitar_string_identifier.profiles.schema import GuitarProfile


@dataclass
class ProposalChange:
    """Uma linha da tabela de resumo do Auto Tuning."""

    param: str
    label: str
    current_value: Any
    proposed_value: Any
    mode: Literal["auto", "manual"] | None
    how_obtained: str


@dataclass
class AutoTuneProposal:
    """Proposta consolidada pronta para revisão e confirmação do usuário."""

    session_id: str
    mode: Literal["live", "dataset"]
    base_profile_name: str
    proposed_profile: GuitarProfile
    changes: list[ProposalChange] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    step_statuses: dict[str, str] = field(default_factory=dict)


class AutoTuneSession:
    """Orquestra as etapas do assistente de Auto Tuning sem alterar o perfil original."""

    def __init__(
        self,
        base_profile: GuitarProfile,
        mode: Literal["live", "dataset"] = "live",
    ) -> None:
        self.session_id = uuid.uuid4().hex[:12]
        self.mode = mode
        self.base_profile = base_profile.model_copy(deep=True)
        self.proposed = base_profile.model_copy(deep=True)

        self.device_probe = DeviceProbe()
        self.noise_estimator = NoiseFloorEstimator()
        self.hum_detector = HumDetector()
        self.level_analyzer = LevelAnalyzer()
        self.tuning_detector = TuningDetector()
        self.weight_estimator = WeightEstimator()
        self.threshold_calibrator = ThresholdCalibrator()

        self.tuning_measurements: dict[int, StringTuningMeasurement] = {}
        self.calibration_samples: dict[tuple[int, int], CalibrationSampleResult] = {}
        self.reasons: dict[str, str] = {}
        self.warnings: list[str] = []
        self.step_statuses: dict[str, str] = {
            "device": "pending" if mode == "live" else "skipped",
            "silence": "pending" if mode == "live" else "skipped",
            "level": "pending" if mode == "live" else "skipped",
            "tuning": "pending",
            "calibration": "pending",
        }

        if mode == "live":
            self.warnings.append(
                "Windows: o navegador não consegue desativar os 'Aprimoramentos de áudio' "
                "(cancelamento de eco/ruído do sistema). Desative manualmente em: "
                "Configurações → Sistema → Som → Propriedades do dispositivo → "
                "Aprimoramentos de áudio → Desativado."
            )
            self.warnings.append(
                "Hardware: ganho físico (knob Gain), chave Hi-Z/Instrumento e a entrada "
                "física da interface não podem ser alterados pelo navegador."
            )

    def apply_device_step(
        self,
        device_name: str | None,
        sample_rate: int,
        channel: int = 0,
        base_latency_sec: float | None = None,
    ) -> dict[str, Any]:
        """Etapa 1 (ao vivo): grava placa escolhida, taxa real, canal e buffer sugerido."""
        buf_size = self.device_probe.suggest_buffer_size(sample_rate, base_latency_sec)
        self.proposed.capture.source_mode = "live_browser"
        if device_name:
            self.proposed.capture.device_name = device_name
            self.reasons["capture.device_name"] = "Selecionada no assistente de placa ao vivo"
        self.proposed.capture.sample_rate = int(sample_rate)
        self.proposed.param_modes["sample_rate"] = "auto"
        self.reasons["capture.sample_rate"] = f"Lida diretamente do AudioContext ({sample_rate} Hz)"

        self.proposed.capture.channel = int(channel)
        self.reasons["capture.channel"] = f"Canal ativo detectado (#{channel})"

        self.proposed.capture.buffer_size = buf_size
        self.proposed.param_modes["buffer_size"] = "auto"
        lat_txt = (
            f"baseLatency={base_latency_sec * 1000:.1f} ms"
            if base_latency_sec is not None
            else "padrão estável WebSocket"
        )
        self.reasons["capture.buffer_size"] = f"Sugerido a partir de {lat_txt}"

        self.step_statuses["device"] = "ok"
        return {
            "device_name": self.proposed.capture.device_name,
            "sample_rate": self.proposed.capture.sample_rate,
            "channel": self.proposed.capture.channel,
            "buffer_size": buf_size,
        }

    def apply_silence_step(self, audio: np.ndarray, sr: int) -> dict[str, Any]:
        """Etapa 2 (ao vivo): mede piso de ruído e detecta ronco de 50/60 Hz."""
        nf = self.noise_estimator.estimate(audio, sr)
        hum = self.hum_detector.detect(audio, sr)

        self.proposed.capture.noise_floor_db = nf.rms_db
        self.proposed.capture.onset_threshold_db = nf.suggested_onset_threshold_db
        self.reasons["capture.noise_floor_db"] = (
            f"Medido em 2 s de silêncio (RMS {nf.rms_db:.1f} dBFS)"
        )
        self.reasons["capture.onset_threshold_db"] = (
            f"Derivado do piso de ruído + margem ({nf.suggested_onset_threshold_db:.1f} dBFS)"
        )

        self.proposed.capture.hum_freq_hz = hum.fundamental_hz
        self.proposed.capture.hum_notch_freqs_hz = hum.harmonics_hz
        if hum.detected:
            self.reasons["capture.hum_notch_freqs_hz"] = (
                f"Ronco de {int(hum.fundamental_hz or 60)} Hz detectado "
                f"({len(hum.harmonics_hz)} faixas excluídas)"
            )
            self.warnings.append(hum.message)
        else:
            self.reasons["capture.hum_notch_freqs_hz"] = "Nenhum ronco de 50/60 Hz detectado"

        if nf.status == "too_loud":
            self.step_statuses["silence"] = "warn"
            self.warnings.append(nf.message)
        else:
            self.step_statuses["silence"] = "ok"

        return {
            "noise_floor": {
                "rms_db": nf.rms_db,
                "peak_db": nf.peak_db,
                "suggested_onset_threshold_db": nf.suggested_onset_threshold_db,
                "estimated_snr_db": nf.estimated_snr_db_at_minus12,
                "status": nf.status,
                "message": nf.message,
            },
            "hum": {
                "detected": hum.detected,
                "fundamental_hz": hum.fundamental_hz,
                "harmonics_hz": hum.harmonics_hz,
                "peak_prominence_db": hum.peak_prominence_db,
                "message": hum.message,
            },
        }

    def apply_level_step(self, audio: np.ndarray) -> dict[str, Any]:
        """Etapa 3 (ao vivo): mede pico, clipping e sugere ganho de referência."""
        lvl = self.level_analyzer.analyze_level(
            audio, noise_floor_db=self.proposed.capture.noise_floor_db
        )
        self.proposed.capture.reference_gain_db = lvl.suggested_reference_gain_db
        self.proposed.param_modes["reference_gain_db"] = "auto"
        self.reasons["capture.reference_gain_db"] = (
            f"Sugerido a partir do pico ({lvl.peak_db:.1f} dBFS) visando -6 dBFS"
        )

        if lvl.status == "good":
            self.step_statuses["level"] = "ok"
        elif lvl.status == "clipping":
            self.step_statuses["level"] = "error"
            self.warnings.append(lvl.message)
        else:
            self.step_statuses["level"] = "warn"
            self.warnings.append(lvl.message)

        return {
            "peak": lvl.peak,
            "peak_db": lvl.peak_db,
            "rms_db": lvl.rms_db,
            "clipped_samples": lvl.clipped_samples,
            "clipped_ratio": lvl.clipped_ratio,
            "status": lvl.status,
            "suggested_reference_gain_db": lvl.suggested_reference_gain_db,
            "message": lvl.message,
        }

    def record_tuning_string(
        self,
        string: int,
        audio: np.ndarray,
        sr: int,
    ) -> dict[str, Any]:
        """Etapa 4: mede uma corda solta e atualiza resumo da afinação."""
        expected = self.base_profile.instrument.tuning.get(string)
        meas = self.tuning_detector.measure_open_string(
            audio, sr, string=string, expected_note=expected
        )
        if meas is None:
            return {
                "string": string,
                "accepted": False,
                "reason": "Não foi possível detectar f0 clara. Toque a corda isoladamente.",
            }

        self.tuning_measurements[string] = meas
        summary = self.tuning_detector.summarize(list(self.tuning_measurements.values()))

        # Atualiza proposta
        for s, note_name in summary.tuning.items():
            self.proposed.instrument.tuning[s] = note_name
        self.proposed.instrument.tuning_cents_offset = dict(summary.cents_offset_by_string)
        self.proposed.instrument.a4_hz = summary.a4_hz
        self.proposed.instrument.tuning_preset_name = summary.preset_name

        self.reasons["instrument.tuning"] = f"Detectada nas cordas soltas ({summary.preset_name})"
        self.reasons["instrument.a4_hz"] = (
            f"Estimada pela mediana dos desvios ({summary.global_cents_offset:+.1f} cents)"
        )

        accepted_n = sum(1 for m in self.tuning_measurements.values() if m.accepted)
        self.step_statuses["tuning"] = "ok" if accepted_n >= 6 else "warn"

        return {
            "string": meas.string,
            "accepted": meas.accepted,
            "reason": meas.reason,
            "f0_hz": meas.f0_hz,
            "nearest_note": meas.nearest_note,
            "cents_from_440": meas.cents_from_440,
            "summary": {
                "preset_name": summary.preset_name,
                "a4_hz": summary.a4_hz,
                "is_alternate_tuning": summary.is_alternate_tuning,
                "needs_confirmation": summary.needs_confirmation,
                "tuning": summary.tuning,
                "cents_offset_by_string": summary.cents_offset_by_string,
            },
        }

    def record_calibration_note(
        self,
        string: int,
        fret: int,
        audio: np.ndarray,
        sr: int,
    ) -> dict[str, Any]:
        """Etapa 5: mede uma nota de calibração (com rejeição automática por resíduo)."""
        calibrator = GuitarCalibrator(analysis_config=self.proposed.analysis)
        sample = calibrator.measure_sample(Note(audio=audio, sr=sr), string, fret)
        if sample.accepted:
            self.calibration_samples[(string, fret)] = sample
            self._consolidate_calibration()
        return {
            "string": sample.string,
            "fret": sample.fret,
            "accepted": sample.accepted,
            "reason": sample.reason,
            "f0": round(sample.f0, 2) if sample.f0 else None,
            "b_measured": sample.b_measured,
            "b0_implied": sample.b0_implied,
            "residual_cents": (
                round(sample.residual_cents, 2) if np.isfinite(sample.residual_cents) else None
            ),
            "n_partials": sample.n_partials,
        }

    def enable_adaptive_analysis(self) -> None:
        """Ativa o modo `auto` para os parâmetros de análise adaptativos (seção 3.4)."""
        for key in ("window_sec", "transient_ms", "max_partials", "peak_tolerance_cents"):
            self.proposed.param_modes[key] = "auto"
        self.reasons["analysis.window_sec"] = "Modo auto: janela adaptada à f0 e duração de sustain"
        self.reasons["analysis.transient_ms"] = (
            "Modo auto: descarte detectado onde a energia do ataque estabiliza"
        )
        self.reasons["analysis.max_partials"] = (
            "Modo auto: todos os parciais acima do ruído e abaixo de Nyquist"
        )
        self.reasons["analysis.peak_tolerance_cents"] = (
            "Modo auto: estreitamento iterativo a partir do B estimado"
        )

    def _consolidate_calibration(self) -> None:
        """Atualiza CalibrationData e pesos a partir das amostras de calibração aceitas."""
        samples = list(self.calibration_samples.values())
        if not samples:
            return
        calibrator = GuitarCalibrator(analysis_config=self.proposed.analysis)
        report = calibrator.calibrate_from_samples(
            samples,
            notes_label=f"Auto Tuning ({self.mode}): {len(samples)} notas aceitas.",
        )
        self.proposed.calibration = report.calibration
        self.reasons["calibration.b0_by_string"] = (
            f"Calibrado em {report.accepted_count} notas (σ(ln B)={report.calibration.log_b_sigma})"
        )

        # Estima pesos por critério de Fisher exclusivamente nos dados de calibração
        w_est = self.weight_estimator.estimate(samples)
        self.proposed.analysis.weight_inharmonicity = w_est.weight_inharmonicity
        self.proposed.analysis.weight_decay = w_est.weight_decay
        self.proposed.analysis.weight_spectral = w_est.weight_spectral
        self.proposed.param_modes["weights"] = "auto"
        self.reasons["analysis.weights"] = w_est.note

        self.step_statuses["calibration"] = (
            "ok" if len(report.calibration.b0_by_string) >= 6 else "warn"
        )

    def apply_threshold_calibration(
        self,
        validation_records: list[dict],
        target_error_rate: float | None = None,
    ) -> dict[str, Any]:
        """Calibra min_confidence e min_margin a partir de validação interna da calibração."""
        target = target_error_rate or self.proposed.analysis.target_error_rate
        res = self.threshold_calibrator.calibrate(validation_records, target_error_rate=target)
        self.proposed.analysis.min_confidence = res.min_confidence
        self.proposed.analysis.min_margin = res.min_margin
        self.proposed.param_modes["min_confidence"] = "auto"
        self.proposed.param_modes["min_margin"] = "auto"
        self.reasons["analysis.min_confidence"] = res.note
        self.reasons["analysis.min_margin"] = res.note
        return {
            "min_confidence": res.min_confidence,
            "min_margin": res.min_margin,
            "achieved_error_rate": res.achieved_error_rate,
            "achieved_coverage": res.achieved_coverage,
            "note": res.note,
        }

    def build_proposal(self) -> AutoTuneProposal:
        """Gera tabela comparativa (valor atual vs proposto + origem) sem salvar nada."""
        self.enable_adaptive_analysis()
        changes: list[ProposalChange] = []

        def add_change(
            param: str,
            label: str,
            cur: Any,
            prop: Any,
            mode_key: str | None = None,
            default_reason: str = "Mantido",
        ) -> None:
            mode_val = self.proposed.param_modes.get(mode_key) if mode_key else None
            reason = self.reasons.get(param, default_reason)
            if cur != prop or param in self.reasons:
                changes.append(
                    ProposalChange(
                        param=param,
                        label=label,
                        current_value=cur,
                        proposed_value=prop,
                        mode=mode_val,
                        how_obtained=reason,
                    )
                )

        bc, pc = self.base_profile.capture, self.proposed.capture
        bi, pi = self.base_profile.instrument, self.proposed.instrument
        ba, pa = self.base_profile.analysis, self.proposed.analysis

        add_change(
            "capture.device_name",
            "Placa de captura",
            bc.device_name or "Padrão",
            pc.device_name or "Padrão",
        )
        add_change(
            "capture.sample_rate",
            "Taxa de amostragem (Hz)",
            bc.sample_rate,
            pc.sample_rate,
            "sample_rate",
        )
        add_change("capture.channel", "Canal ativo", bc.channel, pc.channel)
        add_change(
            "capture.buffer_size",
            "Tamanho do buffer",
            bc.buffer_size,
            pc.buffer_size,
            "buffer_size",
        )
        add_change(
            "capture.noise_floor_db", "Piso de ruído (dBFS)", bc.noise_floor_db, pc.noise_floor_db
        )
        add_change(
            "capture.onset_threshold_db",
            "Limiar de ataque (dBFS)",
            bc.onset_threshold_db,
            pc.onset_threshold_db,
        )
        add_change(
            "capture.reference_gain_db",
            "Ganho de referência (dB)",
            bc.reference_gain_db,
            pc.reference_gain_db,
            "reference_gain_db",
        )
        add_change(
            "capture.hum_notch_freqs_hz",
            "Faixas de ronco excluídas (Hz)",
            bc.hum_notch_freqs_hz,
            pc.hum_notch_freqs_hz,
        )

        add_change("instrument.tuning", "Afinação", bi.tuning, pi.tuning)
        add_change("instrument.a4_hz", "Referência A4 (Hz)", bi.a4_hz, pi.a4_hz)

        add_change(
            "analysis.window_sec",
            "Janela FFT (s)",
            ba.window_sec,
            "auto (adaptativa)",
            "window_sec",
        )
        add_change(
            "analysis.transient_ms",
            "Descarte do transiente (ms)",
            ba.transient_ms,
            "auto (adaptativo)",
            "transient_ms",
        )
        add_change(
            "analysis.max_partials",
            "Nº de parciais",
            ba.max_partials,
            "auto (até Nyquist)",
            "max_partials",
        )
        add_change(
            "analysis.peak_tolerance_cents",
            "Tolerância de picos (cents)",
            ba.peak_tolerance_cents,
            "auto (iterativa)",
            "peak_tolerance_cents",
        )
        add_change(
            "analysis.weights",
            "Pesos (B / decaimento / espectral)",
            f"{ba.weight_inharmonicity} / {ba.weight_decay} / {ba.weight_spectral}",
            f"{pa.weight_inharmonicity} / {pa.weight_decay} / {pa.weight_spectral}",
            "weights",
        )
        add_change(
            "analysis.min_confidence",
            "Confiança mínima",
            ba.min_confidence,
            pa.min_confidence,
            "min_confidence",
        )
        add_change(
            "analysis.min_margin",
            "Margem mínima (empate)",
            ba.min_margin,
            pa.min_margin,
            "min_margin",
        )

        if self.proposed.calibration.b0_by_string:
            b0_cur = f"{len(self.base_profile.calibration.b0_by_string)} cordas"
            n_cal = len(self.proposed.calibration.b0_by_string)
            sig_b = self.proposed.calibration.log_b_sigma
            b0_prop = f"{n_cal} cordas (σ={sig_b})"
            add_change("calibration.b0_by_string", "Calibração B0 por corda", b0_cur, b0_prop)

        return AutoTuneProposal(
            session_id=self.session_id,
            mode=self.mode,
            base_profile_name=self.base_profile.name,
            proposed_profile=self.proposed.model_copy(deep=True),
            changes=changes,
            warnings=list(dict.fromkeys(self.warnings)),
            step_statuses=dict(self.step_statuses),
        )
