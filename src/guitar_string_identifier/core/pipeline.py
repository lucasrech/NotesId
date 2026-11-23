"""Pipeline configurável ponta a ponta: AudioSource -> ... -> Result."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from guitar_string_identifier.analysis.complementary import ComplementaryFeatureExtractor
from guitar_string_identifier.analysis.inharmonicity import InharmonicityExtractor
from guitar_string_identifier.analysis.pitch import YinPitchEstimator
from guitar_string_identifier.analysis.segmentation import OnsetNoteSegmenter, WholeFileSegmenter
from guitar_string_identifier.autotune.adaptive_params import AdaptiveParams
from guitar_string_identifier.autotune.level_analyzer import LevelAnalyzer
from guitar_string_identifier.classifier.classical import GaussianStringClassifier
from guitar_string_identifier.core.interfaces import (
    AudioSource,
    LabelProvider,
    NoteSegmenter,
    PitchEstimator,
    StringClassifier,
)
from guitar_string_identifier.core.music import generate_candidates
from guitar_string_identifier.core.types import Features, Note, Result
from guitar_string_identifier.profiles.schema import GuitarProfile
from guitar_string_identifier.sources.file_source import FileAudioSource


class GuitarIdentifierPipeline:
    """Orquestra todas as etapas trocáveis a partir de um GuitarProfile."""

    def __init__(
        self,
        profile: GuitarProfile,
        segmenter: NoteSegmenter | None = None,
        pitch_estimator: PitchEstimator | None = None,
        inharmonicity_extractor: InharmonicityExtractor | None = None,
        complementary_extractor: ComplementaryFeatureExtractor | None = None,
        classifier: StringClassifier | None = None,
        label_provider: LabelProvider | None = None,
        whole_file: bool = True,
        reject_bad_input: bool = False,
    ) -> None:
        self.profile = profile
        self.reject_bad_input = reject_bad_input
        ac = profile.analysis
        self.adaptive = AdaptiveParams()
        self.level_analyzer = LevelAnalyzer()
        self.segmenter = segmenter or (
            WholeFileSegmenter() if whole_file else OnsetNoteSegmenter()
        )
        self.pitch_estimator = pitch_estimator or YinPitchEstimator(
            transient_ms=ac.transient_ms
        )
        self.inharmonicity_extractor = inharmonicity_extractor or InharmonicityExtractor(
            window_sec=ac.window_sec,
            transient_ms=ac.transient_ms,
            max_partials=ac.max_partials,
            peak_tolerance_cents=ac.peak_tolerance_cents,
            min_peak_snr_db=ac.min_peak_snr_db,
            hum_notch_freqs_hz=profile.capture.hum_notch_freqs_hz,
            iterative_narrowing=profile.is_auto("peak_tolerance_cents"),
        )
        self.complementary_extractor = (
            complementary_extractor
            or ComplementaryFeatureExtractor(transient_ms=ac.transient_ms)
        )
        self.classifier = classifier or GaussianStringClassifier(
            calibration=profile.calibration,
            analysis_config=ac,
        )
        self.label_provider = label_provider

    def analyze_note(self, note: Note, *, provisional: bool = False) -> Result:
        """Executa f0 -> candidatos -> características -> classificador em uma Note."""
        ac = self.profile.analysis
        ic = self.profile.instrument
        cc = self.profile.capture

        # Validação inicial de qualidade (silêncio / clipping) quando ativa
        if self.reject_bad_input:
            q_pre = self.level_analyzer.check_input_quality(
                note.audio,
                note.sr,
                f0=None,
                noise_floor_db=cc.noise_floor_db,
            )
            if not q_pre.ok:
                return Result(
                    string=None,
                    fret=None,
                    note="?",
                    confidence=0.0,
                    candidates=[],
                    f0=None,
                    indeterminate=True,
                    features=Features(),
                    input_warning=q_pre.warning,
                    provisional=provisional,
                )

        f0 = note.f0 if note.f0 is not None else self.pitch_estimator.estimate(note)
        if f0 is None or f0 <= 0:
            return Result(
                string=None,
                fret=None,
                note="?",
                confidence=0.0,
                candidates=[],
                f0=None,
                indeterminate=True,
                features=Features(),
                input_warning="Não foi possível estimar a frequência fundamental (f0).",
                provisional=provisional,
            )

        # Validação de acorde (múltiplas f0)
        if self.reject_bad_input:
            q_poly = self.level_analyzer.check_input_quality(
                note.audio,
                note.sr,
                f0=f0,
                noise_floor_db=cc.noise_floor_db,
            )
            if not q_poly.ok:
                return Result(
                    string=None,
                    fret=None,
                    note="?",
                    confidence=0.0,
                    candidates=[],
                    f0=f0,
                    indeterminate=True,
                    features=Features(),
                    input_warning=q_poly.warning,
                    provisional=provisional,
                )

        note.f0 = f0

        # Resolve parâmetros adaptativos (modo 'auto' vs 'manual')
        resolved = self.adaptive.resolve_for_note(
            audio=note.audio,
            sr=note.sr,
            f0=f0,
            param_modes=self.profile.param_modes,
            manual_window_sec=ac.window_sec,
            manual_transient_ms=ac.transient_ms,
            manual_max_partials=ac.max_partials,
            manual_peak_tol=ac.peak_tolerance_cents,
        )

        inh_feats = self.inharmonicity_extractor.extract(
            note,
            window_sec=resolved.window_sec,
            transient_ms=resolved.transient_ms,
            max_partials=resolved.max_partials,
            peak_tolerance_cents=resolved.peak_tolerance_cents,
            iterative_narrowing=resolved.iterative_narrowing,
        )
        f0_ref = float(inh_feats.get("f0_refined") or f0)
        note.f0 = f0_ref

        comp_feats = self.complementary_extractor.extract(
            note, partials=inh_feats.get("partials")
        )
        all_feats = Features(
            {
                **inh_feats,
                **comp_feats,
                "resolved_window_sec": resolved.window_sec,
                "resolved_transient_ms": resolved.transient_ms,
                "resolved_max_partials": resolved.max_partials,
                "resolved_peak_tol_cents": resolved.peak_tolerance_cents,
            }
        )

        candidates = generate_candidates(
            f0_ref,
            tuning=ic.tuning,
            num_frets=ic.num_frets,
            max_cents_error=ac.max_candidate_cents,
            a4_hz=ic.a4_hz,
            tuning_cents_offset=ic.tuning_cents_offset,
        )
        result = self.classifier.classify(note, all_feats, candidates)
        result.provisional = provisional
        if note.source_path and self.label_provider is not None:
            result.label = self.label_provider.get_label(Path(note.source_path))
        return result

    def analyze_audio(
        self,
        audio: np.ndarray,
        sr: int,
        source_path: str | None = None,
    ) -> list[Result]:
        """Segmenta o áudio e analisa todas as notas encontradas."""
        notes = self.segmenter.segment(audio, sr)
        results: list[Result] = []
        for note in notes:
            note.source_path = source_path
            results.append(self.analyze_note(note))
        return results

    def analyze_source(self, source: AudioSource) -> list[Result]:
        """Lê de uma AudioSource e executa o pipeline."""
        audio, sr = source.read()
        source_path = str(source.path) if isinstance(source, FileAudioSource) else None
        return self.analyze_audio(audio, sr, source_path=source_path)

    def analyze_file(self, path: Path | str) -> Result:
        """Atalho para analisar um arquivo de nota isolada."""
        source = FileAudioSource(path, channel=self.profile.capture.channel)
        results = self.analyze_source(source)
        if not results:
            label = (
                self.label_provider.get_label(Path(path))
                if self.label_provider is not None
                else None
            )
            return Result(
                string=None,
                fret=None,
                note="?",
                confidence=0.0,
                indeterminate=True,
                label=label,
            )
        return results[0]
