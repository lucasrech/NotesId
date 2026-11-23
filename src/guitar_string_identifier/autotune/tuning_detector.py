"""Detecção automática de afinação das cordas, afinações alternativas e referência A4."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from guitar_string_identifier.analysis.inharmonicity import InharmonicityExtractor
from guitar_string_identifier.analysis.pitch import YinPitchEstimator
from guitar_string_identifier.core.music import (
    freq_to_cents,
    hz_to_midi,
    midi_to_hz,
    midi_to_note_name,
    note_name_to_midi,
)
from guitar_string_identifier.core.types import Note

KNOWN_TUNING_PRESETS: dict[str, dict[int, str]] = {
    "Padrão (E-A-D-G-B-E)": {1: "E4", 2: "B3", 3: "G3", 4: "D3", 5: "A2", 6: "E2"},
    "Drop D (D-A-D-G-B-E)": {1: "E4", 2: "B3", 3: "G3", 4: "D3", 5: "A2", 6: "D2"},
    "Meio tom abaixo (Eb)": {1: "D#4", 2: "A#3", 3: "F#3", 4: "C#3", 5: "G#2", 6: "D#2"},
    "Um tom abaixo (D padrão)": {1: "D4", 2: "A3", 3: "F3", 4: "C3", 5: "G2", 6: "D2"},
    "Drop C# (Eb + Drop)": {1: "D#4", 2: "A#3", 3: "F#3", 4: "C#3", 5: "G#2", 6: "C#2"},
    "Drop C": {1: "D4", 2: "A3", 3: "F3", 4: "C3", 5: "G2", 6: "C2"},
    "DADGAD": {1: "D4", 2: "A3", 3: "G3", 4: "D3", 5: "A2", 6: "D2"},
    "Open G (D-G-D-G-B-D)": {1: "D4", 2: "B3", 3: "G3", 4: "D3", 5: "G2", 6: "D2"},
    "Open D (D-A-D-F#-A-D)": {1: "D4", 2: "A3", 3: "F#3", 4: "D3", 5: "A2", 6: "D2"},
}


@dataclass
class StringTuningMeasurement:
    """Medição da afinação de uma única corda solta."""

    string: int
    f0_hz: float
    nearest_note: str
    midi: int
    cents_from_440: float
    accepted: bool
    reason: str = "Aceita."


@dataclass
class TuningSummary:
    """Resumo consolidado da afinação das 6 cordas soltas."""

    tuning: dict[int, str]
    cents_offset_by_string: dict[int, float]
    a4_hz: float
    global_cents_offset: float
    preset_name: str
    is_alternate_tuning: bool
    needs_confirmation: bool
    measurements: list[StringTuningMeasurement] = field(default_factory=list)


class TuningDetector:
    """Detecta f0 de cada corda solta, afinação alternativa e referência global A4."""

    def __init__(self, target_tolerance_cents: float = 48.0) -> None:
        self.target_tolerance_cents = target_tolerance_cents
        self.pitch_estimator = YinPitchEstimator()
        self.inharm_extractor = InharmonicityExtractor(window_sec=0.5, max_partials=8)

    def measure_open_string(
        self,
        audio: np.ndarray,
        sr: int,
        string: int,
        expected_note: str | None = None,
    ) -> StringTuningMeasurement | None:
        """Mede f0 refinada de uma corda solta e calcula a nota mais próxima."""
        note = Note(audio=np.asarray(audio, dtype=np.float64), sr=sr)
        f0 = self.pitch_estimator.estimate(note)
        if f0 is None or f0 <= 50.0:
            return None

        note.f0 = f0
        inh = self.inharm_extractor.extract(note)
        f0_ref = float(inh.get("f0_refined") or f0)

        midi_float = hz_to_midi(f0_ref, a4_hz=440.0)
        nearest_midi = round(midi_float)
        nearest_note = midi_to_note_name(nearest_midi)
        cents_440 = (midi_float - nearest_midi) * 100.0

        accepted = True
        reason = "Aceita."
        if expected_note is not None:
            exp_midi = note_name_to_midi(expected_note)
            # Aceita até ±250 cents da nota esperada para suportar Drop D / afinações alternativas
            cents_from_exp = freq_to_cents(f0_ref, midi_to_hz(exp_midi, a4_hz=440.0))
            if abs(cents_from_exp) > 260.0:
                accepted = False
                reason = (
                    f"Nota medida ({nearest_note}, {f0_ref:.1f} Hz) muito distante "
                    f"da corda {string} esperada ({expected_note})."
                )
        elif abs(cents_440) > self.target_tolerance_cents:
            accepted = False
            reason = f"Desvio de {cents_440:+.1f} cents excede tolerância."

        return StringTuningMeasurement(
            string=string,
            f0_hz=round(f0_ref, 3),
            nearest_note=nearest_note,
            midi=nearest_midi,
            cents_from_440=round(cents_440, 2),
            accepted=accepted,
            reason=reason,
        )

    def summarize(
        self,
        measurements: list[StringTuningMeasurement],
        estimate_global_a4: bool = True,
    ) -> TuningSummary:
        """Consolida medições das cordas soltas, estima A4 global e identifica preset."""
        valid = [m for m in measurements if m.accepted]
        if not valid:
            return TuningSummary(
                tuning=dict(KNOWN_TUNING_PRESETS["Padrão (E-A-D-G-B-E)"]),
                cents_offset_by_string={},
                a4_hz=440.0,
                global_cents_offset=0.0,
                preset_name="Padrão (E-A-D-G-B-E)",
                is_alternate_tuning=False,
                needs_confirmation=False,
                measurements=measurements,
            )

        # Desvio global de referência (mediana dos desvios em cents em relação a A4=440)
        cents_list = [m.cents_from_440 for m in valid]
        median_cents = float(np.median(cents_list))
        # Só desloca A4 se houver consistência global (>= 3 cordas e mediana > 3 cents)
        if estimate_global_a4 and len(valid) >= 3 and abs(median_cents) >= 3.0:
            a4_hz = round(440.0 * (2.0 ** (median_cents / 1200.0)), 2)
            global_cents = round(median_cents, 2)
        else:
            a4_hz = 440.0
            global_cents = 0.0

        tuning: dict[int, str] = {}
        offsets: dict[int, float] = {}
        for m in sorted(valid, key=lambda item: item.string):
            tuning[m.string] = m.nearest_note
            # Desvio residual em relação ao A4 estimado
            ref_hz = midi_to_hz(m.midi, a4_hz=a4_hz)
            offsets[m.string] = round(freq_to_cents(m.f0_hz, ref_hz), 2)

        preset_name = self.match_preset(tuning)
        is_alt = preset_name != "Padrão (E-A-D-G-B-E)"
        needs_conf = is_alt or abs(a4_hz - 440.0) >= 2.0

        return TuningSummary(
            tuning=tuning,
            cents_offset_by_string=offsets,
            a4_hz=a4_hz,
            global_cents_offset=global_cents,
            preset_name=preset_name,
            is_alternate_tuning=is_alt,
            needs_confirmation=needs_conf,
            measurements=measurements,
        )

    @staticmethod
    def match_preset(tuning: dict[int, str]) -> str:
        """Compara a afinação detectada com presets conhecidos de guitarra."""
        if not tuning:
            return "Personalizada"
        for name, preset in KNOWN_TUNING_PRESETS.items():
            if all(
                note_name_to_midi(tuning[s]) == note_name_to_midi(preset[s])
                for s in tuning
                if s in preset
            ):
                return name
        return "Personalizada"
