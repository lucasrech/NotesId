"""Utilitários musicais: conversão nota/MIDI/Hz e geração de candidatos (corda, casa)."""

from __future__ import annotations

import math
import re

from guitar_string_identifier.core.types import Candidate

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}

# Afinação padrão de guitarra 6 cordas: corda 1 (mais aguda) a corda 6 (mais grave)
STANDARD_TUNING: dict[int, str] = {
    1: "E4",
    2: "B3",
    3: "G3",
    4: "D3",
    5: "A2",
    6: "E2",
}

_NOTE_RE = re.compile(r"^([A-Ga-g])([#bB]?)(-?\d+)$")


def note_name_to_midi(name: str) -> int:
    """Converte nome de nota (ex: 'E2', 'C#4', 'Bb3') para número MIDI."""
    match = _NOTE_RE.match(name.strip())
    if not match:
        raise ValueError(f"Nome de nota inválido: {name!r}")
    letter, accidental, octave_str = match.groups()
    key = letter.upper() + accidental.upper()
    semitone = _NOTE_TO_SEMITONE[key]
    octave = int(octave_str)
    return (octave + 1) * 12 + semitone


def midi_to_note_name(midi: int) -> str:
    """Converte número MIDI para nome de nota (ex: 60 -> 'C4')."""
    octave = (midi // 12) - 1
    note = NOTE_NAMES[midi % 12]
    return f"{note}{octave}"


def midi_to_hz(midi: float, a4_hz: float = 440.0) -> float:
    """Converte número MIDI (inteiro ou fracionário) para frequência em Hz."""
    return a4_hz * (2.0 ** ((midi - 69.0) / 12.0))


def hz_to_midi(freq_hz: float, a4_hz: float = 440.0) -> float:
    """Converte frequência em Hz para número MIDI contínuo."""
    if freq_hz <= 0:
        raise ValueError("Frequência deve ser positiva.")
    return 69.0 + 12.0 * math.log2(freq_hz / a4_hz)


def freq_to_cents(f1: float, f2: float) -> float:
    """Retorna diferença em cents de f1 em relação a f2: 1200 * log2(f1 / f2)."""
    if f1 <= 0 or f2 <= 0:
        return float("inf")
    return 1200.0 * math.log2(f1 / f2)


def string_fret_to_midi(string: int, fret: int, tuning: dict[int, str] | None = None) -> int:
    """Calcula o número MIDI de uma posição (corda, casa)."""
    active_tuning = tuning or STANDARD_TUNING
    if string not in active_tuning:
        raise ValueError(f"Corda {string} ausente na afinação.")
    return note_name_to_midi(active_tuning[string]) + fret


def string_fret_to_note(string: int, fret: int, tuning: dict[int, str] | None = None) -> str:
    """Retorna o nome da nota (ex: 'C4') para (corda, casa)."""
    return midi_to_note_name(string_fret_to_midi(string, fret, tuning))


def generate_candidates(
    f0: float,
    tuning: dict[int, str] | None = None,
    num_frets: int = 22,
    max_cents_error: float = 55.0,
    a4_hz: float = 440.0,
) -> list[Candidate]:
    """Lista todos os pares (corda, casa) compatíveis com a f0 medida."""
    if f0 <= 0:
        return []

    active_tuning = tuning or STANDARD_TUNING
    midi_float = hz_to_midi(f0, a4_hz=a4_hz)
    target_midi = round(midi_float)
    cents_from_nearest = abs(midi_float - target_midi) * 100.0
    if cents_from_nearest > max_cents_error:
        return []

    note_name = midi_to_note_name(target_midi)
    expected_f0 = midi_to_hz(target_midi, a4_hz=a4_hz)

    candidates: list[Candidate] = []
    for string_num in sorted(active_tuning.keys()):
        open_midi = note_name_to_midi(active_tuning[string_num])
        fret = target_midi - open_midi
        if 0 <= fret <= num_frets:
            candidates.append(
                Candidate(
                    string=string_num,
                    fret=fret,
                    note=note_name,
                    midi=target_midi,
                    expected_f0=expected_f0,
                )
            )
    return candidates
