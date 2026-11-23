"""Testes unitários para utilitários musicais e geração de candidatos."""

from guitar_string_identifier.core.music import (
    STANDARD_TUNING,
    freq_to_cents,
    generate_candidates,
    hz_to_midi,
    midi_to_hz,
    midi_to_note_name,
    note_name_to_midi,
    string_fret_to_note,
)


def test_note_midi_roundtrip() -> None:
    assert note_name_to_midi("A4") == 69
    assert note_name_to_midi("C4") == 60
    assert note_name_to_midi("E2") == 40
    assert note_name_to_midi("Bb3") == 58
    assert midi_to_note_name(60) == "C4"
    assert midi_to_note_name(69) == "A4"
    assert abs(midi_to_hz(69) - 440.0) < 1e-9
    assert abs(hz_to_midi(261.6255653) - 60.0) < 1e-4


def test_c4_five_positions_standard_tuning() -> None:
    c4_hz = midi_to_hz(60)
    candidates = generate_candidates(c4_hz, tuning=STANDARD_TUNING, num_frets=22)
    positions = [(c.string, c.fret) for c in candidates]
    assert positions == [(2, 1), (3, 5), (4, 10), (5, 15), (6, 20)]
    assert all(c.note == "C4" for c in candidates)


def test_string_fret_to_note() -> None:
    assert string_fret_to_note(6, 0) == "E2"
    assert string_fret_to_note(1, 0) == "E4"
    assert string_fret_to_note(1, 12) == "E5"
    assert string_fret_to_note(5, 3) == "C3"


def test_freq_to_cents() -> None:
    assert abs(freq_to_cents(440.0, 440.0)) < 1e-9
    assert abs(freq_to_cents(880.0, 440.0) - 1200.0) < 1e-9
