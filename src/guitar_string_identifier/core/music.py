"""Utilitários musicais do NotesId: conversões nota ↔ MIDI ↔ Hz e candidatos.

Analogia do dia a dia:
    Na música ocidental, cada tecla do piano tem um nome (`C4`, `A4`), um número
    de série (`MIDI`: 60, 69) e uma velocidade de vibração em Hertz (`261,63 Hz`,
    `440 Hz`). Este módulo funciona como um "tradutor universal" entre esses três
    idiomas e como um "mapa do braço da guitarra" que responde: *dada esta
    frequência em Hz, em quais pares `(corda, casa)` essa nota existe?*

Glossário de referência: `docs/GLOSSARY.pt-BR.md` (nota, oitava, cents, afinação).
"""

from __future__ import annotations

import math
import re

from guitar_string_identifier.core.types import Candidate

__all__ = [
    "NOTE_NAMES",
    "STANDARD_TUNING",
    "freq_to_cents",
    "generate_candidates",
    "hz_to_midi",
    "midi_to_hz",
    "midi_to_note_name",
    "note_name_to_midi",
    "string_fret_to_midi",
    "string_fret_to_note",
]

# As 12 notas da escala cromática ocidental (semitons de 0 a 11 dentro de cada oitava)
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Mapa de equivalência incluindo bemóis (ex.: Db = C#, Bb = A#)
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

# Afinação padrão de guitarra 6 cordas:
# corda 1 = mais fina/aguda (E4 ≈ 329,63 Hz) até corda 6 = mais grossa/grave (E2 ≈ 82,41 Hz)
STANDARD_TUNING: dict[int, str] = {
    1: "E4",
    2: "B3",
    3: "G3",
    4: "D3",
    5: "A2",
    6: "E2",
}

# Número MIDI do Lá central (A4 = 440 Hz por padrão)
_MIDI_A4 = 69.0
# Quantidade de semitons em uma oitava e cents em um semitom
_SEMITONES_PER_OCTAVE = 12.0
_CENTS_PER_OCTAVE = 1200.0

_NOTE_RE = re.compile(r"^([A-Ga-g])([#bB]?)(-?\d+)$")


def note_name_to_midi(name: str) -> int:
    """Converte o nome de uma nota (ex.: `'C4'`, `'Bb3'`) em número MIDI inteiro.

    Args:
        name: Nome musical com letra, acidente opcional (`#` ou `b`) e oitava.

    Returns:
        Número MIDI inteiro (ex.: `60` para `'C4'`, `69` para `'A4'`).

    Raises:
        ValueError: Se o texto não seguir o formato de nota musical válido.

    Examples:
        >>> note_name_to_midi("A4")
        69
        >>> note_name_to_midi("C4")
        60
    """
    match = _NOTE_RE.match(name.strip())
    if not match:
        raise ValueError(f"Nome de nota inválido: {name!r}")
    letter, accidental, octave_str = match.groups()
    key = letter.upper() + accidental.upper()
    semitone = _NOTE_TO_SEMITONE[key]
    octave = int(octave_str)
    # Na convenção MIDI, C-1 = 0, C0 = 12, C4 = 60
    return (octave + 1) * 12 + semitone


def midi_to_note_name(midi: int) -> str:
    """Converte um número MIDI inteiro no nome musical correspondente.

    Args:
        midi: Código MIDI inteiro (ex.: `60`).

    Returns:
        Nome musical usando sustenidos quando necessário (ex.: `'C4'`, `'F#3'`).

    Examples:
        >>> midi_to_note_name(60)
        'C4'
    """
    octave = (midi // 12) - 1
    note = NOTE_NAMES[midi % 12]
    return f"{note}{octave}"


def midi_to_hz(midi: float, a4_hz: float = 440.0) -> float:
    """Converte número MIDI (inteiro ou fracionário) para frequência em Hertz (Hz).

    Usa o temperamento igual: cada semitom multiplica a frequência por `2^(1/12)`.

    Args:
        midi: Número MIDI da nota (69 = A4).
        a4_hz: Frequência de referência do Lá4 em Hz (padrão internacional: 440,0 Hz).

    Returns:
        Frequência em Hz.

    Examples:
        >>> round(midi_to_hz(69), 2)
        440.0
    """
    return a4_hz * (2.0 ** ((midi - _MIDI_A4) / _SEMITONES_PER_OCTAVE))


def hz_to_midi(freq_hz: float, a4_hz: float = 440.0) -> float:
    """Converte frequência em Hz para número MIDI contínuo (pode ter fração).

    Args:
        freq_hz: Frequência positiva em Hz.
        a4_hz: Frequência de referência do Lá4 em Hz (padrão: 440,0 Hz).

    Returns:
        Número MIDI contínuo (ex.: `60.0` para 261,63 Hz; `60.5` = 50 cents acima de C4).

    Raises:
        ValueError: Se `freq_hz` for menor ou igual a zero.
    """
    if freq_hz <= 0:
        raise ValueError("Frequência deve ser positiva.")
    return _MIDI_A4 + _SEMITONES_PER_OCTAVE * math.log2(freq_hz / a4_hz)


def freq_to_cents(f1: float, f2: float) -> float:
    """Calcula a distância musical entre duas frequências, em cents.

    100 cents equivalem a 1 semitom e 1200 cents equivalem a 1 oitava.
    Valores positivos indicam que `f1` está mais aguda que `f2`.

    Args:
        f1: Frequência medida em Hz.
        f2: Frequência de referência em Hz.

    Returns:
        Desvio em cents (`1200 * log2(f1 / f2)`), ou `inf` se alguma for ≤ 0.
    """
    if f1 <= 0 or f2 <= 0:
        return float("inf")
    return _CENTS_PER_OCTAVE * math.log2(f1 / f2)


def string_fret_to_midi(string: int, fret: int, tuning: dict[int, str] | None = None) -> int:
    """Calcula o número MIDI produzido ao apertar `(corda, casa)` numa afinação.

    Args:
        string: Número da corda (1 a 6).
        fret: Número da casa (0 = solta).
        tuning: Mapa `{corda: nota_solta}`; usa `STANDARD_TUNING` se omitido.

    Returns:
        Número MIDI resultante (`midi_corda_solta + fret`).

    Raises:
        ValueError: Se a corda não existir na afinação fornecida.
    """
    active_tuning = tuning or STANDARD_TUNING
    if string not in active_tuning:
        raise ValueError(f"Corda {string} ausente na afinação.")
    return note_name_to_midi(active_tuning[string]) + fret


def string_fret_to_note(string: int, fret: int, tuning: dict[int, str] | None = None) -> str:
    """Retorna o nome da nota (ex.: `'C4'`) para uma posição `(corda, casa)`.

    Args:
        string: Número da corda (1 a 6).
        fret: Número da casa (0 a 22).
        tuning: Afinação da guitarra; usa `STANDARD_TUNING` se omitido.

    Returns:
        Nome musical da nota (ex.: `'C4'`).
    """
    return midi_to_note_name(string_fret_to_midi(string, fret, tuning))


def generate_candidates(
    f0: float,
    tuning: dict[int, str] | None = None,
    num_frets: int = 22,
    max_cents_error: float = 55.0,
    a4_hz: float = 440.0,
    tuning_cents_offset: dict[int, float] | None = None,
) -> list[Candidate]:
    """Lista todas as posições `(corda, casa)` da guitarra compatíveis com a `f0` medida.

    Em vez de tentar adivinhar entre todas as 138 casas do braço, esta função
    filtra apenas os 1 a 5 lugares onde aquela frequência fisicamente existe.
    Considera também a referência global `a4_hz` e o desvio real de cada corda
    (`tuning_cents_offset`), tolerando cordas levemente desafinadas.

    Args:
        f0: Frequência fundamental medida em Hz.
        tuning: Afinação por corda; usa `STANDARD_TUNING` se omitido.
        num_frets: Número total de casas do instrumento (ex.: 22).
        max_cents_error: Tolerância máxima em cents entre a `f0` medida e a casa
            mais próxima (padrão: 55 cents, pouco mais de meio semitom).
        a4_hz: Frequência de referência do Lá4 em Hz (padrão: 440,0 Hz).
        tuning_cents_offset: Desvio real medido em cents por corda solta.

    Returns:
        Lista de objetos `Candidate` ordenados da corda 1 para a corda 6.

    Examples:
        >>> [(c.string, c.fret) for c in generate_candidates(261.63)]
        [(2, 1), (3, 5), (4, 10), (5, 15), (6, 20)]
    """
    if f0 <= 0:
        return []

    active_tuning = tuning or STANDARD_TUNING
    offsets = tuning_cents_offset or {}

    # Passo 1: converte a frequência medida em Hz para a escala contínua MIDI
    midi_float = hz_to_midi(f0, a4_hz=a4_hz)

    # Passo 2: para cada corda, verifica qual casa inteira fica mais próxima de midi_float
    candidates: list[Candidate] = []
    for string_num in sorted(active_tuning.keys()):
        open_midi = note_name_to_midi(active_tuning[string_num])
        # Desloca a referência da corda pelo desvio real medido na calibração (se houver)
        offset_semitones = offsets.get(string_num, 0.0) / 100.0
        fret_float = midi_float - (open_midi + offset_semitones)
        fret = round(fret_float)
        cents_err = abs(fret_float - fret) * 100.0

        # Passo 3: aceita o candidato se a casa existe no braço e o erro está na tolerância
        if 0 <= fret <= num_frets and cents_err <= max_cents_error:
            target_midi = open_midi + fret
            expected_f0 = midi_to_hz(target_midi + offset_semitones, a4_hz=a4_hz)
            candidates.append(
                Candidate(
                    string=string_num,
                    fret=fret,
                    note=midi_to_note_name(target_midi),
                    midi=target_midi,
                    expected_f0=expected_f0,
                )
            )
    return candidates
