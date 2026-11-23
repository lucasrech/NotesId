"""Gerador sintético de corda rígida (Karplus-Strong com dispersão all-pass).

Permite criar notas artificiais com `f0` (Hz) e inharmonicidade `B` exatamente
conhecidos para validar os algoritmos de análise antes de testar em áudio real.
"""

from guitar_string_identifier.synth.stiff_string import (
    SynthParams,
    predict_b_from_open,
    synthesize_from_guitar_position,
    synthesize_stiff_string,
)

__all__ = [
    "SynthParams",
    "predict_b_from_open",
    "synthesize_from_guitar_position",
    "synthesize_stiff_string",
]
