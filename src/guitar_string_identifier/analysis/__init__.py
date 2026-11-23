"""Pacote de análise de sinais do NotesId (segmentação, f0, inharmonicidade e decaimento).

Reúne todos os algoritmos clássicos de processamento digital de sinais (DSP):
- `segmentation`: encontra onde cada nota começa no áudio (ataque / onset).
- `pitch`: mede a frequência fundamental `f0` em Hz via algoritmo YIN.
- `inharmonicity`: mede os parciais via FFT longa e ajusta o coeficiente `B`.
- `complementary`: mede decaimento por parcial (STFT), posição de palhetada e brilho.

Glossário de referência: `docs/GLOSSARY.pt-BR.md`.
"""

from guitar_string_identifier.analysis.complementary import ComplementaryFeatureExtractor
from guitar_string_identifier.analysis.inharmonicity import InharmonicityExtractor
from guitar_string_identifier.analysis.pitch import YinPitchEstimator
from guitar_string_identifier.analysis.segmentation import OnsetNoteSegmenter, WholeFileSegmenter

__all__ = [
    "ComplementaryFeatureExtractor",
    "InharmonicityExtractor",
    "OnsetNoteSegmenter",
    "WholeFileSegmenter",
    "YinPitchEstimator",
]
