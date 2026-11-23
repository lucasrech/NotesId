"""Núcleo arquitetural do NotesId: tipos de dados, contratos, música e pipeline.

Este pacote reúne as peças centrais que conectam todo o sistema:
- `types`: as "fichas" de dados que trafegam entre as etapas (nota, características, resultado).
- `interfaces`: os contratos abstratos que permitem trocar cada etapa sem reescrever o resto.
- `music`: conversões musicais (Hz ↔ nota ↔ MIDI) e geração de candidatos (corda, casa).
- `pipeline`: o maestro que executa as etapas na ordem certa.

Consulte `docs/GLOSSARY.pt-BR.md` para explicações simples de termos como f0, cents e parciais.
"""

from guitar_string_identifier.core.pipeline import GuitarIdentifierPipeline
from guitar_string_identifier.core.types import (
    Candidate,
    Features,
    GroundTruthLabel,
    Note,
    PartialInfo,
    Result,
)

__all__ = [
    "Candidate",
    "Features",
    "GroundTruthLabel",
    "GuitarIdentifierPipeline",
    "Note",
    "PartialInfo",
    "Result",
]
