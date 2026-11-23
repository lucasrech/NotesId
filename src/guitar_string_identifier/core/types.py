"""Tipos de dados fundamentais do pipeline de identificação."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Note:
    """Representa um segmento de áudio contendo uma única nota isolada."""

    audio: np.ndarray
    sr: int
    t0: float = 0.0
    f0: float | None = None
    source_path: str | None = None


@dataclass
class PartialInfo:
    """Informações de um parcial harmônico detectado no espectro."""

    n: int
    ideal_freq: float
    measured_freq: float
    magnitude_db: float
    deviation_cents: float
    decay_rate: float | None = None


@dataclass
class Features:
    """Coleção de características físicas extraídas de uma nota."""

    values: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.values[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self.values


@dataclass
class Candidate:
    """Uma posição candidata (corda, casa) compatível com a frequência fundamental."""

    string: int
    fret: int
    note: str
    midi: int
    expected_f0: float
    score: float = 0.0
    distance: float = float("inf")
    expected_b: float | None = None


@dataclass
class GroundTruthLabel:
    """Rótulo de referência (ground truth) fornecido por um LabelProvider."""

    string: int
    fret: int
    note: str
    pickup: str | None = None
    effect: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """Resultado final da identificação de corda e casa."""

    string: int | None
    fret: int | None
    note: str
    confidence: float
    candidates: list[Candidate] = field(default_factory=list)
    f0: float | None = None
    indeterminate: bool = False
    features: Features | None = None
    label: GroundTruthLabel | None = None
