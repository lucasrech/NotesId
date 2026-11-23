"""Interfaces (contratos) substituíveis da arquitetura de processamento."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from guitar_string_identifier.core.types import (
    Candidate,
    Features,
    GroundTruthLabel,
    Note,
    Result,
)


class AudioSource(ABC):
    """Fonte de áudio (arquivo, dataset, captura ao vivo)."""

    @abstractmethod
    def read(self) -> tuple[np.ndarray, int]:
        """Retorna (audio_mono_float64, sample_rate)."""


class NoteSegmenter(ABC):
    """Segmenta um sinal de áudio em notas isoladas."""

    @abstractmethod
    def segment(self, audio: np.ndarray, sr: int) -> list[Note]:
        """Extrai uma ou mais instâncias de Note do sinal."""


class PitchEstimator(ABC):
    """Estima a frequência fundamental (f0) de uma nota."""

    @abstractmethod
    def estimate(self, note: Note) -> float | None:
        """Retorna f0 em Hz ou None se não for possível estimar."""


class FeatureExtractor(ABC):
    """Extrai um conjunto de características físicas de uma nota."""

    @abstractmethod
    def extract(self, note: Note) -> dict[str, Any]:
        """Retorna um dicionário com as características extraídas."""


class StringClassifier(ABC):
    """Classifica entre os candidatos (corda, casa) usando características físicas."""

    @abstractmethod
    def classify(
        self,
        note: Note,
        features: Features,
        candidates: list[Candidate],
    ) -> Result:
        """Escolhe o melhor candidato (ou indeterminado) e retorna Result."""


class LabelProvider(ABC):
    """Fornece rótulos de referência (ground truth) para arquivos de um dataset."""

    @abstractmethod
    def get_label(self, path: Path) -> GroundTruthLabel | None:
        """Retorna GroundTruthLabel se o arquivo pertencer ao dataset conhecido."""
