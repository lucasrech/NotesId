"""Contratos (interfaces abstratas) das etapas substituíveis do NotesId.

Analogia do dia a dia:
    Assim como tomadas elétricas padronizadas permitem plugar qualquer aparelho
    compatível na mesma parede, estas classes abstratas definem o "formato da
    tomada" de cada estágio do pipeline:

    `AudioSource → NoteSegmenter → PitchEstimator → FeatureExtractor → StringClassifier`

    Dessa forma, podemos trocar a fonte de áudio (arquivo, dataset, navegador ao
    vivo) ou o algoritmo de uma etapa sem precisar modificar as outras.

Glossário de referência: `docs/GLOSSARY.pt-BR.md`.
"""

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

__all__ = [
    "AudioSource",
    "FeatureExtractor",
    "LabelProvider",
    "NoteSegmenter",
    "PitchEstimator",
    "StringClassifier",
]


class AudioSource(ABC):
    """Contrato para qualquer fonte que forneça áudio (arquivo, dataset ou ao vivo)."""

    @abstractmethod
    def read(self) -> tuple[np.ndarray, int]:
        """Lê o sinal de áudio completo disponível na fonte.

        Returns:
            Tupla `(audio, sample_rate)`, onde `audio` é um vetor 1D mono em
            `float64` e `sample_rate` é a taxa de amostragem em Hz.
        """


class NoteSegmenter(ABC):
    """Contrato para algoritmos que recortam notas isoladas de um trecho de áudio."""

    @abstractmethod
    def segment(self, audio: np.ndarray, sr: int) -> list[Note]:
        """Divide o áudio em uma ou mais instâncias de `Note`.

        Args:
            audio: Sinal de áudio em `float64` (mono ou multicanal).
            sr: Taxa de amostragem do sinal em Hz.

        Returns:
            Lista de notas recortadas prontas para análise.
        """


class PitchEstimator(ABC):
    """Contrato para estimadores de frequência fundamental (f0) de uma nota."""

    @abstractmethod
    def estimate(self, note: Note) -> float | None:
        """Estima a frequência fundamental `f0` em Hz (ver glossário: f0).

        Args:
            note: Nota isolada contendo áudio e taxa de amostragem.

        Returns:
            Frequência fundamental em Hz, ou `None` se não houver altura clara.
        """


class FeatureExtractor(ABC):
    """Contrato para extratores de características físicas (B, decaimento, espectro)."""

    @abstractmethod
    def extract(self, note: Note) -> dict[str, Any]:
        """Extrai medições físicas a partir de uma nota já com `f0` conhecida.

        Args:
            note: Nota isolada com áudio, taxa de amostragem e (idealmente) `f0`.

        Returns:
            Dicionário com os valores extraídos (ex.: `{"B": 1.2e-4, ...}`).
        """


class StringClassifier(ABC):
    """Contrato para classificadores clássicos que escolhem entre candidatos `(corda, casa)`."""

    @abstractmethod
    def classify(
        self,
        note: Note,
        features: Features,
        candidates: list[Candidate],
    ) -> Result:
        """Ranqueia os candidatos compatíveis e decide a posição (ou `indeterminado`).

        Args:
            note: Nota analisada.
            features: Características físicas extraídas da nota.
            candidates: Lista de posições `(corda, casa)` possíveis para aquela `f0`.

        Returns:
            Objeto `Result` com a posição escolhida, confiança e ranking completo.
        """


class LabelProvider(ABC):
    """Contrato para leitura de gabarito (ground truth) de arquivos de um dataset."""

    @abstractmethod
    def get_label(self, path: Path) -> GroundTruthLabel | None:
        """Retorna o rótulo oficial se o arquivo seguir a convenção do dataset.

        Args:
            path: Caminho do arquivo de áudio.

        Returns:
            `GroundTruthLabel` com corda, casa e nota esperadas, ou `None` se o
            arquivo não pertencer ao dataset suportado.
        """
