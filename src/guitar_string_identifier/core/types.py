"""Tipos de dados fundamentais que trafegam pelo pipeline do NotesId.

Analogia do dia a dia:
    Pense no pipeline como uma linha de montagem de um laboratório de acústica.
    Cada objeto definido aqui é uma "prancheta padronizada" que passa de uma
    bancada para a próxima: primeiro entra o áudio recortado (`Note`), depois
    são anotadas as medições físicas (`Features` e `PartialInfo`), em seguida
    as posições possíveis no braço da guitarra (`Candidate`) e, por fim, o
    veredito final (`Result`).

Glossário de referência: `docs/GLOSSARY.pt-BR.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "Candidate",
    "Features",
    "GroundTruthLabel",
    "Note",
    "PartialInfo",
    "Result",
]


@dataclass
class Note:
    """Representa um trecho de áudio contendo uma única nota isolada de guitarra.

    Attributes:
        audio: Vetor 1D (`float64`, mono) com as amostras de amplitude do som.
        sr: Taxa de amostragem em Hz (quantas amostras por segundo; ex.: 48000 Hz).
        t0: Instante inicial da nota no arquivo ou transmissão original, em segundos (s).
        f0: Frequência fundamental estimada em Hz (ver glossário: f0), ou `None`
            antes da etapa de detecção de altura.
        source_path: Caminho do arquivo de origem, quando a nota veio de disco.
    """

    audio: np.ndarray
    sr: int
    t0: float = 0.0
    f0: float | None = None
    source_path: str | None = None


@dataclass
class PartialInfo:
    """Medições de um único parcial (sobretom) detectado no espectro da nota.

    Em uma corda real rígida, o parcial de ordem `n` vibra ligeiramente acima
    do harmônico ideal `n * f0` devido à rigidez do aço (inharmonicidade B).

    Attributes:
        n: Ordem do parcial (1 = fundamental, 2 = oitava, 3 = quinta acima, etc.).
        ideal_freq: Frequência esperada em uma corda ideal sem rigidez (`n * f0`), em Hz.
        measured_freq: Frequência real medida no pico espectral, em Hz.
        magnitude_db: Intensidade relativa do pico em decibéis (dB), sendo 0 dB o pico mais forte.
        deviation_cents: Diferença entre `measured_freq` e `ideal_freq`, em cents
            (100 cents = 1 semitom; ver glossário: cents).
        decay_rate: Velocidade com que este parcial enfraquece ao longo do tempo,
            em dB/s (opcional).
    """

    n: int
    ideal_freq: float
    measured_freq: float
    magnitude_db: float
    deviation_cents: float
    decay_rate: float | None = None


@dataclass
class Features:
    """Coleção flexível das características físicas extraídas de uma nota.

    Armazena tanto o coeficiente principal de inharmonicidade `B` quanto métricas
    complementares (decaimento, posição de palhetada, centroide e espectro para gráfico).

    Attributes:
        values: Dicionário chave → valor contendo todas as medições extraídas.
    """

    values: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Retorna o valor de uma característica ou `default` se ausente.

        Args:
            key: Nome da característica (ex.: `"B"`, `"decay_mean_db_s"`).
            default: Valor retornado caso a chave não exista.

        Returns:
            Valor armazenado ou `default`.
        """
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.values[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self.values


@dataclass
class Candidate:
    """Uma posição candidata `(corda, casa)` capaz de produzir a mesma nota medida.

    Por exemplo, para Dó4 (~261,63 Hz) na afinação padrão, o sistema gera 5
    candidatos: `(2, 1)`, `(3, 5)`, `(4, 10)`, `(5, 15)` e `(6, 20)`.

    Attributes:
        string: Número da corda (1 = mais fina/aguda E4 até 6 = mais grossa/grave E2).
        fret: Número da casa no braço (0 = corda solta, 1..22 = casas pressionadas).
        note: Nome musical da nota (ex.: `"C4"`).
        midi: Código numérico MIDI da nota (ex.: 60 = C4, 69 = A4).
        expected_f0: Frequência fundamental teórica dessa posição, em Hz.
        score: Pontuação/probabilidade relativa atribuída pelo classificador (0,0 a 1,0).
        distance: Distância estatística de Mahalanobis (menor = mais compatível).
        expected_b: Coeficiente de inharmonicidade B previsto pela calibração para esta posição.
    """

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
    """Rótulo oficial (gabarito) fornecido por um dataset como o EGFxSet.

    Usado apenas para conferência de acerto na interface e na avaliação —
    nunca é entregue ao classificador durante a inferência.

    Attributes:
        string: Corda real tocada (1 a 6).
        fret: Casa real tocada (0 a 22).
        note: Nome da nota inferido a partir de `(string, fret)` e da afinação.
        pickup: Posição da chave de captadores (ex.: `"Bridge"`, `"Neck"`), se conhecida.
        effect: Efeito aplicado na gravação (ex.: `"Clean"`), se conhecido.
        extra: Metadados adicionais específicos do dataset.
    """

    string: int
    fret: int
    note: str
    pickup: str | None = None
    effect: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """Resultado final emitido pelo pipeline de identificação de uma nota.

    Attributes:
        string: Corda identificada (1 a 6), ou `None` se o sistema se absteve (`indeterminate`).
        fret: Casa identificada (0 a 22), ou `None` se `indeterminate` for `True`.
        note: Nome da nota identificada (ex.: `"C4"`, ou `"?"` se f0 falhou).
        confidence: Grau de confiança na decisão (0,0 a 1,0).
        candidates: Lista de todos os candidatos avaliados, ordenados do melhor ao pior.
        f0: Frequência fundamental refinada em Hz, ou `None`.
        indeterminate: `True` quando houve empate físico ou confiança abaixo do limiar.
        features: Características físicas medidas na nota.
        label: Gabarito opcional para conferência (quando a nota veio de dataset rotulado).
        input_warning: Aviso de entrada rejeitada (ex.: acorde ou clipping severo).
        provisional: `True` durante atualizações progressivas em tempo real (~0,3 s / ~0,6 s).
    """

    string: int | None
    fret: int | None
    note: str
    confidence: float
    candidates: list[Candidate] = field(default_factory=list)
    f0: float | None = None
    indeterminate: bool = False
    features: Features | None = None
    label: GroundTruthLabel | None = None
    input_warning: str | None = None
    provisional: bool = False
