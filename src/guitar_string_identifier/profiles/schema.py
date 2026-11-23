"""Esquema Pydantic para perfis de configuração (captação, instrumento, análise e calibração)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from guitar_string_identifier.core.music import STANDARD_TUNING, note_name_to_midi


class CaptureConfig(BaseModel):
    """Bloco 1: parâmetros de captação de áudio."""

    source_mode: Literal["file", "dataset", "live_browser", "live_server"] = Field(
        default="dataset",
        description="Modo padrão de entrada de áudio.",
    )
    device_name: str | None = Field(
        default=None,
        description="Nome ou identificador do dispositivo de captura (opcional).",
    )
    sample_rate: int = Field(
        default=48000,
        ge=8000,
        le=192000,
        description="Taxa de amostragem esperada em Hz.",
    )
    channel: int = Field(
        default=0,
        ge=0,
        le=32,
        description="Índice do canal de entrada (0 = primeiro canal / mono).",
    )
    reference_gain_db: float = Field(
        default=0.0,
        ge=-60.0,
        le=60.0,
        description="Ganho de referência da interface de áudio em dB.",
    )
    buffer_size: int = Field(
        default=4096,
        ge=256,
        le=65536,
        description="Tamanho do bloco de captura em amostras.",
    )


class StringSpec(BaseModel):
    """Especificação opcional de uma corda (campos None = 'a estimar')."""

    string_type: Literal["plain", "wound"] | None = Field(
        default=None,
        description="Tipo da corda: lisa ('plain'), encapada ('wound') ou None (a estimar).",
    )
    gauge_inches: float | None = Field(
        default=None,
        gt=0.005,
        lt=0.100,
        description="Calibre em polegadas (ex: 0.009) ou None (a estimar).",
    )


class InstrumentConfig(BaseModel):
    """Bloco 2: parâmetros físicos e musicais do instrumento."""

    model_name: str = Field(
        default="Guitarra genérica",
        description="Nome ou modelo do instrumento.",
    )
    tuning: dict[int, str] = Field(
        default_factory=lambda: dict(STANDARD_TUNING),
        description="Afinação por corda (1 = mais aguda, 6 = mais grave).",
    )
    num_frets: int = Field(
        default=22,
        ge=12,
        le=27,
        description="Número de casas do instrumento.",
    )
    scale_length_mm: float = Field(
        default=648.0,
        ge=500.0,
        le=800.0,
        description="Comprimento de escala (nut à ponte) em mm.",
    )
    strings: dict[int, StringSpec] = Field(
        default_factory=lambda: {s: StringSpec() for s in range(1, 7)},
        description="Tipo e calibre por corda (opcional; None = a estimar).",
    )
    pickup_config: str | None = Field(
        default=None,
        description="Configuração de captador (ex: Bridge, Neck, Middle) ou None.",
    )

    @field_validator("tuning")
    @classmethod
    def validate_tuning(cls, value: dict[int, str]) -> dict[int, str]:
        if not value:
            raise ValueError("A afinação deve conter ao menos uma corda.")
        cleaned: dict[int, str] = {}
        for k, note_name in value.items():
            s = int(k)
            note_name_to_midi(note_name)
            cleaned[s] = note_name.strip()
        return cleaned


class AnalysisConfig(BaseModel):
    """Bloco 3: parâmetros de processamento de sinais e classificação."""

    window_sec: float = Field(
        default=0.6,
        ge=0.2,
        le=3.0,
        description="Duração da janela de FFT longa em segundos (>= 0,5 s recomendado).",
    )
    transient_ms: float = Field(
        default=50.0,
        ge=0.0,
        le=300.0,
        description="Duração do transiente inicial descartado em ms.",
    )
    max_partials: int = Field(
        default=12,
        ge=4,
        le=25,
        description="Número máximo de parciais harmônicos rastreados.",
    )
    peak_tolerance_cents: float = Field(
        default=40.0,
        ge=5.0,
        le=120.0,
        description="Tolerância de busca em torno de cada parcial previsto (cents).",
    )
    min_peak_snr_db: float = Field(
        default=12.0,
        ge=3.0,
        le=40.0,
        description="SNR mínimo local para aceitar um pico espectral (dB).",
    )
    max_candidate_cents: float = Field(
        default=55.0,
        ge=10.0,
        le=100.0,
        description="Tolerância máxima de f0 para geração de candidatos (cents).",
    )
    min_confidence: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Confiança mínima para emitir uma decisão (abaixo = indeterminado).",
    )
    min_margin: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Diferença mínima entre 1º e 2º candidato para evitar empate.",
    )
    weight_inharmonicity: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Peso do coeficiente de inharmonicidade B no classificador.",
    )
    weight_decay: float = Field(
        default=0.25,
        ge=0.0,
        le=5.0,
        description="Peso da taxa de decaimento por parcial.",
    )
    weight_spectral: float = Field(
        default=0.05,
        ge=0.0,
        le=2.0,
        description="Peso de apoio para centroide/roll-off espectral.",
    )


class CalibrationData(BaseModel):
    """Parâmetros estimados na calibração da guitarra (por corda)."""

    b0_by_string: dict[int, float] = Field(
        default_factory=dict,
        description="Constante de inharmonicidade B0 da corda solta (casa 0) por corda.",
    )
    log_b_sigma: float = Field(
        default=0.22,
        gt=0.01,
        le=2.0,
        description="Desvio padrão empírico de log(B) em torno do modelo físico.",
    )
    decay_intercept_by_string: dict[int, float] = Field(
        default_factory=dict,
        description="Intercepto da taxa de decaimento média (dB/s) na casa 0 por corda.",
    )
    decay_slope_by_string: dict[int, float] = Field(
        default_factory=dict,
        description="Inclinação da taxa de decaimento média (dB/s por casa) por corda.",
    )
    decay_sigma: float = Field(
        default=3.5,
        gt=0.1,
        le=50.0,
        description="Desvio padrão da taxa de decaimento média (dB/s).",
    )
    centroid_ratio_by_string: dict[int, float] = Field(
        default_factory=dict,
        description="Razão média centroide/f0 por corda (apoio de baixo peso).",
    )
    centroid_sigma: float = Field(
        default=2.0,
        gt=0.1,
        le=20.0,
        description="Desvio padrão da razão centroide/f0.",
    )
    notes_used: int = Field(
        default=0,
        ge=0,
        description="Quantidade de notas aceitas na última calibração.",
    )
    calibration_notes: str = Field(
        default="",
        description="Observações sobre a origem da calibração.",
    )


class GuitarProfile(BaseModel):
    """Perfil completo salvo em configs/profiles/<nome>.yaml."""

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(default="")
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    instrument: InstrumentConfig = Field(default_factory=InstrumentConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    calibration: CalibrationData = Field(default_factory=CalibrationData)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or "/" in cleaned or "\\" in cleaned or ".." in cleaned:
            raise ValueError("Nome de perfil inválido.")
        return cleaned
