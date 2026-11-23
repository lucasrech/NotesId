"""Heurística de seleção de interface de áudio, canal ativo e tamanho de buffer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Palavras-chave que indicam interface de áudio / guitarra (pontuação positiva)
_PREFERRED_KEYWORDS = (
    "focusrite",
    "scarlett",
    "behringer",
    "umc",
    "audient",
    "steinberg",
    "ur22",
    "presonus",
    "motu",
    "volt",
    "universal audio",
    "apollo",
    "line 6",
    "helix",
    "pod",
    "boss",
    "katana",
    "roland",
    "tascam",
    "m-audio",
    "zoom",
    "rme",
    "ssl",
    "usb audio",
    "usb",
    "interface",
    "instrument",
    "guitar",
    "hi-z",
    "asio",
)

# Palavras-chave que indicam microfone interno/virtual (pontuação negativa)
_AVOID_KEYWORDS = (
    "realtek",
    "microfone integrado",
    "microphone array",
    "internal",
    "built-in",
    "webcam",
    "camera",
    "bluetooth",
    "hands-free",
    "headset",
    "virtual",
    "cable",
    "voicemeeter",
    "default",
    "communications",
)


@dataclass
class RankedDevice:
    """Dispositivo de entrada classificado pela heurística de nome."""

    device_id: str
    label: str
    score: int
    reason: str


@dataclass
class ActiveChannelResult:
    """Resultado da detecção de canal ativo em sinal multicanal."""

    active_channel: int
    channel_rms_db: list[float]
    has_signal: bool


class DeviceProbe:
    """Seleciona placa provável por nome, detecta canal ativo e sugere buffer."""

    def rank_devices(self, devices: list[dict[str, str]]) -> list[RankedDevice]:
        """Ordena dispositivos de entrada do mais provável (interface USB) ao menos provável."""
        ranked: list[RankedDevice] = []
        for dev in devices:
            dev_id = str(dev.get("deviceId") or dev.get("device_id") or "")
            label = str(dev.get("label") or "").strip()
            low = label.lower()
            score = 0
            reasons: list[str] = []

            for kw in _PREFERRED_KEYWORDS:
                if kw in low:
                    score += 10
                    reasons.append(f"+interface({kw})")
                    break
            if "usb" in low and score < 10:
                score += 6
                reasons.append("+usb")

            for kw in _AVOID_KEYWORDS:
                if kw in low:
                    score -= 8
                    reasons.append(f"-evitar({kw})")
                    break

            if not label:
                score -= 5
                reasons.append("sem nome")

            ranked.append(
                RankedDevice(
                    device_id=dev_id,
                    label=label or "Entrada sem nome",
                    score=score,
                    reason=", ".join(reasons) if reasons else "neutro",
                )
            )

        ranked.sort(key=lambda d: d.score, reverse=True)
        return ranked

    def select_active_channel(
        self,
        audio: np.ndarray,
        min_signal_db: float = -55.0,
    ) -> ActiveChannelResult:
        """Identifica qual canal recebeu sinal em áudio mono ou multicanal."""
        arr = np.asarray(audio, dtype=np.float64)
        if arr.ndim == 1:
            rms = float(np.sqrt(np.mean(arr**2) + 1e-18))
            rms_db = round(20.0 * math.log10(rms + 1e-12), 2)
            return ActiveChannelResult(
                active_channel=0,
                channel_rms_db=[rms_db],
                has_signal=rms_db >= min_signal_db,
            )

        # Formato esperado: (n_samples, n_channels)
        n_channels = arr.shape[1]
        dbs: list[float] = []
        for ch in range(n_channels):
            sig = arr[:, ch]
            rms = float(np.sqrt(np.mean(sig**2) + 1e-18))
            dbs.append(round(20.0 * math.log10(rms + 1e-12), 2))

        best_ch = int(np.argmax(dbs))
        return ActiveChannelResult(
            active_channel=best_ch,
            channel_rms_db=dbs,
            has_signal=dbs[best_ch] >= min_signal_db,
        )

    def suggest_buffer_size(
        self,
        sample_rate: int,
        base_latency_sec: float | None = None,
    ) -> int:
        """Propõe tamanho de buffer potência de 2 a partir da taxa e baseLatency."""
        if base_latency_sec is not None and base_latency_sec > 0:
            # Alvo ~3x a latência base (mínimo 1024, máximo 4096 para WebSocket estável)
            raw = int(round(sample_rate * max(base_latency_sec * 3.0, 0.04)))
        else:
            raw = int(round(sample_rate * 0.06))

        # Arredonda para potência de 2 em [1024, 4096]
        pow2 = 1 << int(round(math.log2(max(256, raw))))
        return int(np.clip(pow2, 1024, 4096))
