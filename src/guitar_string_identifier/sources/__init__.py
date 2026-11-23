"""Fontes de áudio e provedores de gabarito (LabelProvider) do NotesId.

Inclui leitura de arquivos locais (`FileAudioSource`), listagem e leitura de
rótulos do dataset EGFxSet (`EGFxSetLabelProvider`), buffer em memória para
streaming WebSocket (`BufferAudioSource`) e captura opcional via `sounddevice`.
"""

from guitar_string_identifier.sources.dataset_source import (
    EGFxSetLabelProvider,
    list_egfxset_files,
)
from guitar_string_identifier.sources.file_source import FileAudioSource
from guitar_string_identifier.sources.live_source import BufferAudioSource, SoundDeviceAudioSource

__all__ = [
    "BufferAudioSource",
    "EGFxSetLabelProvider",
    "FileAudioSource",
    "SoundDeviceAudioSource",
    "list_egfxset_files",
]
