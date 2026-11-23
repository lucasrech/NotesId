"""Leitor de rótulos (LabelProvider) e utilitários do dataset EGFxSet."""

from __future__ import annotations

import re
from pathlib import Path

from guitar_string_identifier.core.interfaces import LabelProvider
from guitar_string_identifier.core.music import STANDARD_TUNING, string_fret_to_note
from guitar_string_identifier.core.types import GroundTruthLabel

_EGFX_FILENAME_RE = re.compile(r"^([1-6])-(\d{1,2})\.wav$", re.IGNORECASE)
_VALID_PICKUPS = {"bridge", "bridge-middle", "middle", "middle-neck", "neck"}


class EGFxSetLabelProvider(LabelProvider):
    """Extrai (corda, casa, nota, captador, efeito) da estrutura de pastas do EGFxSet.

    Estrutura verificada no dataset real:
        <base>/<Effect>/<Pickup>/<string>-<fret>.wav
    Exemplo:
        data/egfxset/Clean/Bridge/3-5.wav -> corda 3 (G3), casa 5 -> C4
    """

    def __init__(self, tuning: dict[int, str] | None = None) -> None:
        self.tuning = tuning or STANDARD_TUNING

    def get_label(self, path: Path) -> GroundTruthLabel | None:
        """Extrai corda, casa, captador, efeito e nota inferida do caminho do arquivo.

        Args:
            path: Caminho do arquivo WAV (ex.: `data/egfxset/Clean/Bridge/3-5.wav`).

        Returns:
            `GroundTruthLabel` preenchido ou `None` se o nome não seguir o padrão.
        """
        match = _EGFX_FILENAME_RE.match(path.name)
        if not match:
            return None

        string_num = int(match.group(1))
        fret_num = int(match.group(2))
        if not (1 <= string_num <= 6 and 0 <= fret_num <= 24):
            return None

        pickup_raw = path.parent.name if path.parent else ""
        pickup = pickup_raw if pickup_raw.lower() in _VALID_PICKUPS else None
        effect = path.parent.parent.name if (pickup and path.parent.parent) else None

        note = string_fret_to_note(string_num, fret_num, tuning=self.tuning)
        return GroundTruthLabel(
            string=string_num,
            fret=fret_num,
            note=note,
            pickup=pickup,
            effect=effect,
        )


def list_egfxset_files(
    root: Path | str,
    effect: str | None = "Clean",
    pickup: str | None = None,
    string: int | None = None,
    fret: int | None = None,
) -> list[tuple[Path, GroundTruthLabel]]:
    """Lista arquivos WAV do EGFxSet com seus rótulos, com filtros opcionais."""
    root_path = Path(root)
    if not root_path.exists():
        return []

    provider = EGFxSetLabelProvider()
    items: list[tuple[Path, GroundTruthLabel]] = []
    for wav_path in sorted(root_path.rglob("*.wav")):
        label = provider.get_label(wav_path)
        if label is None:
            continue
        if effect and (label.effect or "").lower() != effect.lower():
            continue
        if pickup and (label.pickup or "").lower() != pickup.lower():
            continue
        if string is not None and label.string != string:
            continue
        if fret is not None and label.fret != fret:
            continue
        items.append((wav_path, label))

    items.sort(key=lambda pair: (pair[1].pickup or "", pair[1].string, pair[1].fret))
    return items
