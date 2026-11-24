"""Testes de compatibilidade multiplataforma do script scripts/download_egfxset.py."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_download_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "download_egfxset.py"
    spec = importlib.util.spec_from_file_location("download_egfxset", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_md5_verification(tmp_path: Path) -> None:
    mod = _load_download_module()
    f = tmp_path / "sample.bin"
    payload = b"notesid-test-payload-12345"
    f.write_bytes(payload)
    expected = "md5:" + hashlib.md5(payload).hexdigest()

    assert mod.verify_checksum(f, expected) is True
    assert mod.verify_checksum(f, "md5:00000000000000000000000000000000") is False


def test_download_with_resume_pure_python(tmp_path: Path) -> None:
    mod = _load_download_module()
    src = tmp_path / "source.bin"
    payload = b"abcdefghij" * 20
    src.write_bytes(payload)

    dest = tmp_path / "out.bin"
    mod.download_with_resume(src.as_uri(), dest, chunk_size=32)
    assert dest.read_bytes() == payload
