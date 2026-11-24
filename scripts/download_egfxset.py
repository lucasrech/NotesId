#!/usr/bin/env python3
"""Baixa e verifica o dataset EGFxSet (Zenodo 7044411).

Por padrão baixa apenas os áudios limpos (Clean.zip) e o arquivo de metadados
(egfxset_metadata.csv), com retomada de download e verificação MD5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD_ID = "7044411"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
DEFAULT_FILES = {"Clean.zip", "egfxset_metadata.csv"}


def fetch_record_files() -> list[dict]:
    """Consulta a API do Zenodo para listar arquivos, tamanhos e checksums."""
    with urllib.request.urlopen(ZENODO_API_URL, timeout=30) as resp:
        data = json.load(resp)
    return data.get("files", [])


def compute_md5(path: Path) -> str:
    """Calcula hash MD5 de um arquivo em blocos."""
    md5 = hashlib.md5()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def verify_checksum(path: Path, expected_checksum: str | None) -> bool:
    """Verifica checksum no formato 'md5:<hex>' retornado pelo Zenodo."""
    if not expected_checksum or not expected_checksum.startswith("md5:"):
        return True
    expected_md5 = expected_checksum.split(":", 1)[1].strip().lower()
    actual_md5 = compute_md5(path).lower()
    return actual_md5 == expected_md5


def download_with_resume(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    """Baixa um arquivo em Python puro com suporte a retomada (HTTP Range).

    Funciona de forma idêntica no Linux, macOS e Windows nativo sem depender
    do binário externo `curl`.

    Args:
        url: URL direta do arquivo.
        dest: Caminho de destino em disco.
        chunk_size: Tamanho de cada bloco lido da rede em bytes (padrão: 1 MiB).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing_bytes = dest.stat().st_size if dest.exists() else 0

    headers: dict[str, str] = {"User-Agent": "NotesId-DatasetDownloader/1.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing_bytes > 0:
            # 416 Range Not Satisfiable normalmente indica arquivo já completo
            return
        raise

    with resp:
        status = getattr(resp, "status", 200)
        if status == 206 and existing_bytes > 0:
            mode = "ab"
            downloaded = existing_bytes
        else:
            mode = "wb"
            downloaded = 0

        content_len = resp.headers.get("Content-Length")
        total = (downloaded + int(content_len)) if content_len and content_len.isdigit() else None

        with dest.open(mode) as fp:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                fp.write(chunk)
                downloaded += len(chunk)
                if total and total > 0:
                    pct = 100.0 * downloaded / total
                    print(
                        f"\r  {downloaded / 1e6:6.1f} / {total / 1e6:6.1f} MB ({pct:5.1f}%)",
                        end="",
                        flush=True,
                    )
        if total:
            print()


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extrai arquivo zip mantendo a estrutura interna."""
    print(f"Extraindo {zip_path.name} em {dest_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def main() -> int:
    """Ponto de entrada CLI para listar, baixar, verificar MD5 e extrair o EGFxSet."""
    parser = argparse.ArgumentParser(
        description="Download do dataset EGFxSet (Zenodo 7044411)."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/egfxset"),
        help="Diretório de destino (padrão: data/egfxset).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Apenas lista os arquivos disponíveis no Zenodo e seus tamanhos.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Baixa todos os efeitos (~5,8 GB) em vez de apenas Clean.zip + metadados.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Lista específica de arquivos a baixar (ex: Clean.zip TubeScreamer.zip).",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Não extrai arquivos .zip após o download.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Mantém os arquivos .zip após extração (padrão: mantém por segurança).",
    )
    args = parser.parse_args()

    print(f"Consultando Zenodo (registro {ZENODO_RECORD_ID})...")
    files = fetch_record_files()

    if args.list:
        total = 0
        for item in sorted(files, key=lambda x: x["key"]):
            sz = item.get("size", 0)
            total += sz
            print(f"  {item['key']:28s} {sz / 1e6:8.1f} MB  {item.get('checksum', '')}")
        print(f"Total: {total / 1e9:.2f} GB ({len(files)} arquivos)")
        return 0

    if args.all:
        selected_names = {f["key"] for f in files}
    elif args.files:
        selected_names = set(args.files)
    else:
        selected_names = DEFAULT_FILES

    selected = [f for f in files if f["key"] in selected_names]
    missing = selected_names - {f["key"] for f in selected}
    if missing:
        print(f"Erro: arquivos não encontrados no registro: {sorted(missing)}", file=sys.stderr)
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)

    for item in sorted(selected, key=lambda x: x["size"]):
        key = item["key"]
        url = item["links"]["self"]
        checksum = item.get("checksum")
        size_mb = item.get("size", 0) / 1e6
        dest_file = args.dest / key

        if dest_file.exists() and verify_checksum(dest_file, checksum):
            print(f"[OK] {key} já existe e checksum confere.")
        else:
            print(f"[BAIXANDO] {key} ({size_mb:.1f} MB)...")
            download_with_resume(url, dest_file)
            print(f"[VERIFICANDO] checksum MD5 de {key}...")
            if not verify_checksum(dest_file, checksum):
                print(f"[ERRO] Checksum MD5 inválido para {key}!", file=sys.stderr)
                return 1
            print(f"[OK] {key} verificado com sucesso.")

        if key.endswith(".zip") and not args.no_extract:
            extract_zip(dest_file, args.dest)

    print("Download concluído.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
