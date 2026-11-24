"""Testes para as opções de CLI (--host, --port, --https) e servidor HTTPS/WSS."""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
import threading
import time
import urllib.request
from pathlib import Path

import pytest
import uvicorn
import websockets
from cryptography import x509

from guitar_string_identifier.web.app import create_app, parse_server_args
from guitar_string_identifier.web.tls import ensure_self_signed_cert


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_default_cli_args_are_local_http():
    args = parse_server_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.https is False
    assert args.cert_dir == Path(".certs")


def test_cli_args_accept_lan_and_https():
    args = parse_server_args(
        ["--host", "0.0.0.0", "--port", "8443", "--https", "--cert-dir", "/tmp/certs"]
    )
    assert args.host == "0.0.0.0"
    assert args.port == 8443
    assert args.https is True
    assert args.cert_dir == Path("/tmp/certs")


def test_self_signed_cert_generation_and_reuse(tmp_path: Path):
    cert_dir = tmp_path / "certs"
    cert_path, key_path, san_ips = ensure_self_signed_cert(
        cert_dir=cert_dir,
        extra_ips=["192.168.0.10"],
    )
    assert cert_path.exists()
    assert key_path.exists()
    assert "127.0.0.1" in san_ips
    assert "192.168.0.10" in san_ips

    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    dns_names = san.get_values_for_type(x509.DNSName)
    ip_addrs = {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
    assert "localhost" in dns_names
    assert "127.0.0.1" in ip_addrs
    assert "192.168.0.10" in ip_addrs

    # Segunda chamada com os mesmos IPs deve reutilizar o certificado existente
    mtime_before = cert_path.stat().st_mtime_ns
    cert_path2, key_path2, _ = ensure_self_signed_cert(
        cert_dir=cert_dir,
        extra_ips=["192.168.0.10"],
    )
    assert cert_path2 == cert_path
    assert key_path2 == key_path
    assert cert_path.stat().st_mtime_ns == mtime_before


def test_https_and_wss_live_endpoint_with_self_signed_cert(tmp_path: Path):
    cert_path, key_path, _ = ensure_self_signed_cert(cert_dir=tmp_path / "tls")
    port = _find_free_port()
    app = create_app()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        ssl_certfile=str(cert_path),
        ssl_keyfile=str(key_path),
        log_level="error",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    try:
        # Aguarda o servidor subir
        deadline = time.monotonic() + 10.0
        while not server.started:
            if time.monotonic() > deadline:
                pytest.fail("Servidor HTTPS de teste não iniciou no tempo esperado.")
            time.sleep(0.05)

        # Desabilita verificação de certificado APENAS neste teste automatizado
        unverified_ctx = ssl._create_unverified_context()

        # 1. Verifica endpoint HTTP sobre TLS (HTTPS)
        with urllib.request.urlopen(
            f"https://127.0.0.1:{port}/api/health",
            context=unverified_ctx,
            timeout=5.0,
        ) as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["status"] == "ok"

        # 2. Verifica WebSocket seguro (WSS) em /ws/live
        async def _check_wss() -> list[dict]:
            received: list[dict] = []
            async with websockets.connect(
                f"wss://127.0.0.1:{port}/ws/live",
                ssl=unverified_ctx,
            ) as ws:
                await ws.send(
                    json.dumps(
                        {"type": "config", "sample_rate": 48000, "profile": "egfxset"}
                    )
                )
                for _ in range(2):
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    received.append(json.loads(raw))
            return received

        messages = asyncio.run(_check_wss())
        types = {m["type"] for m in messages}
        assert "ready" in types
        assert "status" in types
        status_msg = next(m for m in messages if m["type"] == "status")
        assert status_msg["state"] == "listening"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)
