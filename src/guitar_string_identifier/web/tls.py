"""Geração de certificado TLS autoassinado para uso local em HTTPS na rede local.

Permite que o servidor web do NotesId rode com ``--https`` tanto em Linux
quanto em Windows sem depender do binário externo ``openssl``. Os arquivos
gerados (``cert.pem`` e ``key.pem``) ficam em diretório ignorado pelo Git
(por padrão ``.certs/``) e incluem o IP local na extensão Subject Alternative
Name (SAN) para que navegadores aceitem o endereço após a confirmação manual.
"""

from __future__ import annotations

import datetime
import ipaddress
import os
import socket
from collections.abc import Sequence
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

__all__ = ["detect_local_ips", "ensure_self_signed_cert"]


def detect_local_ips() -> list[str]:
    """Detecta endereços IPv4 não-loopback da máquina na rede local.

    Utiliza um socket UDP sem envio de pacotes reais para descobrir a interface
    de rota padrão e complementa com a resolução do hostname local.

    Returns:
        Lista ordenada e sem duplicatas de endereços IPv4 locais detectados.
    """
    ips: set[str] = set()

    # Descoberta pela interface de saída padrão (não envia pacotes na rede)
    for target in ("192.0.2.1", "10.255.255.255"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect((target, 80))
                addr = sock.getsockname()[0]
                if addr and not addr.startswith("127."):
                    ips.add(addr)
        except OSError:
            continue

    # Complemento via resolução do hostname local
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = str(info[4][0])
            if addr and not addr.startswith("127."):
                ips.add(addr)
    except OSError:
        pass

    return sorted(ips)


def _extract_san_ips(cert_path: Path) -> set[str]:
    """Lê os endereços IP presentes na extensão SAN de um certificado PEM.

    Args:
        cert_path: Caminho do arquivo ``cert.pem``.

    Returns:
        Conjunto de IPs (como strings) contidos na extensão SAN, ou conjunto
        vazio caso o certificado não possa ser lido ou esteja expirado.
    """
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        now = datetime.datetime.now(datetime.UTC)
        if cert.not_valid_after_utc <= now + datetime.timedelta(days=7):
            return set()
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        return {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
    except Exception:
        return set()


def ensure_self_signed_cert(
    cert_dir: Path,
    extra_ips: Sequence[str] = (),
    validity_days: int = 365,
) -> tuple[Path, Path, list[str]]:
    """Garante a existência de um par certificado/chave TLS autoassinado.

    Reutiliza o par existente em ``cert_dir`` se ele ainda for válido por mais
    de 7 dias e já contemplar todos os IPs solicitados no SAN. Caso contrário,
    gera um novo certificado ECDSA P-256 (compatível com Chrome/Edge/Safari e
    Windows/Linux sem depender de ``openssl``).

    Args:
        cert_dir: Diretório onde ``cert.pem`` e ``key.pem`` serão salvos.
        extra_ips: Endereços IP adicionais a incluir no Subject Alternative Name.
        validity_days: Validade do certificado gerado em dias.

    Returns:
        Tupla ``(cert_path, key_path, san_ips)`` com os caminhos dos arquivos
        PEM e a lista de endereços IP incluídos no SAN.
    """
    cert_dir = Path(cert_dir)
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"

    ip_strings: set[str] = {"127.0.0.1", "::1"}
    ip_strings.update(detect_local_ips())
    for raw_ip in extra_ips:
        candidate = raw_ip.strip()
        if candidate and candidate not in {"0.0.0.0", "::"}:
            try:
                ipaddress.ip_address(candidate)
                ip_strings.add(candidate)
            except ValueError:
                pass

    if cert_path.exists() and key_path.exists():
        existing_ips = _extract_san_ips(cert_path)
        if ip_strings.issubset(existing_ips):
            return cert_path, key_path, sorted(existing_ips)

    private_key = ec.generate_private_key(ec.SECP256R1())
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "NotesId Local HTTPS"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "NotesId Self-Signed"),
        ]
    )

    san_entries: list[x509.GeneralName] = [x509.DNSName("localhost")]
    for ip_str in sorted(ip_strings):
        san_entries.append(x509.IPAddress(ipaddress.ip_address(ip_str)))

    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=validity_days))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    cert_path.write_bytes(cert_pem)
    key_path.write_bytes(key_pem)
    if os.name == "posix":
        os.chmod(key_path, 0o600)

    return cert_path, key_path, sorted(ip_strings)
