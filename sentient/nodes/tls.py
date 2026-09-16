"""Self-signed TLS for the LAN listener, LAN address discovery and certificate fingerprints.

The certificate lives in ``<home>/tls/`` and is reused until it is about to expire, so the
SHA-256 fingerprint devices pin stays stable. Devices never rely on the hostname in the
certificate: they compare the fingerprint (see docs/NODES.md).
"""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import logging
import socket
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sentient import paths

log = logging.getLogger(__name__)

CERT_NAME = "node.crt"
KEY_NAME = "node.key"
VALID_DAYS = 5 * 365
RENEW_BEFORE = timedelta(days=30)


@dataclass
class CertInfo:
    cert_file: Path
    key_file: Path
    fingerprint: str  # 64 lowercase hex characters, SHA-256 of the DER certificate
    not_after: datetime
    created: bool


def tls_dir() -> Path:
    return paths.home() / "tls"


def fingerprint_der(der: bytes) -> str:
    return hashlib.sha256(der).hexdigest()


def normalize_fingerprint(value: str) -> str:
    """Accept ``AB:CD:...``, ``sha256:abcd...`` or plain hex; return lowercase hex."""
    v = (value or "").strip().lower()
    if v.startswith("sha256:"):
        v = v[7:]
    return v.replace(":", "").replace(" ", "")


def lan_ipv4s() -> list[str]:
    """Private, non-loopback IPv4 addresses of this machine, the default-route address first."""
    found: list[str] = []
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))  # no packet is sent; this only picks the outgoing interface
        found.append(s.getsockname()[0])
    with contextlib.suppress(OSError):
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(str(info[4][0]))
    out: list[str] = []
    for ip in found:
        try:
            addr = ipaddress.IPv4Address(ip)
        except ValueError:
            continue
        if addr.is_loopback or addr.is_link_local or addr.is_unspecified or ip in out:
            continue
        out.append(ip)
    return out


def _load(cert_file: Path) -> tuple[str, datetime] | None:
    from cryptography import x509
    from cryptography.hazmat.primitives.serialization import Encoding

    try:
        cert = x509.load_pem_x509_certificate(cert_file.read_bytes())
    except Exception:
        return None
    return fingerprint_der(cert.public_bytes(Encoding.DER)), cert.not_valid_after_utc


def ensure_certificate(directory: Path | None = None, *, ips: list[str] | None = None) -> CertInfo:
    """Return the LAN certificate, generating a new one when missing, unreadable or expiring."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    directory = directory or tls_dir()
    cert_file, key_file = directory / CERT_NAME, directory / KEY_NAME
    now = datetime.now(UTC)
    if cert_file.exists() and key_file.exists():
        loaded = _load(cert_file)
        if loaded and loaded[1] - RENEW_BEFORE > now:
            return CertInfo(cert_file, key_file, loaded[0], loaded[1], created=False)

    directory.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    host = socket.gethostname() or "sentient"
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Sentient local device link")])
    sans: list[x509.GeneralName] = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]
    with contextlib.suppress(ValueError):
        sans.append(x509.DNSName(f"{host}.local".encode("idna").decode()))
    for ip in ips if ips is not None else lan_ipv4s():
        with contextlib.suppress(ValueError):
            sans.append(x509.IPAddress(ipaddress.IPv4Address(ip)))
    not_after = now + timedelta(days=VALID_DAYS)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(not_after)
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    key_file.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    with contextlib.suppress(Exception):
        key_file.chmod(0o600)
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    fp = fingerprint_der(cert.public_bytes(serialization.Encoding.DER))
    log.info("generated LAN certificate %s (sha256 %s)", cert_file, fp)
    return CertInfo(cert_file, key_file, fp, cert.not_valid_after_utc, created=True)
