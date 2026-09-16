from cryptography import x509

from sentient.nodes import tls
from sentient.nodes.tls import ensure_certificate, normalize_fingerprint


def test_web_app_served_on_gateway(nodes_client):
    client = nodes_client()
    res = client.get("/node/")
    assert res.status_code == 200 and "text/html" in res.headers["content-type"]
    assert "Pair" in res.text and "app.js" in res.text
    js = client.get("/node/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert "/ws/voice?node_token=" in js.text and "camera.photo" in js.text
    assert client.get("/node/style.css").headers["content-type"].startswith("text/css")
    assert client.get("/node", follow_redirects=False).status_code in {307, 308}
    assert client.get("/node/%2e%2e/service.py").status_code == 404
    assert client.get("/node/missing.js").status_code == 404


def test_lan_status_when_disabled(nodes_client):
    client = nodes_client()
    status = client.get("/api/nodes/lan", headers=client.api_headers).json()
    assert status["enabled"] is False and status["running"] is False and status["urls"] == []
    assert client.get("/api/nodes/lan").status_code == 401


def test_certificate_generated_and_reused(tmp_path, monkeypatch):
    first = ensure_certificate(tmp_path, ips=["192.168.1.20"])
    assert first.created and len(first.fingerprint) == 64
    cert = x509.load_pem_x509_certificate(first.cert_file.read_bytes())
    sans = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "localhost" in sans.get_values_for_type(x509.DNSName)
    assert "192.168.1.20" in [str(ip) for ip in sans.get_values_for_type(x509.IPAddress)]
    again = ensure_certificate(tmp_path)
    assert not again.created and again.fingerprint == first.fingerprint

    from datetime import timedelta

    monkeypatch.setattr(tls, "RENEW_BEFORE", timedelta(days=10 * 365))  # "about to expire"
    renewed = ensure_certificate(tmp_path)
    assert renewed.created and renewed.fingerprint != first.fingerprint


def test_fingerprint_normalization():
    assert normalize_fingerprint("sha256:AB:cd:01") == "abcd01"
    assert normalize_fingerprint(" AB CD ") == "abcd"
