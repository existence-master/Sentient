#include "cert_pin.h"

#include <ctype.h>
#include <string.h>

#include "esp_log.h"
#include "esp_tls.h"
#include "mbedtls/base64.h"
#include "mbedtls/sha256.h"
#include "mbedtls/ssl.h"

static const char *TAG = "cert_pin";

static void hex_of(const unsigned char *digest, char *out /* 65 bytes */) {
    static const char *hex = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        out[i * 2] = hex[digest[i] >> 4];
        out[i * 2 + 1] = hex[digest[i] & 0x0f];
    }
    out[64] = '\0';
}

static bool same_fingerprint(const char *a, const char *b) {
    // Compare case-insensitively and ignore ':' separators, so any display format works.
    while (*a && *b) {
        if (*a == ':') { a++; continue; }
        if (*b == ':') { b++; continue; }
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return false;
        a++;
        b++;
    }
    while (*a == ':') a++;
    while (*b == ':') b++;
    return *a == '\0' && *b == '\0';
}

// DER -> PEM, because esp_websocket_client wants a PEM string as its CA.
static bool der_to_pem(const unsigned char *der, size_t der_len, char *out, size_t cap) {
    static const char *HEADER = "-----BEGIN CERTIFICATE-----\n";
    static const char *FOOTER = "-----END CERTIFICATE-----\n";
    size_t header_len = strlen(HEADER), footer_len = strlen(FOOTER);
    if (cap < header_len + footer_len + 4) return false;
    size_t written = 0;
    memcpy(out, HEADER, header_len);
    // mbedtls writes the base64 body with newlines every 64 characters when we pass the right buffer.
    int rc = mbedtls_base64_encode((unsigned char *)out + header_len, cap - header_len - footer_len - 1, &written,
                                   der, der_len);
    if (rc != 0) {
        ESP_LOGE(TAG, "base64 of the certificate failed (-0x%04x)", -rc);
        return false;
    }
    // Insert the line breaks PEM needs (64 chars per line) by rewriting the body in place.
    char body[CERT_PEM_MAX];
    if (written >= sizeof(body)) return false;
    memcpy(body, out + header_len, written);
    size_t pos = header_len;
    for (size_t i = 0; i < written; i += 64) {
        size_t chunk = (written - i) < 64 ? (written - i) : 64;
        if (pos + chunk + 1 + footer_len + 1 > cap) return false;
        memcpy(out + pos, body + i, chunk);
        pos += chunk;
        out[pos++] = '\n';
    }
    memcpy(out + pos, FOOTER, footer_len);
    out[pos + footer_len] = '\0';
    return true;
}

bool cert_pin_fetch(const char *host, int port, const char *expected_fp, char *pem_out, size_t pem_cap,
                    char *seen_fp_out) {
    seen_fp_out[0] = '\0';
    // No CA is configured on purpose: esp_tls then does not verify, and we verify by hash instead.
    esp_tls_cfg_t cfg = {
        .skip_common_name = true,
        .timeout_ms = 8000,
    };
    esp_tls_t *tls = esp_tls_init();
    if (tls == NULL) return false;
    bool ok = false;
    if (esp_tls_conn_new_sync(host, (int)strlen(host), port, &cfg, tls) != 1) {
        ESP_LOGE(TAG, "could not open a TLS connection to %s:%d", host, port);
        esp_tls_conn_destroy(tls);
        return false;
    }
    mbedtls_ssl_context *ssl = (mbedtls_ssl_context *)esp_tls_get_ssl_context(tls);
    const mbedtls_x509_crt *crt = ssl ? mbedtls_ssl_get_peer_cert(ssl) : NULL;
    if (crt == NULL || crt->raw.p == NULL) {
        ESP_LOGE(TAG, "the engine presented no certificate");
    } else {
        unsigned char digest[32];
        mbedtls_sha256(crt->raw.p, crt->raw.len, digest, 0 /* 0 = SHA-256, not SHA-224 */);
        hex_of(digest, seen_fp_out);
        if (expected_fp && expected_fp[0] && !same_fingerprint(expected_fp, seen_fp_out)) {
            ESP_LOGE(TAG, "FINGERPRINT MISMATCH: expected %s, got %s", expected_fp, seen_fp_out);
            ESP_LOGE(TAG, "someone may be intercepting the connection; not sending the token");
        } else {
            if (!expected_fp || expected_fp[0] == '\0') {
                ESP_LOGW(TAG, "no expected fingerprint: trusting %s on first use", seen_fp_out);
            }
            ok = der_to_pem(crt->raw.p, crt->raw.len, pem_out, pem_cap);
            if (ok) ESP_LOGI(TAG, "pinned certificate %s (%u bytes PEM)", seen_fp_out, (unsigned)strlen(pem_out));
        }
    }
    esp_tls_conn_destroy(tls);
    return ok;
}
