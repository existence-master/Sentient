// Certificate pinning in two steps (docs/NODES.md section 11.4):
//   1. open a plain TLS connection, hash the server certificate, compare with the expected fingerprint
//   2. keep that certificate as PEM and hand it to the websocket client as its only trusted CA
#pragma once
#include <stdbool.h>
#include <stddef.h>

#define CERT_PEM_MAX 2048 // an EC P-256 self-signed certificate is ~800 bytes of PEM

// Connects to host:port, verifies the certificate against `expected_fp` (64 hex chars,
// case-insensitive; empty means trust-on-first-use) and writes it as PEM into `pem_out`.
// `seen_fp_out` (65 bytes) receives the fingerprint that was actually presented.
bool cert_pin_fetch(const char *host, int port, const char *expected_fp, char *pem_out, size_t pem_cap,
                    char *seen_fp_out);
