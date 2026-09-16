// Small NVS wrapper: the pairing token, the pinned certificate and the last known engine address.
#pragma once
#include <stdbool.h>
#include <stddef.h>

void storage_init(void);

// Returns false when the key is missing or does not fit in `cap`.
bool storage_get(const char *key, char *out, size_t cap);
bool storage_set(const char *key, const char *value);
void storage_erase(const char *key);

#define KEY_TOKEN "token"       // node token from `welcome`
#define KEY_CERT "cert_pem"     // pinned engine certificate, PEM
#define KEY_FINGERPRINT "fp"    // its SHA-256, 64 hex chars
#define KEY_HOST "host"         // last engine address, so a reboot can skip discovery
#define KEY_PORT "port"
