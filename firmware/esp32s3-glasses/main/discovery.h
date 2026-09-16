// Finding the engine on the network: _sentient._tcp.local. (docs/NODES.md section 1).
#pragma once
#include <stdbool.h>

typedef struct {
    char host[64];        // IPv4 address as text
    int port;             // usually 7778
    char path[48];        // "/ws/node" from the TXT record
    char fingerprint[72]; // SHA-256 of the TLS certificate from the TXT record (may be empty)
} engine_addr_t;

void discovery_init(void);

// Queries mDNS for up to `timeout_ms`. Returns false when nothing answered.
bool discovery_find(engine_addr_t *out, int timeout_ms);
