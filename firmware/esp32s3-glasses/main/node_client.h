// The node protocol over WS /ws/node: hello/welcome, invoke/result, events, ping (docs/NODES.md section 4).
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Starts the client and keeps it connected (own reconnect backoff, docs/NODES.md section 8).
// `cert_pem` is the pinned engine certificate; `pair_code` may be empty once a token is stored.
bool node_client_start(const char *host, int port, const char *path, const char *cert_pem, const char *pair_code);

bool node_client_connected(void);
const char *node_client_token(void); // the token the voice socket authenticates with

// All sends are serialised, so a result and its binary payload always stay adjacent.
bool node_client_send_json(const char *json);
bool node_client_send_binary(const uint8_t *data, size_t len);
bool node_client_send_result_json(const char *json, const uint8_t *payload, size_t payload_len);

void node_client_send_event(const char *event, const char *data_json); // data_json may be NULL
void node_client_send_state(int battery, bool charging);
