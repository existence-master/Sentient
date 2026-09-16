// Wi-Fi station with the reconnect backoff from docs/NODES.md section 8.
#pragma once
#include <stdbool.h>

void net_start(void);                    // starts Wi-Fi and keeps it connected forever
bool net_wait_connected(int timeout_ms);  // blocks until an IP is assigned
bool net_is_connected(void);
