// OV3660 capture for `camera.photo` (docs/NODES.md section 11.3).
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool camera_start(void);   // false when the sensor does not answer (check the pins in app_config.h)
bool camera_available(void);

// Captures one JPEG. The buffer belongs to the driver: send it, then call camera_release().
// It lives in PSRAM, so it can be handed to the websocket client without copying.
bool camera_capture(const uint8_t **data, size_t *len, int *width, int *height);
void camera_release(void);
