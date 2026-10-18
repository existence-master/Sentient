// The talk button: press to listen, release to send (docs/NODES.md section 11.6).
#pragma once
#include <stdbool.h>

typedef enum {
    BUTTON_PRESS,
    BUTTON_RELEASE,
    BUTTON_LONG_PRESS,
} button_event_t;

typedef void (*button_cb_t)(button_event_t event);

void button_start(button_cb_t callback);

// Right after start-up, before button_start(): waits up to `window_ms` for a press, then true when the button is
// held for `hold_ms` without letting go. Then waits (up to 10 s) for the release, so the hold does not also start a
// talk press. The press comes after reset on purpose: BUTTON_GPIO defaults to GPIO0, a strapping pin, and holding
// it low through reset starts the ROM's download mode instead of this firmware.
bool button_held_at_boot(int window_ms, int hold_ms);
