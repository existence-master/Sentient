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

// At power on, before button_start(): true when the button is held down for `hold_ms` without letting go.
// Then waits (up to 10 s) for it to be released, so the hold does not also start a talk press.
bool button_held_at_boot(int hold_ms);
