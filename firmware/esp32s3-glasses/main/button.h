// The talk button: press to listen, release to send (docs/NODES.md section 11.6).
#pragma once

typedef enum {
    BUTTON_PRESS,
    BUTTON_RELEASE,
    BUTTON_LONG_PRESS,
} button_event_t;

typedef void (*button_cb_t)(button_event_t event);

void button_start(button_cb_t callback);
