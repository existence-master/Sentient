#include "button.h"

#include "app_config.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "button";
static button_cb_t s_cb;

// Polling instead of an interrupt: 10 ms is plenty for a finger, and it keeps the
// callback out of ISR context so it can send websocket frames directly.
static void button_task(void *arg) {
    bool pressed = false;
    bool long_sent = false;
    int64_t changed_us = 0;
    while (true) {
        bool level_low = gpio_get_level(BUTTON_GPIO) == 0; // pull-up: low means pressed
        int64_t now = esp_timer_get_time();
        if (level_low != pressed && (now - changed_us) > BUTTON_DEBOUNCE_MS * 1000) {
            pressed = level_low;
            changed_us = now;
            long_sent = false;
            ESP_LOGD(TAG, "button %s", pressed ? "press" : "release");
            if (s_cb) s_cb(pressed ? BUTTON_PRESS : BUTTON_RELEASE);
        } else if (pressed && !long_sent && (now - changed_us) > BUTTON_LONG_PRESS_MS * 1000) {
            long_sent = true;
            if (s_cb) s_cb(BUTTON_LONG_PRESS);
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

void button_start(button_cb_t callback) {
    s_cb = callback;
    gpio_config_t cfg = {
        .pin_bit_mask = 1ULL << BUTTON_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&cfg));
    xTaskCreate(button_task, "button", 3072, NULL, 5, NULL);
}
