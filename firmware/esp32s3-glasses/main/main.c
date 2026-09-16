// Sentient glasses: ESP32-S3 + INMP441 microphone + MAX98357A amplifier + OV3660 camera.
// Protocol: docs/NODES.md (section 11 covers this board). Flashing: README.md.
#include <stdlib.h>
#include <string.h>

#include "app_config.h"
#include "audio.h"
#include "battery.h"
#include "button.h"
#include "camera.h"
#include "caps.h"
#include "cert_pin.h"
#include "discovery.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "net.h"
#include "node_client.h"
#include "storage.h"
#include "voice.h"

static const char *TAG = "glasses";
static char s_cert_pem[CERT_PEM_MAX];

static void on_button(button_event_t event) {
    switch (event) {
        case BUTTON_PRESS:
            node_client_send_event("button", "{\"action\":\"press\"}");
            voice_press();
            break;
        case BUTTON_RELEASE:
            node_client_send_event("button", "{\"action\":\"release\"}");
            voice_release();
            break;
        case BUTTON_LONG_PRESS:
            node_client_send_event("button", "{\"action\":\"long_press\"}");
            break;
    }
}

// Reports the battery on connect, every BATTERY_REPORT_INTERVAL_MS, and on every 2% change.
static void battery_task(void *arg) {
    int last = -100;
    int64_t last_sent_ms = 0;
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(30000));
        if (!node_client_connected()) continue;
        int percent = battery_percent();
        if (percent < 0) continue;
        int64_t now_ms = esp_log_timestamp();
        bool due = (now_ms - last_sent_ms) > BATTERY_REPORT_INTERVAL_MS;
        if (due || percent <= last - BATTERY_REPORT_DELTA || percent >= last + BATTERY_REPORT_DELTA) {
            node_client_send_state(percent, false);
            last = percent;
            last_sent_ms = now_ms;
        }
    }
}

// Finds the engine: the address saved in NVS first, then mDNS, then the one compiled in.
static bool locate_engine(engine_addr_t *engine) {
    memset(engine, 0, sizeof(*engine));
    if (strlen(ENGINE_HOST) > 0) {
        strlcpy(engine->host, ENGINE_HOST, sizeof(engine->host));
        engine->port = ENGINE_PORT;
        strlcpy(engine->path, "/ws/node", sizeof(engine->path));
        return true;
    }
    discovery_init();
    for (int attempt = 0; attempt < 3; attempt++) {
        if (discovery_find(engine, 3000)) return true;
        ESP_LOGW(TAG, "mDNS found nothing, retrying (%d/3)", attempt + 1);
        vTaskDelay(pdMS_TO_TICKS(2000));
    }
    char host[64], port[8];
    if (storage_get(KEY_HOST, host, sizeof(host)) && storage_get(KEY_PORT, port, sizeof(port))) {
        ESP_LOGI(TAG, "falling back to the last known engine %s:%s", host, port);
        strlcpy(engine->host, host, sizeof(engine->host));
        engine->port = atoi(port);
        strlcpy(engine->path, "/ws/node", sizeof(engine->path));
        return true;
    }
    return false;
}

// Pins the engine certificate: the stored one if we have it, otherwise fetch and verify it once.
static bool pin_certificate(const engine_addr_t *engine) {
    if (storage_get(KEY_CERT, s_cert_pem, sizeof(s_cert_pem))) {
        ESP_LOGI(TAG, "using the certificate pinned in NVS");
        return true;
    }
    char expected[72];
    strlcpy(expected, ENGINE_FINGERPRINT, sizeof(expected));
    if (expected[0] == '\0') strlcpy(expected, engine->fingerprint, sizeof(expected));
    if (expected[0] == '\0') {
        ESP_LOGW(TAG, "no expected fingerprint (set ENGINE_FINGERPRINT from Sentient > Devices)");
    }
    char seen[72] = {0};
    if (!cert_pin_fetch(engine->host, engine->port, expected, s_cert_pem, sizeof(s_cert_pem), seen)) {
        ESP_LOGE(TAG, "certificate check failed; not connecting");
        return false;
    }
    storage_set(KEY_CERT, s_cert_pem);
    storage_set(KEY_FINGERPRINT, seen);
    return true;
}

void app_main(void) {
    ESP_LOGI(TAG, "Sentient glasses %s starting", NODE_APP_VERSION);
    storage_init();
    battery_init();
    audio_init();
    if (!camera_start()) ESP_LOGW(TAG, "continuing without a camera");

    net_start();
    if (!net_wait_connected(30000)) {
        ESP_LOGE(TAG, "no Wi-Fi; check WIFI_SSID and WIFI_PASSWORD in app_config.h");
        return;
    }

    engine_addr_t engine;
    if (!locate_engine(&engine)) {
        ESP_LOGE(TAG, "no Sentient engine found: turn on 'Accept devices on your network' in Sentient");
        return;
    }
    char port_text[8];
    snprintf(port_text, sizeof(port_text), "%d", engine.port);
    storage_set(KEY_HOST, engine.host);
    storage_set(KEY_PORT, port_text);

    if (!pin_certificate(&engine)) return;

    if (!node_client_start(engine.host, engine.port, engine.path, s_cert_pem, PAIRING_CODE)) {
        ESP_LOGE(TAG, "could not start the node client");
        return;
    }
    voice_init(engine.host, engine.port, s_cert_pem);
    button_start(on_button);
    xTaskCreate(battery_task, "battery", 3072, NULL, 3, NULL);

    caps_show("Sentient", "Ready. Hold the button to talk.");
}
