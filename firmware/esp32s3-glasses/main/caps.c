#include "caps.h"

#include <stdio.h>
#include <string.h>

#include "battery.h"
#include "cJSON.h"
#include "camera.h"
#include "esp_log.h"
#include "node_client.h"

static const char *TAG = "caps";

void caps_show(const char *title, const char *text) {
    // The reference board has no screen: the serial console is the display.
    // For an SSD1306, call your driver here (README: "Adding a small OLED").
    if (title && title[0]) {
        ESP_LOGI(TAG, "+---- %s ----", title);
    }
    ESP_LOGI(TAG, "| %s", text ? text : "");
}

static void send_ok(const char *id, const char *data_json) {
    char buf[256];
    snprintf(buf, sizeof(buf), "{\"type\":\"result\",\"id\":\"%s\",\"ok\":true,\"data\":%s}", id,
             data_json ? data_json : "{}");
    node_client_send_json(buf);
}

static void send_error(const char *id, const char *code, const char *message) {
    char buf[320];
    snprintf(buf, sizeof(buf),
             "{\"type\":\"result\",\"id\":\"%s\",\"ok\":false,\"error\":{\"code\":\"%s\",\"message\":\"%s\"}}", id,
             code, message);
    node_client_send_json(buf);
}

static const char *string_field(cJSON *params, const char *key, const char *fallback) {
    const cJSON *item = cJSON_GetObjectItem(params, key);
    return cJSON_IsString(item) ? item->valuestring : fallback;
}

// camera.photo: send the result first, then the JPEG as one binary frame straight from PSRAM
// (docs/NODES.md 4.4 / 11.3). No base64, no copy of a 30-60 KB buffer.
static void do_photo(const char *id) {
    const uint8_t *data = NULL;
    size_t len = 0;
    int width = 0, height = 0;
    if (!camera_capture(&data, &len, &width, &height)) {
        send_error(id, "failed", "The camera did not return an image.");
        return;
    }
    char header[224];
    snprintf(header, sizeof(header),
             "{\"type\":\"result\",\"id\":\"%s\",\"ok\":true,"
             "\"data\":{\"mime\":\"image/jpeg\",\"binary\":true,\"width\":%d,\"height\":%d}}",
             id, width, height);
    if (!node_client_send_result_json(header, data, len)) {
        ESP_LOGW(TAG, "could not send the photo (%u bytes)", (unsigned)len);
    }
    camera_release(); // always hand the frame buffer back, even when the send failed
}

void caps_handle_invoke(const char *id, const char *capability, const char *params_json) {
    cJSON *params = cJSON_Parse(params_json ? params_json : "{}");
    if (params == NULL) params = cJSON_CreateObject();

    if (strcmp(capability, "display.text") == 0) {
        caps_show(NULL, string_field(params, "text", ""));
        send_ok(id, "{\"shown\":true}");
    } else if (strcmp(capability, "display.card") == 0) {
        caps_show(string_field(params, "title", ""), string_field(params, "text", ""));
        send_ok(id, "{\"shown\":true}");
    } else if (strcmp(capability, "notify.show") == 0) {
        caps_show(string_field(params, "title", "Sentient"), string_field(params, "text", ""));
        send_ok(id, "{\"shown\":true}");
    } else if (strcmp(capability, "battery") == 0) {
        char data[64];
        snprintf(data, sizeof(data), "{\"battery\":%d,\"charging\":false}", battery_percent());
        send_ok(id, data);
    } else if (strcmp(capability, "camera.photo") == 0) {
        do_photo(id);
    } else {
        send_error(id, "unsupported", "These glasses cannot do that.");
    }
    cJSON_Delete(params);
}
