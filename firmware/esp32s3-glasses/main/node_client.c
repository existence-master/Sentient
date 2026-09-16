#include "node_client.h"

#include <stdio.h>
#include <string.h>

#include "app_config.h"
#include "audio.h"
#include "battery.h"
#include "camera.h"
#include "caps.h"
#include "cJSON.h"
#include "esp_log.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "esp_websocket_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "storage.h"

static const char *TAG = "node";

static esp_websocket_client_handle_t s_client;
static SemaphoreHandle_t s_send_lock;
static QueueHandle_t s_jobs;
static esp_timer_handle_t s_ping_timer;
static char s_token[96];
static char s_pair_code[16];
static char s_uri[192];
static volatile bool s_connected;
static volatile bool s_fatal; // bad token / revoked: stop retrying, the user must pair again
static int s_backoff_ms = WIFI_BACKOFF_MIN_MS;

// Reassembly of one text message (the engine only sends small JSON to devices).
static char s_rx[WS_TEXT_MAX];
static size_t s_rx_len;

// A binary payload that follows an invoke (audio.pcm): the engine announces it with
// params.binary = true and params.bytes, then sends one binary frame.
static struct {
    bool active;
    char id[24];
    size_t remaining;
} s_incoming;

typedef struct {
    char id[24];
    char capability[40];
    char params[512]; // JSON text; device invokes carry only a handful of small fields
} invoke_job_t;

// ------------------------------------------------------------------ sending
bool node_client_send_json(const char *json) {
    if (s_client == NULL || !esp_websocket_client_is_connected(s_client)) return false;
    xSemaphoreTake(s_send_lock, portMAX_DELAY);
    int sent = esp_websocket_client_send_text(s_client, json, strlen(json), pdMS_TO_TICKS(WS_SEND_TIMEOUT_MS));
    xSemaphoreGive(s_send_lock);
    return sent >= 0;
}

bool node_client_send_binary(const uint8_t *data, size_t len) {
    if (s_client == NULL || !esp_websocket_client_is_connected(s_client)) return false;
    xSemaphoreTake(s_send_lock, portMAX_DELAY);
    int sent = esp_websocket_client_send_bin(s_client, (const char *)data, len, pdMS_TO_TICKS(WS_SEND_TIMEOUT_MS));
    xSemaphoreGive(s_send_lock);
    return sent >= 0;
}

// The result JSON and its payload must not be separated by another message, so both go under one lock.
bool node_client_send_result_json(const char *json, const uint8_t *payload, size_t payload_len) {
    if (s_client == NULL || !esp_websocket_client_is_connected(s_client)) return false;
    xSemaphoreTake(s_send_lock, portMAX_DELAY);
    int sent = esp_websocket_client_send_text(s_client, json, strlen(json), pdMS_TO_TICKS(WS_SEND_TIMEOUT_MS));
    if (sent >= 0 && payload != NULL && payload_len > 0) {
        sent = esp_websocket_client_send_bin(s_client, (const char *)payload, payload_len,
                                             pdMS_TO_TICKS(WS_SEND_TIMEOUT_MS));
    }
    xSemaphoreGive(s_send_lock);
    return sent >= 0;
}

void node_client_send_event(const char *event, const char *data_json) {
    char buf[256];
    snprintf(buf, sizeof(buf), "{\"type\":\"event\",\"event\":\"%s\",\"data\":%s}", event,
             data_json ? data_json : "{}");
    node_client_send_json(buf);
}

void node_client_send_state(int battery, bool charging) {
    if (battery < 0) return;
    char buf[128];
    snprintf(buf, sizeof(buf), "{\"type\":\"state\",\"battery\":%d,\"charging\":%s}", battery,
             charging ? "true" : "false");
    node_client_send_json(buf);
}

bool node_client_connected(void) {
    return s_connected;
}

const char *node_client_token(void) {
    return s_token;
}

// ------------------------------------------------------------------ hello
static void send_hello(void) {
    cJSON *msg = cJSON_CreateObject();
    cJSON_AddStringToObject(msg, "type", "hello");
    cJSON_AddNumberToObject(msg, "protocol", 1);
    cJSON_AddStringToObject(msg, "name", NODE_NAME);
    cJSON_AddStringToObject(msg, "kind", NODE_KIND);
    cJSON_AddStringToObject(msg, "platform", "esp32-s3");
    cJSON_AddStringToObject(msg, "app_version", NODE_APP_VERSION);

    cJSON *caps = cJSON_AddArrayToObject(msg, "capabilities");
    if (camera_available()) cJSON_AddItemToArray(caps, cJSON_CreateString("camera.photo"));
    cJSON_AddItemToArray(caps, cJSON_CreateString("display.text"));
    cJSON_AddItemToArray(caps, cJSON_CreateString("display.card"));
    cJSON_AddItemToArray(caps, cJSON_CreateString("notify.show"));
    cJSON_AddItemToArray(caps, cJSON_CreateString("audio.pcm")); // raw PCM16, no base64 (docs/NODES.md 11.2)
    cJSON_AddItemToArray(caps, cJSON_CreateString("mic.stream"));
    cJSON_AddItemToArray(caps, cJSON_CreateString("button.events"));
    cJSON_AddItemToArray(caps, cJSON_CreateString("battery"));

    if (s_token[0] != '\0') {
        cJSON_AddStringToObject(msg, "token", s_token);
    } else if (s_pair_code[0] != '\0') {
        cJSON_AddStringToObject(msg, "pair_code", s_pair_code);
    }
    int percent = battery_percent();
    if (percent >= 0) {
        cJSON *state = cJSON_AddObjectToObject(msg, "state");
        cJSON_AddNumberToObject(state, "battery", percent);
        cJSON_AddBoolToObject(state, "charging", false);
    }
    char *text = cJSON_PrintUnformatted(msg);
    if (text != NULL) {
        node_client_send_json(text);
        cJSON_free(text);
    }
    cJSON_Delete(msg);
}

static void ping_cb(void *arg) {
    node_client_send_json("{\"type\":\"ping\"}");
}

// ------------------------------------------------------------------ incoming messages
static void handle_welcome(cJSON *msg) {
    const cJSON *token = cJSON_GetObjectItem(msg, "token");
    if (cJSON_IsString(token)) { // only on the first pairing
        strlcpy(s_token, token->valuestring, sizeof(s_token));
        storage_set(KEY_TOKEN, s_token);
        ESP_LOGI(TAG, "paired; token saved to NVS");
        s_pair_code[0] = '\0';
    }
    const cJSON *assistant = cJSON_GetObjectItem(msg, "assistant");
    const cJSON *keepalive = cJSON_GetObjectItem(msg, "keepalive_s");
    int keepalive_s = cJSON_IsNumber(keepalive) ? keepalive->valueint : 60;
    ESP_LOGI(TAG, "connected to %s, keepalive %ds",
             cJSON_IsString(assistant) ? assistant->valuestring : "Sentient", keepalive_s);
    s_connected = true;
    s_backoff_ms = WIFI_BACKOFF_MIN_MS;

    // The engine never pings us, so this timer is what keeps the connection alive.
    if (s_ping_timer == NULL) {
        esp_timer_create_args_t args = {.callback = ping_cb, .name = "node_ping"};
        esp_timer_create(&args, &s_ping_timer);
    } else {
        esp_timer_stop(s_ping_timer);
    }
    esp_timer_start_periodic(s_ping_timer, (uint64_t)keepalive_s * 1000000ULL);
    node_client_send_state(battery_percent(), false);
}

static void handle_error(cJSON *msg) {
    const cJSON *code = cJSON_GetObjectItem(msg, "code");
    const cJSON *text = cJSON_GetObjectItem(msg, "message");
    const char *c = cJSON_IsString(code) ? code->valuestring : "?";
    ESP_LOGE(TAG, "engine error [%s] %s", c, cJSON_IsString(text) ? text->valuestring : "");
    if (strcmp(c, "bad_token") == 0 || strcmp(c, "revoked") == 0) {
        storage_erase(KEY_TOKEN);
        s_token[0] = '\0';
        s_fatal = true;
        ESP_LOGE(TAG, "this device was removed in Sentient: set PAIRING_CODE in app_config.h and reflash");
    } else if (strcmp(c, "pairing_required") == 0 || strcmp(c, "bad_code") == 0 ||
               strcmp(c, "rate_limited") == 0 || strcmp(c, "replaced") == 0) {
        s_fatal = true;
    }
}

// An invoke that carries a payload: the binary frame follows immediately (audio.pcm).
static void handle_binary_invoke(cJSON *msg, const char *id, const char *capability, cJSON *params) {
    const cJSON *bytes = cJSON_GetObjectItem(params, "bytes");
    const cJSON *rate = cJSON_GetObjectItem(params, "sample_rate");
    if (strcmp(capability, "audio.pcm") != 0) {
        char buf[192];
        snprintf(buf, sizeof(buf),
                 "{\"type\":\"result\",\"id\":\"%s\",\"ok\":false,"
                 "\"error\":{\"code\":\"unsupported\",\"message\":\"no binary %s here\"}}",
                 id, capability);
        node_client_send_json(buf);
        return;
    }
    audio_play_begin(cJSON_IsNumber(rate) ? rate->valueint : AMP_DEFAULT_SAMPLE_RATE);
    s_incoming.active = true;
    s_incoming.remaining = cJSON_IsNumber(bytes) ? (size_t)bytes->valuedouble : 0;
    strlcpy(s_incoming.id, id, sizeof(s_incoming.id));
    (void)msg;
}

static void handle_text(const char *text) {
    cJSON *msg = cJSON_Parse(text);
    if (msg == NULL) return;
    const cJSON *type = cJSON_GetObjectItem(msg, "type");
    if (cJSON_IsString(type)) {
        if (strcmp(type->valuestring, "welcome") == 0) {
            handle_welcome(msg);
        } else if (strcmp(type->valuestring, "error") == 0) {
            handle_error(msg);
        } else if (strcmp(type->valuestring, "invoke") == 0) {
            const cJSON *id = cJSON_GetObjectItem(msg, "id");
            const cJSON *cap = cJSON_GetObjectItem(msg, "capability");
            cJSON *params = cJSON_GetObjectItem(msg, "params");
            if (cJSON_IsString(id) && cJSON_IsString(cap)) {
                const cJSON *binary = params ? cJSON_GetObjectItem(params, "binary") : NULL;
                if (cJSON_IsTrue(binary)) {
                    handle_binary_invoke(msg, id->valuestring, cap->valuestring, params);
                } else {
                    // Run it on the worker task: a photo takes ~300 ms and must not block this socket.
                    invoke_job_t job = {0};
                    strlcpy(job.id, id->valuestring, sizeof(job.id));
                    strlcpy(job.capability, cap->valuestring, sizeof(job.capability));
                    char *params_text = params ? cJSON_PrintUnformatted(params) : NULL;
                    strlcpy(job.params, params_text ? params_text : "{}", sizeof(job.params));
                    if (params_text) cJSON_free(params_text);
                    if (xQueueSend(s_jobs, &job, 0) != pdTRUE) ESP_LOGW(TAG, "invoke queue full");
                }
            }
        }
        // "pong" needs no action: any traffic counts as activity.
    }
    cJSON_Delete(msg);
}

static void handle_binary(const char *data, size_t len) {
    if (!s_incoming.active) return;
    audio_play_write((const uint8_t *)data, len);
    if (len >= s_incoming.remaining) {
        s_incoming.remaining = 0;
        s_incoming.active = false;
        char buf[128];
        snprintf(buf, sizeof(buf), "{\"type\":\"result\",\"id\":\"%s\",\"ok\":true,\"data\":{\"playing\":true}}",
                 s_incoming.id);
        node_client_send_json(buf);
    } else {
        s_incoming.remaining -= len;
    }
}

static void on_ws_event(void *arg, esp_event_base_t base, int32_t event_id, void *event_data) {
    esp_websocket_event_data_t *d = (esp_websocket_event_data_t *)event_data;
    switch (event_id) {
        case WEBSOCKET_EVENT_CONNECTED:
            ESP_LOGI(TAG, "socket open, saying hello");
            send_hello();
            break;
        case WEBSOCKET_EVENT_DISCONNECTED:
        case WEBSOCKET_EVENT_CLOSED:
            s_connected = false;
            s_incoming.active = false;
            if (s_ping_timer) esp_timer_stop(s_ping_timer);
            ESP_LOGW(TAG, "socket closed");
            break;
        case WEBSOCKET_EVENT_DATA:
            if (d->op_code == 0x08) break; // close frame: the DISCONNECTED event follows
            if (d->op_code == 0x09 || d->op_code == 0x0A) break; // protocol ping/pong
            if (d->op_code == 0x02 || (d->op_code == 0x00 && s_incoming.active)) {
                handle_binary(d->data_ptr, d->data_len);
                break;
            }
            if (d->payload_offset == 0) s_rx_len = 0;
            if (s_rx_len + d->data_len < sizeof(s_rx)) {
                memcpy(s_rx + s_rx_len, d->data_ptr, d->data_len);
                s_rx_len += d->data_len;
            } else {
                ESP_LOGW(TAG, "dropping a message larger than %d bytes", WS_TEXT_MAX);
                s_rx_len = 0;
                break;
            }
            if (s_rx_len >= (size_t)d->payload_len) {
                s_rx[s_rx_len] = '\0';
                handle_text(s_rx);
                s_rx_len = 0;
            }
            break;
        default:
            break;
    }
}

// ------------------------------------------------------------------ tasks
static void worker_task(void *arg) {
    invoke_job_t job;
    while (true) {
        if (xQueueReceive(s_jobs, &job, portMAX_DELAY) == pdTRUE) {
            caps_handle_invoke(job.id, job.capability, job.params);
        }
    }
}

// Our own reconnect curve (the library's fixed retry is disabled), matching docs/NODES.md section 8.
static void supervisor_task(void *arg) {
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        if (s_fatal) continue;
        if (esp_websocket_client_is_connected(s_client)) continue;
        int jitter = (int)(esp_random() % (uint32_t)(s_backoff_ms / 2 + 1));
        vTaskDelay(pdMS_TO_TICKS(s_backoff_ms + jitter));
        if (s_fatal || esp_websocket_client_is_connected(s_client)) continue;
        ESP_LOGI(TAG, "reconnecting to %s", s_uri);
        esp_websocket_client_stop(s_client);
        esp_websocket_client_start(s_client);
        s_backoff_ms = s_backoff_ms * 2 > WIFI_BACKOFF_MAX_MS ? WIFI_BACKOFF_MAX_MS : s_backoff_ms * 2;
    }
}

bool node_client_start(const char *host, int port, const char *path, const char *cert_pem, const char *pair_code) {
    s_send_lock = xSemaphoreCreateMutex();
    s_jobs = xQueueCreate(4, sizeof(invoke_job_t));
    storage_get(KEY_TOKEN, s_token, sizeof(s_token));
    strlcpy(s_pair_code, pair_code ? pair_code : "", sizeof(s_pair_code));
    if (s_token[0] == '\0' && s_pair_code[0] == '\0') {
        ESP_LOGE(TAG, "no token and no pairing code: set PAIRING_CODE in app_config.h");
        return false;
    }
    snprintf(s_uri, sizeof(s_uri), "wss://%s:%d%s", host, port, (path && path[0]) ? path : "/ws/node");

    esp_websocket_client_config_t cfg = {
        .uri = s_uri,
        .cert_pem = cert_pem,                 // the pinned engine certificate (docs/NODES.md 11.4)
        .skip_cert_common_name_check = true,  // the certificate is issued to "Sentient local device link"
        .buffer_size = WS_BUFFER_SIZE,
        .task_stack = NODE_TASK_STACK,
        .disable_auto_reconnect = true,       // supervisor_task owns the backoff
        .network_timeout_ms = 10000,
        .ping_interval_sec = 0,               // app-level pings instead, so the radio wakes once
    };
    s_client = esp_websocket_client_init(&cfg);
    if (s_client == NULL) {
        ESP_LOGE(TAG, "could not create the websocket client");
        return false;
    }
    esp_websocket_register_events(s_client, WEBSOCKET_EVENT_ANY, on_ws_event, NULL);
    esp_err_t err = esp_websocket_client_start(s_client);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "websocket start failed: %s", esp_err_to_name(err));
        return false;
    }
    xTaskCreate(worker_task, "node_worker", 5120, NULL, 5, NULL);
    xTaskCreate(supervisor_task, "node_super", 3072, NULL, 4, NULL);
    return true;
}
