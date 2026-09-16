#include "voice.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "app_config.h"
#include "audio.h"
#include "cJSON.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_websocket_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "node_client.h"

static const char *TAG = "voice";

static esp_websocket_client_handle_t s_client;
static SemaphoreHandle_t s_lock;
static char s_host[64];
static const char *s_cert;
static int s_port;

static volatile bool s_streaming;  // microphone frames are going out
static volatile bool s_ready;      // the engine answered `ready`
static volatile bool s_speaking;   // a reply is playing
static volatile bool s_expect_pcm; // an `audio` header arrived; the next binary frame is its audio
static int s_reply_rate = MIC_SAMPLE_RATE;
static int64_t s_last_activity_us;

static char s_rx[WS_TEXT_MAX];
static size_t s_rx_len;

static int16_t s_mic_chunk[MIC_CHUNK_SAMPLES];

static bool send_text(const char *json) {
    if (s_client == NULL || !esp_websocket_client_is_connected(s_client)) return false;
    xSemaphoreTake(s_lock, portMAX_DELAY);
    int sent = esp_websocket_client_send_text(s_client, json, strlen(json), pdMS_TO_TICKS(WS_SEND_TIMEOUT_MS));
    xSemaphoreGive(s_lock);
    return sent >= 0;
}

// The engine may send a complete WAV (default) or raw PCM16 when it supports audio_format "pcm16".
// Both start playing immediately; a WAV just has a 44-byte header to skip, and its header carries
// the real sample rate.
static void play_chunk(const char *data, size_t len, bool first_chunk) {
    if (first_chunk && len > 44 && memcmp(data, "RIFF", 4) == 0) {
        int rate = (int)((uint8_t)data[24] | ((uint8_t)data[25] << 8) | ((uint8_t)data[26] << 16) |
                         ((uint8_t)data[27] << 24));
        audio_play_begin(rate > 0 ? rate : s_reply_rate);
        audio_play_write((const uint8_t *)data + 44, len - 44);
        return;
    }
    if (first_chunk) audio_play_begin(s_reply_rate);
    audio_play_write((const uint8_t *)data, len);
}

static void handle_text(const char *text) {
    cJSON *msg = cJSON_Parse(text);
    if (msg == NULL) return;
    const cJSON *type = cJSON_GetObjectItem(msg, "type");
    if (cJSON_IsString(type)) {
        const char *t = type->valuestring;
        if (strcmp(t, "ready") == 0) {
            const cJSON *rate = cJSON_GetObjectItem(msg, "sample_rate");
            if (cJSON_IsNumber(rate)) s_reply_rate = rate->valueint;
            s_ready = true;
            ESP_LOGI(TAG, "voice ready (replies at %d Hz)", s_reply_rate);
        } else if (strcmp(t, "audio") == 0) {
            s_expect_pcm = true; // exactly one binary frame follows this header
            s_speaking = true;
        } else if (strcmp(t, "audio_end") == 0) {
            audio_play_end(3000);
            s_speaking = false;
        } else if (strcmp(t, "transcript") == 0) {
            const cJSON *value = cJSON_GetObjectItem(msg, "text");
            if (cJSON_IsString(value)) ESP_LOGI(TAG, "you said: %s", value->valuestring);
        } else if (strcmp(t, "error") == 0) {
            const cJSON *m = cJSON_GetObjectItem(msg, "message");
            ESP_LOGW(TAG, "voice error: %s", cJSON_IsString(m) ? m->valuestring : "");
        }
    }
    cJSON_Delete(msg);
}

static void on_ws_event(void *arg, esp_event_base_t base, int32_t event_id, void *event_data) {
    esp_websocket_event_data_t *d = (esp_websocket_event_data_t *)event_data;
    switch (event_id) {
        case WEBSOCKET_EVENT_CONNECTED: {
            char start[160];
            // audio_format pcm16 keeps the ESP32 out of WAV parsing; the engine falls back to wav
            // when it does not support it yet, and play_chunk() handles both.
            snprintf(start, sizeof(start),
                     "{\"type\":\"start\",\"channel\":\"glasses\",\"sample_rate\":%d,\"audio_format\":\"pcm16\"}",
                     MIC_SAMPLE_RATE);
            send_text(start);
            break;
        }
        case WEBSOCKET_EVENT_DISCONNECTED:
        case WEBSOCKET_EVENT_CLOSED:
            s_ready = s_streaming = s_speaking = s_expect_pcm = false;
            break;
        case WEBSOCKET_EVENT_DATA: {
            if (d->op_code == 0x08 || d->op_code == 0x09 || d->op_code == 0x0A) break;
            if (d->op_code == 0x02 || (d->op_code == 0x00 && s_expect_pcm)) {
                play_chunk(d->data_ptr, d->data_len, d->payload_offset == 0);
                if (d->payload_offset + d->data_len >= (size_t)d->payload_len) s_expect_pcm = false;
                break;
            }
            if (d->payload_offset == 0) s_rx_len = 0;
            if (s_rx_len + d->data_len < sizeof(s_rx)) {
                memcpy(s_rx + s_rx_len, d->data_ptr, d->data_len);
                s_rx_len += d->data_len;
            } else {
                s_rx_len = 0;
                break;
            }
            if (s_rx_len >= (size_t)d->payload_len) {
                s_rx[s_rx_len] = '\0';
                handle_text(s_rx);
                s_rx_len = 0;
            }
            break;
        }
        default:
            break;
    }
}

// Reads the microphone and sends 40 ms PCM16 frames while the button is held.
static void mic_task(void *arg) {
    while (true) {
        if (!s_streaming) {
            // Close an idle socket so the radio and the TLS session are not kept alive for nothing.
            if (s_client != NULL && esp_websocket_client_is_connected(s_client) && !s_speaking &&
                esp_timer_get_time() - s_last_activity_us > (int64_t)VOICE_IDLE_CLOSE_MS * 1000) {
                ESP_LOGI(TAG, "closing the idle voice socket");
                send_text("{\"type\":\"stop\"}");
                esp_websocket_client_close(s_client, pdMS_TO_TICKS(1000));
            }
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }
        size_t samples = audio_mic_read(s_mic_chunk, 200);
        if (samples == 0) continue;
        if (s_client != NULL && esp_websocket_client_is_connected(s_client)) {
            xSemaphoreTake(s_lock, portMAX_DELAY);
            esp_websocket_client_send_bin(s_client, (const char *)s_mic_chunk, samples * sizeof(int16_t),
                                          pdMS_TO_TICKS(WS_SEND_TIMEOUT_MS));
            xSemaphoreGive(s_lock);
            s_last_activity_us = esp_timer_get_time();
        }
    }
}

void voice_init(const char *host, int port, const char *cert_pem) {
    strlcpy(s_host, host, sizeof(s_host));
    s_port = port;
    s_cert = cert_pem;
    s_lock = xSemaphoreCreateMutex();
    xTaskCreate(mic_task, "voice_mic", 4096, NULL, 6, NULL);
}

static bool ensure_socket(void) {
    if (s_client != NULL && esp_websocket_client_is_connected(s_client)) return true;
    const char *token = node_client_token();
    if (token == NULL || token[0] == '\0') {
        ESP_LOGW(TAG, "not paired yet");
        return false;
    }
    if (s_client == NULL) {
        static char uri[256];
        snprintf(uri, sizeof(uri), "wss://%s:%d/ws/voice?node_token=%s", s_host, s_port, token);
        esp_websocket_client_config_t cfg = {
            .uri = uri,
            .cert_pem = s_cert, // same pinned certificate as the node socket
            .skip_cert_common_name_check = true,
            .buffer_size = WS_BUFFER_SIZE,
            .task_stack = NODE_TASK_STACK,
            .disable_auto_reconnect = true,
            .network_timeout_ms = 10000,
            .ping_interval_sec = 0,
        };
        s_client = esp_websocket_client_init(&cfg);
        if (s_client == NULL) return false;
        esp_websocket_register_events(s_client, WEBSOCKET_EVENT_ANY, on_ws_event, NULL);
    }
    s_ready = false;
    if (esp_websocket_client_start(s_client) != ESP_OK) return false;
    for (int i = 0; i < 100 && !s_ready; i++) vTaskDelay(pdMS_TO_TICKS(50)); // up to 5 s for `ready`
    return s_ready;
}

void voice_press(void) {
    s_last_activity_us = esp_timer_get_time();
    if (s_speaking) voice_interrupt(); // barge-in
    if (!ensure_socket()) return;
    audio_mic_start();
    s_streaming = true;
    ESP_LOGI(TAG, "listening");
}

void voice_release(void) {
    if (!s_streaming) return;
    s_streaming = false;
    audio_mic_stop();
    send_text("{\"type\":\"end_utterance\"}");
    s_last_activity_us = esp_timer_get_time();
    ESP_LOGI(TAG, "utterance sent");
}

void voice_interrupt(void) {
    audio_play_stop();
    s_speaking = false;
    send_text("{\"type\":\"interrupt\"}");
}

bool voice_is_speaking(void) {
    return s_speaking || audio_is_playing();
}
