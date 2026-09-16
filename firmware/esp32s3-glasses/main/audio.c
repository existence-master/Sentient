#include "audio.h"

#include <string.h>

#include "app_config.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/ringbuf.h"
#include "freertos/task.h"

static const char *TAG = "audio";

static i2s_chan_handle_t s_rx; // INMP441
static i2s_chan_handle_t s_tx; // MAX98357A
static RingbufHandle_t s_play_ring;
static int s_play_rate = AMP_DEFAULT_SAMPLE_RATE;
static volatile bool s_playing;
static volatile bool s_mic_on;

// The INMP441 sends 24-bit samples MSB-aligned in a 32-bit slot, so capture is 32-bit and
// converted here. Playback is plain 16-bit.
static int32_t s_mic_raw[MIC_CHUNK_SAMPLES];

// ------------------------------------------------------------------ playback task
// Pulls from the ring buffer and pushes to I2S, so websocket chunks never block the socket task.
static void play_task(void *arg) {
    while (true) {
        size_t len = 0;
        uint8_t *data = (uint8_t *)xRingbufferReceiveUpTo(s_play_ring, &len, pdMS_TO_TICKS(100), 1024);
        if (data == NULL) {
            if (s_playing && xRingbufferGetCurFreeSize(s_play_ring) == AUDIO_RING_BYTES) s_playing = false;
            continue;
        }
        size_t written = 0;
        s_playing = true;
        i2s_channel_write(s_tx, data, len, &written, pdMS_TO_TICKS(1000));
        vRingbufferReturnItem(s_play_ring, data);
    }
}

void audio_init(void) {
    // ---- microphone: I2S0 RX, Philips, mono LEFT (INMP441 with L/R tied to GND) ----
    i2s_chan_config_t rx_cfg = I2S_CHANNEL_DEFAULT_CONFIG(MIC_I2S_PORT, I2S_ROLE_MASTER);
    rx_cfg.dma_desc_num = MIC_DMA_BUFFERS;
    rx_cfg.dma_frame_num = MIC_DMA_FRAMES;
    rx_cfg.auto_clear = true;
    ESP_ERROR_CHECK(i2s_new_channel(&rx_cfg, NULL, &s_rx));

    i2s_std_config_t rx_std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(MIC_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = MIC_PIN_SCK,
            .ws = MIC_PIN_WS,
            .dout = I2S_GPIO_UNUSED,
            .din = MIC_PIN_SD,
            .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
        },
    };
    rx_std.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx, &rx_std));

    // ---- speaker: I2S1 TX, Philips, 16-bit mono (the same sample goes to both slots) ----
    i2s_chan_config_t tx_cfg = I2S_CHANNEL_DEFAULT_CONFIG(AMP_I2S_PORT, I2S_ROLE_MASTER);
    tx_cfg.dma_desc_num = AMP_DMA_BUFFERS;
    tx_cfg.dma_frame_num = AMP_DMA_FRAMES;
    tx_cfg.auto_clear = true; // send silence instead of repeating the last buffer on underrun
    ESP_ERROR_CHECK(i2s_new_channel(&tx_cfg, &s_tx, NULL));

    i2s_std_config_t tx_std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(AMP_DEFAULT_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = AMP_PIN_BCLK,
            .ws = AMP_PIN_LRC,
            .dout = AMP_PIN_DIN,
            .din = I2S_GPIO_UNUSED,
            .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_tx, &tx_std));
    ESP_ERROR_CHECK(i2s_channel_enable(s_tx));

    s_play_ring = xRingbufferCreate(AUDIO_RING_BYTES, RINGBUF_TYPE_BYTEBUF);
    ESP_ERROR_CHECK(s_play_ring == NULL ? ESP_ERR_NO_MEM : ESP_OK);
    xTaskCreate(play_task, "audio_play", 4096, NULL, 6, NULL);
    ESP_LOGI(TAG, "I2S ready: mic %d Hz on port %d, amp on port %d", MIC_SAMPLE_RATE, MIC_I2S_PORT, AMP_I2S_PORT);
}

// ------------------------------------------------------------------ microphone
void audio_mic_start(void) {
    if (s_mic_on) return;
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx));
    s_mic_on = true;
}

void audio_mic_stop(void) {
    if (!s_mic_on) return;
    i2s_channel_disable(s_rx);
    s_mic_on = false;
}

size_t audio_mic_read(int16_t *out, int timeout_ms) {
    if (!s_mic_on) return 0;
    size_t got = 0;
    if (i2s_channel_read(s_rx, s_mic_raw, sizeof(s_mic_raw), &got, pdMS_TO_TICKS(timeout_ms)) != ESP_OK) return 0;
    size_t samples = got / sizeof(int32_t);
    for (size_t i = 0; i < samples; i++) {
        // >> 16 would be the plain 24-in-32 to 16-bit conversion; MIC_GAIN_SHIFT (14) is louder.
        int32_t v = s_mic_raw[i] >> MIC_GAIN_SHIFT;
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        out[i] = (int16_t)v;
    }
    return samples;
}

// ------------------------------------------------------------------ playback
void audio_play_begin(int sample_rate) {
    if (sample_rate <= 0) sample_rate = AMP_DEFAULT_SAMPLE_RATE;
    if (sample_rate != s_play_rate) {
        // The engine's TTS may be 24 kHz while voice replies are 16 kHz: retune the clock.
        i2s_std_clk_config_t clk = I2S_STD_CLK_DEFAULT_CONFIG(sample_rate);
        ESP_ERROR_CHECK(i2s_channel_disable(s_tx));
        ESP_ERROR_CHECK(i2s_channel_reconfig_std_clock(s_tx, &clk));
        ESP_ERROR_CHECK(i2s_channel_enable(s_tx));
        s_play_rate = sample_rate;
        ESP_LOGI(TAG, "playback at %d Hz", sample_rate);
    }
    s_playing = true;
}

void audio_play_write(const uint8_t *pcm, size_t len) {
    if (pcm == NULL || len == 0) return;
    // Drop rather than block the socket task if the network outruns the speaker.
    if (xRingbufferSend(s_play_ring, pcm, len, pdMS_TO_TICKS(200)) != pdTRUE) {
        ESP_LOGW(TAG, "playback buffer full, dropped %u bytes", (unsigned)len);
    }
}

void audio_play_end(int timeout_ms) {
    int waited = 0;
    while (waited < timeout_ms && xRingbufferGetCurFreeSize(s_play_ring) < AUDIO_RING_BYTES) {
        vTaskDelay(pdMS_TO_TICKS(20));
        waited += 20;
    }
    s_playing = false;
}

void audio_play_stop(void) {
    size_t len = 0;
    void *data;
    while ((data = xRingbufferReceiveUpTo(s_play_ring, &len, 0, 1024)) != NULL) {
        vRingbufferReturnItem(s_play_ring, data);
    }
    s_playing = false;
}

bool audio_is_playing(void) {
    return s_playing;
}
