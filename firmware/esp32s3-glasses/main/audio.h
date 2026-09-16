// I2S capture (INMP441) and playback (MAX98357A). See docs/NODES.md sections 11.1 and 11.2.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

void audio_init(void);

// --- microphone -------------------------------------------------------------------------------
// Reads one chunk of MIC_CHUNK_SAMPLES and converts it to PCM16 mono 16 kHz.
// `out` must hold MIC_CHUNK_SAMPLES int16 values. Returns the number of samples written.
size_t audio_mic_read(int16_t *out, int timeout_ms);
void audio_mic_start(void);
void audio_mic_stop(void);

// --- speaker ----------------------------------------------------------------------------------
// Prepares playback at `sample_rate` (reconfigures the I2S clock only when it changed).
void audio_play_begin(int sample_rate);
// Queues PCM16 mono samples; safe to call with the small chunks a websocket delivers.
void audio_play_write(const uint8_t *pcm, size_t len);
// Waits for the queue to drain (up to timeout_ms), then leaves the amplifier idle.
void audio_play_end(int timeout_ms);
// Drops everything queued: barge-in.
void audio_play_stop(void);
bool audio_is_playing(void);
