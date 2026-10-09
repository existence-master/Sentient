# Sentient device protocol (nodes), version 1

This is the protocol a device speaks to connect to Sentient: smart glasses firmware (ESP32-S3 class), a phone
app, a watch, or the built-in web app at `/node/`. It is plain JSON over one WebSocket plus the voice socket.
The REST API the desktop app uses is in `docs/API.md` section 13.

A working client in Python lives in `sentient/nodes/reference.py` (`sentient node --help`); the browser client
is `sentient/nodes/web/app.js`.

---

## 1. Where to connect

| Where | Address | Who |
|---|---|---|
| LAN listener (when *Settings › Devices › Accept devices on your network* is on) | `wss://<lan-ip>:7778/ws/node` (port = `nodes.lan_port`) | glasses, phones on the same network |
| Loopback gateway | `ws://127.0.0.1:<gateway port>/ws/node` | the desktop app, developer tools on the same PC |

The LAN listener serves **only**: `WS /ws/node`, `WS /ws/voice`, `GET /node/` (web app) and
`POST /api/nodes/upload`. Nothing else of the engine is reachable from the network.

### Discovery (mDNS / DNS-SD)
Sentient announces `_sentient._tcp.local.` (instance `Sentient on <hostname>`), IPv4 A records, and TXT:

| key | example | meaning |
|---|---|---|
| `version` | `3.0.0a0` | engine version |
| `protocol` | `1` | this document's version |
| `fingerprint` | `3f9a…` (64 hex) | SHA-256 of the TLS certificate (see §2) |
| `port` | `7778` | same as the SRV port |
| `path` | `/ws/node` | node socket path |
| `tls` | `1` | always TLS on the LAN |

ESP-IDF: `mdns_query_ptr("_sentient", "_tcp", 3000, 5, &results)`. Arduino: `MDNS.queryService("sentient", "tcp")`.
If discovery fails, let the user enter the address, or take it from the pairing QR link (§3).

## 2. TLS and certificate pinning

The LAN listener uses a self-signed certificate (ECDSA P-256, stored in `~/.sentient/tls/`, reused for years so the
fingerprint stays stable). Devices must **not** use hostname or CA verification. Instead:

1. Complete the TLS handshake with verification disabled.
2. Compute SHA-256 over the server certificate in DER form. Compare it with the expected fingerprint
   (64 lowercase hex characters; compare case-insensitively and ignore `:` separators).
3. If it does not match, close the connection **before sending anything** and tell the user.

Where the expected fingerprint comes from, in order: the value you stored after the first successful pairing;
the `fp` parameter of the pairing link; the TXT record (trust-on-first-use; weakest). The desktop shows it
next to the pairing code (`GET /api/nodes/lan` → `fingerprint`).

mbedTLS: set `MBEDTLS_SSL_VERIFY_OPTIONAL`, then after `mbedtls_ssl_handshake` take
`mbedtls_ssl_get_peer_cert(&ssl)->raw` and hash it with `mbedtls_sha256`.

## 3. Pairing

1. The user opens *Devices › Pair a device* in Sentient. The desktop calls `POST /api/nodes/pairing` and shows:
   - a **6-digit code** (valid 10 minutes, single use),
   - a QR code of `web_url` (`https://<ip>:7778/node/#code=123456`) that a phone camera opens directly,
   - the link `sentient://pair?url=wss%3A%2F%2F192.168.1.20%3A7778%2Fws%2Fnode&code=123456&fp=<fingerprint>` for apps
     and firmware that can scan (glasses can read this QR with their camera).
2. The device connects and sends `hello` with `pair_code`.
3. The engine answers `welcome` with a `token`. **Store it** (NVS / keychain / localStorage) together with the URL and
   fingerprint. Every later connection sends `token` instead of `pair_code`.

Wrong codes are rate-limited: 5 failures per minute per client address, and 30 failures in 10 minutes cancel every
outstanding code. A device removed in Sentient has its token revoked.

## 4. Messages

Every message is one JSON text frame with a `type`. Unknown fields must be ignored (the engine may add fields).
Binary frames are only used as described in §4.4.

### 4.1 `hello` → `welcome` | `error`
The first message after connecting, within 15 seconds.

```json
{"type": "hello", "protocol": 1, "name": "Sarthak's glasses", "kind": "glasses",
 "platform": "esp32-s3", "app_version": "0.4.1",
 "capabilities": ["camera.photo", "display.text", "speak", "audio.play", "button.events", "battery"],
 "token": "…", "state": {"battery": 81, "charging": false, "worn": true}}
```

- `kind`: `phone` | `glasses` | `watch` | `custom` (`desktop` is reserved for the desktop app).
- Exactly one of `token` / `pair_code` (a `pair_code` is used if the token is rejected).
- `state` is optional (same fields as §4.3).

```json
{"type": "welcome", "protocol": 1, "node_id": "9c1e0f3a2b4d6e71", "name": "Sarthak's glasses",
 "assistant": "Sentient", "server_version": "3.0.0a0", "keepalive_s": 60, "idle_timeout_s": 180,
 "stopped": false, "token": "only-on-first-pairing"}
```

`stopped` is true while the user has pressed **Stop everything** (§4.5).

The desktop app itself connects on the loopback gateway with `?token=<gateway token>`; it becomes node `desktop`
without pairing. The gateway token is refused on the LAN listener.

### 4.2 `invoke` → `result`
The engine asks the device to do something.

```json
{"type": "invoke", "id": "a1b2c3d4e5f6", "capability": "display.text", "params": {"text": "Turn left in 50 m"}, "timeout_ms": 20000}
```

Answer once, with the same `id`, before `timeout_ms` (late answers are dropped):

```json
{"type": "result", "id": "a1b2c3d4e5f6", "ok": true, "data": {"shown": true}}
{"type": "result", "id": "a1b2c3d4e5f6", "ok": false, "error": {"code": "permission_denied", "message": "Camera permission was denied."}}
```

`error` may also be a plain string. Invokes can overlap; answer each independently.

### 4.3 Device → engine: `event`, `state`, `ping`
```json
{"type": "event", "event": "button", "data": {"action": "press"}}
{"type": "event", "event": "wake", "data": {"phrase": "hey sentient"}}
{"type": "state", "battery": 64, "charging": false, "worn": true}
{"type": "ping", "ts": 12345}          → {"type": "pong", "ts": 12345}
```

- `event` names: `button`, `wake`, `gesture`, `battery` (`data: {level, charging}`), `presence` (`data: {worn}`),
  `notification_action` (`data: {id, action}`); any `[a-z0-9_.-]{1,40}` name is accepted and published to the
  desktop as `node.event`.
- `state`: `battery` 0..100, `charging` and `worn` booleans; send when they change.
- Sending `hello` again on an open connection updates the capability list.

### 4.4 Large payloads (photos)
Three equivalent ways to return `camera.photo` / `screen.capture` data, pick what suits the device:

1. **Inline base64**: `"data": {"mime": "image/jpeg", "base64": "…"}` (simple; ~33% bigger; phones and PCs).
2. **Binary follow frame** (best for microcontrollers): send
   `{"type": "result", "id": "…", "ok": true, "data": {"mime": "image/jpeg", "binary": true, "width": 1280, "height": 720}}`
   and immediately afterwards **one binary frame** with the raw JPEG bytes.
3. **HTTP upload**: `POST /api/nodes/upload` with `Authorization: Bearer <node token>`, `Content-Type: image/jpeg` and the
   raw bytes as the body (or multipart field `file`), max 20 MB → `{"upload_id", "mime", "size"}`; then answer
   `{"ok": true, "data": {"mime": "image/jpeg", "upload_id": "…"}}`. Uploads are single use and expire after an hour.

WebSocket frames up to 32 MB are accepted on the LAN listener.

The engine sends payloads the same way: an `invoke` whose `params` contain `"binary": true` and `"bytes": <length>` is
followed by exactly one binary frame with that payload. This is how `audio.pcm` delivers speech (§5), so a
microcontroller never has to hold a base64 string of a whole sentence in RAM.

### 4.5 Stop everything
A paired device can stop all of Sentient's work at once, and resume it. The engine handles this itself, never
through the model.

```json
{"type": "stop_all"}                   → {"type": "stop_state", "stopped": true, "stopped_at": "2026-10-09T10:00:00+00:00", "source": "device"}
{"type": "resume"}                     → {"type": "stop_state", "stopped": false, "stopped_at": null, "source": "device"}
```

`stop_state` is also sent to every connected device when the user stops or resumes anywhere else (desktop, tray,
Telegram, Discord). A phone app can show a Stop or Resume button from it; other devices may ignore it.

## 5. Capabilities

| capability | params | data |
|---|---|---|
| `camera.photo` | `facing?: "back"\|"front"`, `max_width?` | `{mime, base64}` (or §4.4), `width?`, `height?` |
| `screen.capture` | | `{mime, base64}` |
| `location.get` | `timeout_ms?` | `{lat, lon, accuracy_m, label?}` |
| `notify.show` | `title?`, `text`, `tag?` | `{shown}` |
| `display.text` | `text`, `duration_ms?` | `{shown}` |
| `display.card` | `title?`, `text`, `image_base64?`, `image_mime?`, `duration_ms?` | `{shown}` |
| `audio.play` | `mime: "audio/wav"`, `base64`, `text?` (what is said) | `{playing}` |
| `audio.pcm` | `format: "pcm16"`, `sample_rate`, `channels`, `text?`, `binary: true`, `bytes` + one binary frame | `{playing}` |
| `speak` | `text`, `lang?` | `{spoken}` (on-device TTS) |
| `mic.stream` | | advertises that the device talks over `/ws/voice` (§6) |
| `clipboard.read` / `clipboard.write` | `text` (write) | `{text}` (read) |
| `button.events` | | advertises that the device sends `button` events |
| `battery` | | `{battery, charging}` |

`audio.play` WAVs are PCM16 mono with a 44-byte header; a speaker driver can skip the header and play the samples at
the rate in the header (usually 24 kHz or 16 kHz). For speech the engine picks, in order, `speak` (on-device TTS),
`audio.pcm` (raw frame, best for microcontrollers) and `audio.play` (base64 WAV).

### Recommended set for glasses
`camera.photo` (binary follow frame, JPEG ≤ 1280 px), `display.text` (and `display.card` with a screen),
`audio.pcm`, `mic.stream`, `button.events`, `battery`. Skip on-device `speak` unless the chip has TTS: the engine
speaks better. A tiny display should show the last `display.text` until the next one or `duration_ms`.

Agent tools use these capabilities: `device_take_photo`, `device_capture_screen`, `device_get_location`,
`device_notify`, `device_display`, `device_speak`. Taking a photo or a screenshot asks the user for approval unless
`nodes.camera_requires_approval` is off.

## 6. Voice from a device

Open `wss://<host>:7778/ws/voice?node_token=<token>` (same TLS pinning) and follow `docs/API.md` section 9:

```json
{"type": "start", "channel": "glasses", "sample_rate": 16000, "audio_format": "pcm16", "mode": "conversation"}
```

- Microphone: binary frames of **PCM16 little-endian mono** at `sample_rate` (16000 recommended), 20-100 ms per frame.
  On button release send `{"type": "end_utterance"}`; without a button the server's voice activity detection decides.
- Replies: each `{"type": "audio", "format": "wav"|"pcm16", "sentence_index", "text"}` header is followed by exactly one
  binary frame with that sentence's audio. With `audio_format: "pcm16"` the frame is raw PCM16 mono (no header) at the
  `sample_rate` announced in `ready`. `audio_end` closes the turn.
- Barge-in: `{"type": "interrupt"}` when the user presses the button while the assistant speaks.
- Hands-free: `mode: "wake"` listens for the wake word first (section 16).
- Keep the voice socket closed while idle to save power; open it on button press (it is ready within one round trip).

## 7. Errors and close codes

`{"type": "error", "code": "…", "message": "…"}`; for authentication errors the engine closes right after.

| code | close | what to do |
|---|---|---|
| `pairing_required` | 4401 | show "pair this device" |
| `bad_code` | 4401 | wrong or expired code: ask for a new code |
| `bad_token` | 4401 | token unknown: delete it, pair again |
| `revoked` | 4401 | removed in Sentient: delete the token, pair again |
| `rate_limited` | 4429 | wait at least 60 s before trying a code again |
| `disabled` | 4403 | devices are off in Sentient: retry every few minutes |
| `protocol` | 4400 (on hello) | bug in the device: first message must be `hello`; later protocol errors do not close |
| `replaced` | 4409 | the same device connected again elsewhere: do not reconnect automatically |
| `unknown_type` | none | ignore |

Invoke failures seen by the engine (`POST /api/nodes/{id}/invoke`) add: `offline`, `unsupported`, `timeout`,
`not_allowed`, `bad_upload`, plus whatever `error.code` the device returned.

## 8. Reconnect, keepalive and battery

- Reconnect with exponential backoff: 1 s, 2 s, 4 s … capped at 30 s, plus 0-50% random jitter. Reset after a
  `welcome`. Do not reconnect after `bad_code`, `rate_limited` (wait), `replaced`, `bad_token`/`revoked` (re-pair).
- The engine never pings devices (no WebSocket protocol pings on the LAN listener), so the radio can sleep. The device
  sends `{"type": "ping"}` every `keepalive_s` (default 60 s). A device silent for `idle_timeout_s` (default 180 s) is
  considered gone and its socket is closed.
- Any message counts as activity; a device sending `state` regularly does not need extra pings.
- Wi-Fi modem sleep between pings is fine; keep TCP keepalive off.

## 9. Example transcripts

### First pairing and a photo (glasses)
```
→ {"type":"hello","protocol":1,"name":"Glasses","kind":"glasses","platform":"esp32-s3","app_version":"0.1",
   "capabilities":["camera.photo","display.text","audio.play","mic.stream","button.events","battery"],"pair_code":"482913"}
← {"type":"welcome","protocol":1,"node_id":"9c1e0f3a2b4d6e71","name":"Glasses","assistant":"Sentient",
   "server_version":"3.0.0a0","keepalive_s":60,"idle_timeout_s":180,"token":"Qm9…"}
→ {"type":"state","battery":92,"charging":false,"worn":true}
← {"type":"invoke","id":"5e2f7a90c1d3","capability":"camera.photo","params":{"facing":"back"},"timeout_ms":20000}
→ {"type":"result","id":"5e2f7a90c1d3","ok":true,"data":{"mime":"image/jpeg","binary":true,"width":1280,"height":720}}
→ <binary: 184 213 bytes of JPEG>
← {"type":"invoke","id":"77ab01c2d3e4","capability":"display.text","params":{"text":"That is a Monstera plant."},"timeout_ms":20000}
→ {"type":"result","id":"77ab01c2d3e4","ok":true,"data":{"shown":true}}
→ {"type":"ping"}
← {"type":"pong","ts":null}
```

### Reconnect with a revoked token
```
→ {"type":"hello","protocol":1,"name":"Glasses","kind":"glasses","capabilities":[…],"token":"Qm9…"}
← {"type":"error","code":"bad_token","message":"This device is no longer paired. Pair it again with a new code."}
← <close 4401>
```

### Push-to-talk (voice socket)
```
→ {"type":"start","channel":"glasses","sample_rate":16000,"audio_format":"pcm16"}
← {"type":"ready","session_id":"…","stt":"faster_whisper","tts":"kokoro","sample_rate":16000}
← {"type":"state","state":"listening","session_id":"…"}
→ <binary PCM16 frames while the button is held>
→ {"type":"end_utterance"}
← {"type":"state","state":"transcribing",…}  {"type":"transcript","text":"what's the weather","final":true,…}
← {"type":"state","state":"thinking",…}  {"type":"text_delta","text":"It is 24 degrees"…}
← {"type":"audio","format":"pcm16","sentence_index":0,"text":"It is 24 degrees and sunny."} → <binary PCM16>
← {"type":"audio_end",…}  {"type":"state","state":"listening",…}
```

## 10. Testing a device without hardware

- Web app: open `http://127.0.0.1:<gateway port>/node/` on the PC (camera and microphone work on localhost) or the
  `https://<lan-ip>:7778/node/` link on a phone (accept the certificate warning once).
- Reference node: `sentient node --url "sentient://pair?url=…&code=…&fp=…"` or
  `sentient node --url wss://192.168.1.20:7778/ws/node --code 123456 --fingerprint <fp> --camera 0`.
  Press Enter to send a button press, type `b 40` to report 40% battery, `q` to quit. `--insecure` skips pinning (dev only).

---

## 11. Reference hardware: ESP32-S3 glasses

The first Sentient glasses: **ESP32-S3** (8 MB PSRAM, Wi-Fi), **INMP441** I2S microphone, **MAX98357A** I2S
amplifier, **OV3660** camera, one button. Firmware: `firmware/esp32s3-glasses/` (ESP-IDF). Pin numbers below are
`#define`s in `main/app_config.h`; change them to match the board, they are not fixed by the protocol.

Advertise: `camera.photo`, `display.text`, `notify.show`, `audio.pcm`, `mic.stream`, `button.events`, `battery`.

### 11.1 Microphone (INMP441 → I2S0 RX)

| setting | value | why |
|---|---|---|
| peripheral | `I2S_NUM_0`, master, RX only | leaves `I2S_NUM_1` for the amplifier, so capture and playback run at the same time |
| standard | Philips, `I2S_SLOT_MODE_MONO`, `slot_mask = I2S_STD_SLOT_LEFT` | INMP441 with `L/R` tied to GND drives the left slot |
| data width | `I2S_DATA_BIT_WIDTH_32BIT` on the wire | the INMP441 sends 24 bits MSB-aligned inside a 32-bit slot |
| sample rate | 16000 | what `/ws/voice` expects; no resampling on the device |
| conversion | `int16 = clamp(raw32 >> MIC_GAIN_SHIFT)`, default shift 14 | `>> 16` would be correct 16-bit but very quiet; 14 adds ~12 dB. Tune `MIC_GAIN_SHIFT` (12 loud, 16 quiet) |
| DMA | 4 buffers × 240 frames | ~60 ms of slack; the reader task wakes every ~15 ms |
| pins (defaults) | `SCK=GPIO4`, `WS=GPIO5`, `SD=GPIO6` | any free GPIO; keep them away from the camera's data bus |

Send microphone audio as **PCM16 little-endian mono at 16 kHz** in binary frames of 640 samples (1280 bytes,
40 ms). That is the exact format §6 asks for, so no conversion happens anywhere.

### 11.2 Speaker (MAX98357A → I2S1 TX)

| setting | value |
|---|---|
| peripheral | `I2S_NUM_1`, master, TX only |
| format | Philips, `I2S_DATA_BIT_WIDTH_16BIT`, `I2S_SLOT_MODE_MONO` (the same sample goes to both slots) |
| sample rate | whatever the engine announces: 16000 from `/ws/voice` with `audio_format: "pcm16"`, or the `sample_rate` in `audio.pcm` params (often 24000) |
| pins (defaults) | `BCLK=GPIO15`, `LRCLK/WS=GPIO16`, `DIN=GPIO7`; `SD` pulled high (or to a GPIO to mute), `GAIN` left floating for 9 dB |
| rate changes | `i2s_channel_disable()` → `i2s_channel_reconfig_std_clock()` → `i2s_channel_enable()` before the first frame of a reply |

Two ways audio arrives, both raw PCM16 that goes straight to I2S with no decoding:

- **Voice replies**: `/ws/voice` `start` with `audio_format: "pcm16"` sends an `{"type":"audio",…}` header followed by one
  binary frame per sentence, at the `sample_rate` from `ready`.
- **`audio.pcm` invokes** (what `device_speak` uses for this board): `params` carry `format`, `sample_rate`, `channels`,
  `binary: true` and `bytes`, and the samples follow in one binary frame.

A sentence is roughly 32 KB/s at 16 kHz, so stream the websocket chunks into a ring buffer and start playing at the first
chunk instead of waiting for the whole frame. Never buffer a whole reply in DRAM.

### 11.3 Camera (OV3660 → `camera.photo`)

```
pixel_format = PIXFORMAT_JPEG    frame_size = FRAMESIZE_VGA (640x480)   jpeg_quality = 12   (0 best … 63 worst)
fb_count = 2                     fb_location = CAMERA_FB_IN_PSRAM       grab_mode = CAMERA_GRAB_LATEST
xclk_freq_hz = 20 MHz            OV3660 extras: set_vflip(1), set_brightness(1), set_saturation(-2)
```

VGA at quality 12 gives a **30-60 KB** JPEG, which decides how to return it:

| way | cost on this board | verdict |
|---|---|---|
| binary follow frame (§4.4 #2) | sends straight from the PSRAM frame buffer, no copy, no extra RAM | **use this** |
| `POST /api/nodes/upload` (§4.4 #3) | a second TLS session, ~40 KB heap during the upload | fallback when the socket is busy with audio, or for photos over ~200 KB |
| inline base64 (§4.4 #1) | +33% bytes and a 40-80 KB string built in RAM on top of the frame buffer | avoid |

Always `esp_camera_fb_return()` the buffer immediately after the send, and keep only one capture in flight.

### 11.4 TLS on the ESP32-S3

An mbedTLS session costs roughly 35-45 KB of heap with default buffers. Two sockets (node + voice) means two sessions.
What keeps it comfortable:

- `CONFIG_MBEDTLS_DYNAMIC_BUFFER=y` frees handshake buffers afterwards (saves ~20 KB per session).
- `CONFIG_MBEDTLS_SSL_IN_CONTENT_LEN=4096` and `OUT_CONTENT_LEN=2048`: our frames are small, and the engine never sends
  a TLS record larger than this to a device.
- The engine's certificate is **ECDSA P-256**, so the handshake needs far less RAM and CPU than RSA-2048.
- If RAM is still tight: open the voice socket only while the button is held and close it after the reply, keep
  `buffer_size` at 2048, put Wi-Fi/LWIP buffers in PSRAM (`CONFIG_SPIRAM_TRY_ALLOCATE_WIFI_LWIP=y`), and drop the
  camera to `FRAMESIZE_QVGA` while talking.

**Pinning** (§2) in two steps, because `esp_websocket_client` does not expose the peer certificate:

1. Once per engine: `esp_tls_conn_new_sync()` with no CA (verification off), then
   `mbedtls_ssl_get_peer_cert()` → SHA-256 of `crt->raw` → compare with the expected fingerprint. The user reads that
   fingerprint off the Devices screen in Sentient (`GET /api/nodes/lan` → `fingerprint`), or it comes from the mDNS TXT
   record or the pairing QR's `fp=`.
2. Store that certificate as PEM in NVS and pass it as `cert_pem` to `esp_websocket_client` with
   `skip_cert_common_name_check = true` for every later connection. mbedTLS then verifies the real connection against
   exactly that certificate, and the check costs nothing at runtime.

Redo step 1 only if the connection starts failing with a certificate error (Sentient was reset): treat it as "pair again".

### 11.5 Wi-Fi, keepalive, sleep and battery

- **Wi-Fi**: reconnect on `WIFI_EVENT_STA_DISCONNECTED` with 1, 2, 4 … 30 s backoff plus jitter, the same curve as §8.
  Re-run mDNS discovery after every join: the engine's address changes with DHCP, the fingerprint does not.
- **Keepalive**: use the `keepalive_s` from `welcome` (default 60) for `{"type":"ping"}` on the node socket. Do not
  enable the websocket library's own protocol pings: the engine does not need them and they wake the radio twice.
- **Modem sleep**: `esp_wifi_set_ps(WIFI_PS_MAX_MODEM)` is safe with a 60 s ping and cuts idle current a lot.
- **Deep sleep** drops the sockets, so use it only for long idles: wake on the button with `esp_sleep_enable_ext1_wakeup()`,
  then reconnect with the token from NVS (no pairing needed) and send the button event once connected. Budget ~2 s from
  wake to `welcome` on a known network.
- **Battery**: read the divider with ADC1 oneshot + calibration, send `{"type":"state","battery":N,"charging":false}` on
  connect, every 5 minutes, and whenever it moves 2% or more. Report percent, not volts.

### 11.6 The button

One GPIO with an internal pull-up, 30 ms debounce, polled in a small task:

- **Press**: send `{"type":"event","event":"button","data":{"action":"press"}}` on the node socket, open `/ws/voice?node_token=…`
  (already pinned, so it is one round trip), send `start`, and begin streaming microphone frames.
- **Release**: send `{"type":"end_utterance"}`, stop streaming, keep the socket open for the reply, then close it after
  `VOICE_IDLE_CLOSE_MS` (default 20 s) of silence.
- **Press while the assistant is speaking**: send `{"type":"interrupt"}` and stop playback immediately (barge-in).
- A long press (2 s) can send `{"event":"button","data":{"action":"long_press"}}`; the engine just forwards it as `node.event`.
