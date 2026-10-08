# 0011. Connect devices with a small JSON protocol over WebSockets

- **Status:** Accepted
- **Date:** 2026-09-15

## Context

Sentient is meant to live on phones and on smart glasses built around an ESP32-S3. Devices need to show
text, speak, take photos and stream audio, from hardware with little memory, over the home network.

## Decision

Devices ("nodes") connect over WebSockets with a `hello`, `invoke` and `result` protocol documented in
[`docs/NODES.md`](../NODES.md). Pairing uses a six-digit code that is exchanged for a per-device token stored
as a hash. On the home network the engine runs a separate TLS listener (off by default) that serves only device
routes, announced over mDNS, with certificate fingerprint pinning. Large payloads (photos, audio) travel as
binary frames so microcontrollers never hold them as base64 text.

## Consequences

The same engine serves the desktop, a phone web app, a reference device program and firmware. The main API
never reaches the network. Firmware authors have a precise spec; protocol changes must keep it current.

## Alternatives considered

MQTT: good for microcontrollers, but adds a broker. Bluetooth only: no phone web app, short range. Exposing
the main API on the LAN: a large attack surface.
