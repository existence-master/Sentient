/* Sentient web device app: pairs a phone browser as a node (docs/NODES.md). No build step. */
(() => {
  "use strict";

  const APP_VERSION = "web-1";
  const STORE_KEY = "sentient.node.v1";
  const $ = (id) => document.getElementById(id);
  const wsBase = () => (location.protocol === "https:" ? "wss://" : "ws://") + location.host;

  // ------------------------------------------------------------------ storage
  const store = {
    load() {
      try { return JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch { return {}; }
    },
    save(patch) {
      const next = Object.assign(this.load(), patch);
      try { localStorage.setItem(STORE_KEY, JSON.stringify(next)); } catch { /* private mode */ }
      return next;
    },
    clear() { try { localStorage.removeItem(STORE_KEY); } catch { /* ignore */ } },
  };

  // ------------------------------------------------------------------ ui helpers
  function setStatus(text, kind) {
    const el = $("status");
    el.textContent = text;
    el.className = "pill " + (kind === "ok" ? "pill-ok" : kind === "wait" ? "pill-wait" : "pill-off");
  }

  function toast(title, body, ms = 5000) {
    const el = document.createElement("div");
    el.className = "toast";
    const b = document.createElement("b");
    b.textContent = title;
    el.appendChild(b);
    if (body) el.appendChild(document.createTextNode(body));
    $("toasts").appendChild(el);
    setTimeout(() => el.remove(), ms);
    if (navigator.vibrate) navigator.vibrate(60);
  }

  function logActivity(text, imgSrc) {
    const li = document.createElement("li");
    const t = document.createElement("time");
    t.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    li.appendChild(t);
    if (imgSrc) {
      const img = document.createElement("img");
      img.src = imgSrc;
      img.alt = "";
      li.appendChild(img);
    }
    const span = document.createElement("span");
    span.textContent = text;
    li.appendChild(span);
    const log = $("log");
    log.insertBefore(li, log.firstChild);
    while (log.children.length > 30) log.lastChild.remove();
  }

  function showScreen(name) {
    $("pair").hidden = name !== "pair";
    $("home").hidden = name !== "home";
  }

  function showPairError(msg) {
    const el = $("pair-error");
    el.textContent = msg || "";
    el.hidden = !msg;
  }

  // ------------------------------------------------------------------ capabilities
  function capabilities() {
    const caps = ["notify.show", "display.text", "display.card", "audio.play", "button.events"];
    const media = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    if (media) caps.push("camera.photo", "mic.stream");
    if (navigator.geolocation) caps.push("location.get");
    if ("speechSynthesis" in window) caps.push("speak");
    if (navigator.getBattery) caps.push("battery");
    return caps;
  }

  function platform() {
    return (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || "web";
  }

  // A name the user recognises in the Devices list, without making them type one.
  function defaultDeviceName() {
    const ua = navigator.userAgent || "";
    const p = platform();
    if (/iPhone/i.test(ua)) return "My iPhone";
    if (/iPad/i.test(ua)) return "My iPad";
    if (/Android/i.test(ua)) return /Mobile/i.test(ua) ? "My Android phone" : "My Android tablet";
    if (/Mac/i.test(p)) return "My Mac";
    if (/Win/i.test(p)) return "My Windows PC";
    if (/Linux/i.test(p)) return "My Linux device";
    return "My device";
  }

  // ------------------------------------------------------------------ audio output
  const player = {
    ctx: null, queue: [], source: null, busy: false,
    unlock() {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      if (!this.ctx) this.ctx = new AC();
      if (this.ctx.state === "suspended") this.ctx.resume().catch(() => {});
    },
    ready() { return !!this.ctx && this.ctx.state === "running"; },
    enqueue(buffer) {
      this.queue.push(buffer);
      if (!this.busy) this.next();
    },
    async next() {
      const buf = this.queue.shift();
      if (!buf) { this.busy = false; return; }
      this.busy = true;
      this.unlock();
      try {
        const audio = await this.ctx.decodeAudioData(buf.slice(0));
        const src = this.ctx.createBufferSource();
        src.buffer = audio;
        src.connect(this.ctx.destination);
        src.onended = () => { if (this.source === src) { this.source = null; this.next(); } };
        this.source = src;
        src.start();
      } catch {
        this.next();
      }
    },
    stop() {
      this.queue = [];
      const s = this.source;
      this.source = null;
      this.busy = false;
      if (s) { try { s.stop(); } catch { /* already stopped */ } }
    },
  };
  document.addEventListener("pointerdown", () => player.unlock(), { once: true });

  function b64ToBuffer(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out.buffer;
  }

  // ------------------------------------------------------------------ capability handlers
  function wait(ms) { return new Promise((r) => setTimeout(r, ms)); }

  async function takePhoto(params) {
    const facing = params.facing === "front" ? "user" : "environment";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
    const video = $("preview");
    $("flash").hidden = false;
    try {
      video.srcObject = stream;
      await video.play();
      await wait(600); // let exposure settle
      const maxW = Math.max(320, Math.min(Number(params.max_width) || 1280, 2560));
      const scale = Math.min(1, maxW / (video.videoWidth || maxW));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round((video.videoWidth || 1280) * scale);
      canvas.height = Math.round((video.videoHeight || 720) * scale);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
      logActivity("Took a photo", dataUrl);
      return { mime: "image/jpeg", base64: dataUrl.split(",")[1], width: canvas.width, height: canvas.height };
    } finally {
      stream.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
      $("flash").hidden = true;
    }
  }

  function getLocation(params) {
    return new Promise((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          logActivity("Shared location");
          resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude, accuracy_m: Math.round(pos.coords.accuracy) });
        },
        (err) => reject(new Error(err.code === 1 ? "Location permission was denied on the phone." : err.message)),
        { enableHighAccuracy: true, timeout: Math.min(Number(params.timeout_ms) || 15000, 20000), maximumAge: 30000 },
      );
    });
  }

  function notify(params) {
    const title = params.title || store.load().assistant || "Sentient";
    const text = params.text || params.body || "";
    let system = false;
    if ("Notification" in window && Notification.permission === "granted" && document.hidden) {
      try { new Notification(title, { body: text, tag: params.tag || undefined }); system = true; } catch { /* Android needs a service worker */ }
    }
    toast(title, text, 8000);
    logActivity(`Notification: ${text}`);
    return { shown: true, system };
  }

  let displayTimer = null;
  function display(params, card) {
    $("display-title").textContent = card ? (params.title || "") : "";
    $("display-title").hidden = !card || !params.title;
    $("display-text").textContent = params.text || "";
    const img = $("display-image");
    if (card && params.image_base64) {
      img.src = `data:${params.image_mime || "image/jpeg"};base64,${params.image_base64}`;
      img.hidden = false;
    } else {
      img.hidden = true;
      img.removeAttribute("src");
    }
    $("display").hidden = false;
    clearTimeout(displayTimer);
    if (params.duration_ms) displayTimer = setTimeout(() => { $("display").hidden = true; }, Number(params.duration_ms));
    logActivity(`Shown: ${params.text || params.title || ""}`);
    return { shown: true };
  }

  function speak(params) {
    return new Promise((resolve) => {
      const u = new SpeechSynthesisUtterance(String(params.text || ""));
      if (params.lang) u.lang = params.lang;
      let settled = false;
      const done = (spoken) => { if (!settled) { settled = true; resolve({ spoken }); } };
      u.onend = () => done(true);
      u.onerror = () => done(false);
      speechSynthesis.cancel();
      speechSynthesis.speak(u);
      logActivity(`Said: ${params.text || ""}`);
      setTimeout(() => done(true), 30000);
    });
  }

  async function playAudio(params) {
    player.unlock();
    if (!player.ready()) throw new Error("Tap the Sentient page once so it is allowed to play sound.");
    let buf;
    if (params.base64) buf = b64ToBuffer(params.base64);
    else if (params.url) buf = await (await fetch(params.url)).arrayBuffer();
    else throw new Error("Nothing to play.");
    player.enqueue(buf);
    logActivity(params.text ? `Said: ${params.text}` : "Played a sound");
    return { playing: true };
  }

  async function battery() {
    const b = await navigator.getBattery();
    return { battery: Math.round(b.level * 100), charging: b.charging };
  }

  const HANDLERS = {
    "camera.photo": takePhoto,
    "location.get": getLocation,
    "notify.show": notify,
    "display.text": (p) => display(p, false),
    "display.card": (p) => display(p, true),
    "speak": speak,
    "audio.play": playAudio,
    "battery": battery,
    "button.events": async () => ({ ok: true }),
  };

  // ------------------------------------------------------------------ node socket
  const node = {
    ws: null, retryMs: 1000, retryTimer: null, pingTimer: null, pairing: null, halted: false,

    connect(pairCode, pairName) {
      clearTimeout(this.retryTimer);
      if (this.ws && this.ws.readyState <= 1) this.ws.close();
      const saved = store.load();
      if (!saved.token && !pairCode) { showScreen("pair"); setStatus("Not paired", "off"); return; }
      this.pairing = pairCode ? { code: pairCode, name: pairName } : null;
      this.halted = false;
      setStatus(pairCode ? "Pairing…" : "Connecting…", "wait");
      const ws = new WebSocket(`${wsBase()}/ws/node`);
      this.ws = ws;
      ws.onopen = () => {
        const hello = {
          type: "hello", protocol: 1, kind: "phone", platform: platform(), app_version: APP_VERSION,
          name: (this.pairing && this.pairing.name) || saved.name || "Phone", capabilities: capabilities(),
        };
        if (this.pairing) hello.pair_code = this.pairing.code;
        else hello.token = saved.token;
        ws.send(JSON.stringify(hello));
      };
      ws.onmessage = (ev) => { if (typeof ev.data === "string") this.onMessage(JSON.parse(ev.data)); };
      ws.onclose = () => {
        if (this.ws !== ws) return;
        this.ws = null;
        clearInterval(this.pingTimer);
        $("pair-btn").disabled = false;
        if (this.halted) return;
        const secs = Math.round(this.retryMs / 1000);
        setStatus(`Offline, retrying in ${secs}s`, "off");
        this.retryTimer = setTimeout(() => this.connect(), this.retryMs + Math.random() * 500);
        this.retryMs = Math.min(this.retryMs * 2, 30000);
      };
      ws.onerror = () => {};
    },

    send(obj) {
      if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(obj));
    },

    async onMessage(msg) {
      if (msg.type === "welcome") {
        const saved = store.save({
          node_id: msg.node_id, name: msg.name, assistant: msg.assistant,
          ...(msg.token ? { token: msg.token } : {}),
        });
        this.retryMs = 1000;
        if (this.pairing) {
          toast("Paired", `This device is now connected to ${msg.assistant}.`);
          this.pairing = null;
          clearCodeFromUrl(); // the code is used up; a reload should not try it again
        }
        $("assistant-name").textContent = msg.assistant || "Sentient";
        $("device-label").textContent = saved.name || "";
        setStatus("Connected", "ok");
        showScreen("home");
        setStopped(!!msg.stopped);
        clearInterval(this.pingTimer);
        this.pingTimer = setInterval(() => this.send({ type: "ping" }), (msg.keepalive_s || 60) * 1000);
        this.sendBattery();
      } else if (msg.type === "error") {
        this.onError(msg);
      } else if (msg.type === "invoke") {
        this.onInvoke(msg);
      } else if (msg.type === "stop_state") {
        setStopped(!!msg.stopped);
      }
    },

    onError(msg) {
      const auth = ["pairing_required", "bad_token", "bad_code", "revoked", "rate_limited"];
      if (msg.code === "disabled") {
        this.halted = true;
        setStatus("Devices are off", "off");
        this.retryTimer = setTimeout(() => this.connect(), 60000);
      } else if (msg.code === "replaced") {
        this.halted = true;
        setStatus("Open in another tab", "off");
      } else if (auth.includes(msg.code)) {
        this.halted = true;
        if (msg.code === "bad_token" || msg.code === "revoked") store.save({ token: null });
        showScreen("pair");
        showPairError(msg.message);
        setStatus("Not paired", "off");
        $("pair-btn").disabled = false;
      } else {
        logActivity(`Error: ${msg.message}`);
      }
    },

    async onInvoke(msg) {
      const handler = HANDLERS[msg.capability];
      if (!handler) {
        this.send({ type: "result", id: msg.id, ok: false, error: { code: "unsupported", message: `This phone cannot do ${msg.capability}.` } });
        return;
      }
      try {
        const data = await handler(msg.params || {});
        this.send({ type: "result", id: msg.id, ok: true, data: data || {} });
      } catch (err) {
        const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
        this.send({
          type: "result", id: msg.id, ok: false,
          error: { code: denied ? "permission_denied" : "failed", message: denied ? "Permission was denied on the phone." : String((err && err.message) || err) },
        });
      }
    },

    async sendBattery() {
      if (!navigator.getBattery) return;
      try {
        const b = await navigator.getBattery();
        const push = () => this.send({ type: "state", battery: Math.round(b.level * 100), charging: b.charging });
        push();
        if (!b._sentientHooked) {
          b.addEventListener("levelchange", push);
          b.addEventListener("chargingchange", push);
          b._sentientHooked = true;
        }
      } catch { /* not available */ }
    },
  };

  // ------------------------------------------------------------------ voice (push to talk)
  const WORKLET = "class P extends AudioWorkletProcessor{process(i){const c=i[0]&&i[0][0];if(c)this.port.postMessage(c.slice(0));return true}}registerProcessor('pcm-tap',P);";
  const TARGET_RATE = 16000;

  function toPcm16(f32, inRate) {
    let samples = f32;
    if (inRate !== TARGET_RATE) {
      const ratio = inRate / TARGET_RATE;
      const len = Math.floor(f32.length / ratio);
      samples = new Float32Array(len);
      for (let i = 0; i < len; i++) {
        const start = Math.floor(i * ratio);
        const end = Math.min(f32.length, Math.floor((i + 1) * ratio));
        let sum = 0;
        for (let j = start; j < end; j++) sum += f32[j];
        samples[i] = sum / Math.max(1, end - start);
      }
    }
    const out = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  const TALK_LABELS = {
    idle: "Hold to talk", listening: "Listening…", transcribing: "Got it…", thinking: "Thinking…",
    speaking: "Speaking… tap to stop", standby: "Say the wake word",
  };

  function setTalk(state) {
    voice.state = state;
    const btn = $("talk");
    btn.classList.remove("listening", "thinking", "speaking");
    if (state === "listening" && voice.micOn) btn.classList.add("listening");
    if (state === "transcribing" || state === "thinking") btn.classList.add("thinking");
    if (state === "speaking") btn.classList.add("speaking");
    let label = TALK_LABELS[state] || TALK_LABELS.idle;
    if (state === "listening") label = voice.micOn ? (voice.handsFree ? "Listening… tap to stop" : "Listening… release to send") : TALK_LABELS.idle;
    $("talk-label").textContent = label;
  }

  const voice = {
    ws: null, ready: null, sessionId: null, state: "idle", micOn: false, handsFree: false,
    stream: null, ctx: null, nodes: [], chunks: [], chunkLen: 0, pendingAudio: false, idleTimer: null, approvalId: null,

    open() {
      if (this.ws && this.ws.readyState <= 1 && this.ready) return this.ready;
      const token = store.load().token;
      if (!token) return Promise.reject(new Error("Pair this device first."));
      this.ready = new Promise((resolve, reject) => {
        const ws = new WebSocket(`${wsBase()}/ws/voice?node_token=${encodeURIComponent(token)}`);
        ws.binaryType = "arraybuffer";
        this.ws = ws;
        const timer = setTimeout(() => reject(new Error("Voice did not answer.")), 10000);
        ws.onopen = () => {
          const start = { type: "start", channel: "phone", sample_rate: TARGET_RATE, audio_format: "wav" };
          if (this.sessionId) start.session_id = this.sessionId;
          ws.send(JSON.stringify(start));
        };
        ws.onmessage = (ev) => this.onMessage(ev, () => { clearTimeout(timer); resolve(); });
        ws.onclose = (ev) => {
          clearTimeout(timer);
          if (this.ws === ws) {
            this.ws = null;
            this.ready = null;
            if (this.micOn) this.stopMic(false);
            setTalk("idle");
            // 4401 here means the voice socket refused the node token: either this device was
            // unpaired, or the engine does not accept device tokens on /ws/voice yet.
            if (ev.code === 4401) toast("Voice unavailable", "Sentient did not accept this device for voice yet.");
          }
          reject(new Error("Voice connection closed."));
        };
        ws.onerror = () => {};
      });
      return this.ready;
    },

    bumpIdle() {
      clearTimeout(this.idleTimer);
      this.idleTimer = setTimeout(() => {
        if (!this.micOn && this.ws && this.state !== "speaking" && this.state !== "thinking") {
          this.ws.send(JSON.stringify({ type: "stop" }));
        }
      }, 120000);
    },

    onMessage(ev, onReady) {
      this.bumpIdle();
      if (typeof ev.data !== "string") {
        if (this.pendingAudio) { this.pendingAudio = false; player.enqueue(ev.data); }
        return;
      }
      const m = JSON.parse(ev.data);
      switch (m.type) {
        case "ready": this.sessionId = m.session_id; onReady(); break;
        case "state": setTalk(m.state); break;
        case "transcript": showBubble("you", m.text); showBubble("reply", ""); break;
        case "text_delta": appendReply(m.text); break;
        case "done": if (m.content) showBubble("reply", m.content); break;
        case "audio": this.pendingAudio = true; break;
        case "approval_request": showApproval(m); break;
        case "approval.ack": $("approval").hidden = true; break;
        case "error": if (!m.recoverable) toast("Voice", m.message); break;
        default: break;
      }
    },

    async startMic() {
      await this.open();
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      const AC = window.AudioContext || window.webkitAudioContext;
      try { this.ctx = new AC({ sampleRate: TARGET_RATE }); } catch { this.ctx = new AC(); }
      const ctx = this.ctx;
      const rate = ctx.sampleRate;
      const src = ctx.createMediaStreamSource(this.stream);
      const mute = ctx.createGain();
      mute.gain.value = 0;
      mute.connect(ctx.destination);
      const push = (f32) => {
        if (!this.micOn) return;
        const pcm = toPcm16(f32, rate);
        this.chunks.push(pcm);
        this.chunkLen += pcm.length;
        if (this.chunkLen >= 640) this.flush(); // ~40 ms
      };
      let tap;
      if (ctx.audioWorklet && window.AudioWorkletNode) {
        const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }));
        await ctx.audioWorklet.addModule(url);
        tap = new AudioWorkletNode(ctx, "pcm-tap");
        tap.port.onmessage = (e) => push(e.data);
      } else {
        tap = ctx.createScriptProcessor(2048, 1, 1);
        tap.onaudioprocess = (e) => push(e.inputBuffer.getChannelData(0).slice(0));
      }
      src.connect(tap);
      tap.connect(mute);
      this.nodes = [src, tap, mute];
      this.micOn = true;
      $("conversation").hidden = false;
      setTalk("listening");
    },

    flush() {
      if (!this.chunkLen || !this.ws || this.ws.readyState !== 1) { this.chunks = []; this.chunkLen = 0; return; }
      const merged = new Int16Array(this.chunkLen);
      let off = 0;
      for (const c of this.chunks) { merged.set(c, off); off += c.length; }
      this.chunks = [];
      this.chunkLen = 0;
      this.ws.send(merged.buffer);
    },

    stopMic(endUtterance) {
      if (!this.micOn) return;
      this.flush();
      this.micOn = false;
      this.handsFree = false;
      this.nodes.forEach((n) => { try { n.disconnect(); } catch { /* ignore */ } });
      this.nodes = [];
      if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
      if (this.ctx) this.ctx.close().catch(() => {});
      this.ctx = null;
      if (endUtterance && this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify({ type: "end_utterance" }));
      setTalk(this.state === "listening" ? "idle" : this.state);
    },

    interrupt() {
      player.stop();
      if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify({ type: "interrupt" }));
    },

    async sendText(text) {
      await this.open();
      $("conversation").hidden = false;
      showBubble("you", text);
      showBubble("reply", "");
      this.ws.send(JSON.stringify({ type: "text", text }));
    },
  };

  function showBubble(which, text) {
    const el = $(which);
    el.textContent = text || "";
    el.hidden = !text;
    $("conversation").hidden = false;
  }

  function appendReply(text) {
    const el = $("reply");
    el.textContent += text;
    el.hidden = !el.textContent;
  }

  function showApproval(m) {
    voice.approvalId = m.approval_id;
    $("approval-text").textContent = `${store.load().assistant || "Sentient"} wants to use ${m.name}. Allow it?`;
    $("approval").hidden = false;
  }

  function answerApproval(decision) {
    if (voice.ws && voice.approvalId) voice.ws.send(JSON.stringify({ type: "approval.respond", approval_id: voice.approvalId, decision }));
    $("approval").hidden = true;
  }

  // ------------------------------------------------------------------ talk button
  function setupTalk() {
    const btn = $("talk");
    let pressedAt = 0;
    let startedThisPress = false;

    btn.addEventListener("pointerdown", async (e) => {
      e.preventDefault();
      try { btn.setPointerCapture(e.pointerId); } catch { /* ignore */ }
      pressedAt = performance.now();
      startedThisPress = false;
      player.unlock();
      node.send({ type: "event", event: "button", data: { action: "press" } });
      if (voice.state === "speaking" || player.busy) voice.interrupt();
      if (voice.micOn) { voice.stopMic(true); return; } // second tap ends hands-free listening
      try {
        await voice.startMic();
        startedThisPress = true;
      } catch (err) {
        const denied = err && err.name === "NotAllowedError";
        toast("Microphone", denied ? "Allow the microphone for this page to talk." : String(err.message || err));
        setTalk("idle");
      }
    });

    const release = () => {
      node.send({ type: "event", event: "button", data: { action: "release" } });
      if (!voice.micOn || !startedThisPress) return;
      if (performance.now() - pressedAt < 350) {
        voice.handsFree = true; // a tap: keep listening, the server decides when you are done
        setTalk("listening");
        return;
      }
      voice.stopMic(true);
    };
    btn.addEventListener("pointerup", release);
    btn.addEventListener("pointercancel", release);
    btn.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  // ------------------------------------------------------------------ permissions
  async function refreshPermissions() {
    if (!navigator.permissions) return;
    for (const btn of document.querySelectorAll("[data-perm]")) {
      const name = btn.dataset.perm === "location" ? "geolocation" : btn.dataset.perm;
      try {
        const st = await navigator.permissions.query({ name });
        btn.classList.toggle("granted", st.state === "granted");
      } catch { /* not queryable in this browser */ }
    }
  }

  async function requestPermission(kind) {
    try {
      if (kind === "camera" || kind === "microphone") {
        const s = await navigator.mediaDevices.getUserMedia(kind === "camera" ? { video: true } : { audio: true });
        s.getTracks().forEach((t) => t.stop());
      } else if (kind === "location") {
        await getLocation({});
      } else if (kind === "notifications" && "Notification" in window) {
        await Notification.requestPermission();
      }
    } catch (err) {
      toast("Permission", String(err.message || err));
    }
    refreshPermissions();
  }

  // ------------------------------------------------------------------ boot
  // The pairing code arrives as /node/#code=123456 (the QR link) or ?code=123456.
  // The hash is left in place until pairing succeeds, so a reload still works.
  function codeFromUrl() {
    const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
    const query = new URLSearchParams(location.search);
    return (hash.get("code") || query.get("code") || "").replace(/\D/g, "").slice(0, 6);
  }

  function clearCodeFromUrl() {
    if (location.hash || location.search) history.replaceState(null, "", location.pathname);
  }

  // Called on load and whenever the hash changes, so opening the link in an already
  // open tab prefills too (that navigation fires hashchange and nothing else).
  function applyPairingCode() {
    const code = codeFromUrl();
    if (!code) return false;
    const field = $("code");
    field.value = code;
    if (!$("device-name").value.trim()) $("device-name").value = defaultDeviceName();
    showPairError("");
    showScreen("pair");
    setStatus(store.load().token ? "Pair again?" : "Not paired", "off");
    // The code is filled in, so the only thing left to do is confirm the name.
    try {
      $("device-name").focus({ preventScroll: true });
    } catch {
      $("device-name").focus();
    }
    return true;
  }

  // Stop everything (docs/NODES.md §4.5): one button that stops all of Sentient's work, then offers Resume.
  let stopped = false;
  function setStopped(value) {
    stopped = value;
    $("stopped-note").hidden = !value;
    $("stop-all").textContent = value ? "Resume" : "Stop everything";
    $("stop-all").classList.toggle("resume", value);
  }

  function boot() {
    if (!window.isSecureContext) $("insecure").hidden = false;
    const saved = store.load();
    if (saved.assistant) $("assistant-name").textContent = saved.assistant;
    $("device-name").value = saved.name || defaultDeviceName();

    $("pair-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const c = $("code").value.replace(/\D/g, "");
      if (c.length !== 6) { showPairError("The code has 6 digits."); return; }
      showPairError("");
      $("pair-btn").disabled = true;
      node.retryMs = 1000;
      node.connect(c, $("device-name").value.trim() || "My phone");
    });
    $("code").addEventListener("input", (e) => {
      e.target.value = e.target.value.replace(/\D/g, "").slice(0, 6);
    });
    $("display-close").addEventListener("click", () => { $("display").hidden = true; });
    $("stop-all").addEventListener("click", () => {
      if (!node.ws || node.ws.readyState !== 1) { toast("Sentient", "Not connected right now. Try again in a moment."); return; }
      node.send({ type: stopped ? "resume" : "stop_all" });
    });
    $("approve").addEventListener("click", () => answerApproval("allow"));
    $("deny").addEventListener("click", () => answerApproval("deny"));
    $("type-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = $("type-input").value.trim();
      if (!text) return;
      $("type-input").value = "";
      player.unlock();
      try { await voice.sendText(text); } catch (err) { toast("Sentient", String(err.message || err)); }
    });
    $("forget").addEventListener("click", () => {
      if (!confirm("Forget this device? You will need a new pairing code to connect again.")) return;
      node.halted = true;
      if (node.ws) node.ws.close();
      if (voice.ws) voice.ws.close();
      store.clear();
      showScreen("pair");
      setStatus("Not paired", "off");
    });
    document.querySelectorAll("[data-perm]").forEach((b) => b.addEventListener("click", () => requestPermission(b.dataset.perm)));
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && !node.ws && !node.halted && store.load().token) { node.retryMs = 1000; node.connect(); }
    });
    window.addEventListener("online", () => { if (!node.ws && !node.halted && store.load().token) node.connect(); });
    window.addEventListener("hashchange", applyPairingCode);
    setupTalk();
    refreshPermissions();

    if (applyPairingCode()) return; // a code in the link: show the pair screen with it filled in
    if (saved.token) {
      showScreen("home");
      node.connect();
    } else {
      showScreen("pair");
    }
  }

  boot();
})();
