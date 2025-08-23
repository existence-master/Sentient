import { AudioPlayer } from "./AudioPlayer"

export class WebSocketClient {
	constructor(options = {}) {
		this.options = options
		const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws"
		const serverUrlHttp =
			process.env.NEXT_PUBLIC_APP_SERVER_URL || "http://localhost:5000"
		const serverUrlWs = serverUrlHttp.replace(/^http/, "ws")
		this.serverUrl = `${serverUrlWs}/voice/ws/voice`

		this.ws = null
		this.mediaStream = null
		this.audioContext = null
		this.workletNode = null
		this.sourceNode = null
		this.audioPlayer = null
		this.analyser = null
		this.token = null
		this.chatId = null
	}

	async connect(deviceId, token) {
		if (this.ws) {
			console.warn("[VoiceClient] Already connected or connecting.")
			return
		}
		this.token = token
		this.chatId = null

		try {
			this.audioPlayer = new AudioPlayer(16000)
			const constraints = {
				audio: deviceId
					? {
							deviceId: { exact: deviceId },
							sampleRate: 16000,
							echoCancellation: true,
							noiseSuppression: true
						}
					: {
							sampleRate: 16000,
							echoCancellation: true,
							noiseSuppression: true
						},
				video: false
			}
			this.mediaStream =
				await navigator.mediaDevices.getUserMedia(constraints)
			await this.setupAudioProcessing()
			this.ws = new WebSocket(this.serverUrl)
			this.setupWebSocketHandlers()
		} catch (error) {
			console.error("[VoiceClient] Connection failed:", error)
			this.options.onConnectError?.(error)
			this.disconnect()
			throw error
		}
	}

	async setupAudioProcessing() {
		if (!this.mediaStream) {
			console.error("[VoiceClient] No media stream to process.")
			return
		}

		this.audioContext = new (window.AudioContext ||
			window.webkitAudioContext)({ sampleRate: 16000 })

		try {
			await this.audioContext.audioWorklet.addModule("/audioProcessor.js")
		} catch (e) {
			console.error("[VoiceClient] Failed to load AudioWorklet:", e)
			throw new Error("Could not load audio processor.")
		}

		this.sourceNode = this.audioContext.createMediaStreamSource(
			this.mediaStream
		)
		this.workletNode = new AudioWorkletNode(
			this.audioContext,
			"audio-processor"
		)

		this.workletNode.port.onmessage = (event) => {
			if (this.ws?.readyState === WebSocket.OPEN) {
				const pcmData = event.data
				const int16Data = new Int16Array(pcmData.length)
				for (let i = 0; i < pcmData.length; i++) {
					int16Data[i] = Math.max(-1, Math.min(1, pcmData[i])) * 32767
				}
				this.ws.send(int16Data.buffer)
			}
		}

		this.sourceNode.connect(this.workletNode)
		this.workletNode.connect(this.audioContext.destination)

		this.analyser = this.audioContext.createAnalyser()
		this.analyser.fftSize = 512
		this.sourceNode.connect(this.analyser)
		this.options.onAudioLevel?.(0)
		this.startAnalysisLoop()
	}

	startAnalysisLoop() {
		if (!this.analyser) return
		const dataArray = new Uint8Array(this.analyser.frequencyBinCount)
		const analyze = () => {
			if (!this.analyser) return
			this.analyser.getByteFrequencyData(dataArray)
			let sum = 0
			for (const amplitude of dataArray) {
				sum += amplitude * amplitude
			}
			const rms = Math.sqrt(sum / dataArray.length)
			const normalizedLevel = (rms / 128) * 1.5
			this.options.onAudioLevel?.(Math.min(normalizedLevel, 1.0))
			requestAnimationFrame(analyze)
		}
		analyze()
	}

	setupWebSocketHandlers() {
		if (!this.ws) return

		this.ws.onopen = () => {
			if (this.token) {
				this.ws.send(
					JSON.stringify({
						type: "auth",
						token: this.token,
						chatId: this.chatId
					})
				)
			} else {
				console.error(
					"[VoiceClient] No auth token available. Closing connection."
				)
				this.ws.close(1008, "Auth token missing")
			}
		}

		this.ws.onmessage = async (event) => {
			if (event.data instanceof Blob) {
				const arrayBuffer = await event.data.arrayBuffer()
				const float32Array = new Float32Array(arrayBuffer)
				this.audioPlayer?.addChunk(float32Array)
			} else {
				try {
					const message = JSON.parse(event.data)
					if (message.type === "auth_success") {
						this.options.onConnected?.()
					} else {
						this.options.onMessage?.(message)
					}
				} catch (e) {
					console.error("[VoiceClient] Error parsing JSON:", e)
				}
			}
		}

		this.ws.onerror = (error) => {
			console.error("[VoiceClient] WebSocket error:", error)
			this.options.onConnectError?.(
				new Error("WebSocket connection error.")
			)
			this.disconnect()
		}

		this.ws.onclose = (event) => {
			this.options.onDisconnected?.()
			this.cleanup()
		}
	}

	disconnect() {
		if (this.ws) {
			this.ws.close(1000, "Client initiated disconnect")
		} else {
			this.cleanup()
		}
	}

	cleanup() {
		this.audioPlayer?.stop()
		this.audioPlayer = null

		if (this.mediaStream) {
			this.mediaStream.getTracks().forEach((track) => track.stop())
			this.mediaStream = null
		}
		if (this.workletNode) this.workletNode.disconnect()
		if (this.sourceNode) this.sourceNode.disconnect()
		if (this.analyser) this.analyser.disconnect()
		if (this.audioContext?.state !== "closed") this.audioContext?.close()

		this.ws = null
		this.audioContext = null
		this.workletNode = null
		this.sourceNode = null
		this.analyser = null
		this.token = null
		this.chatId = null
	}
}
