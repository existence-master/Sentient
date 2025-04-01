// src/interface/lib/WebSocketAudioClient.js

/**
 * @typedef {Object} WebSocketClientOptions
 * @property {() => void} [onConnected]
 * @property {() => void} [onDisconnected]
 * @property {(audioChunk: Float32Array) => void} [onAudioChunkReceived] - Callback for received audio data
 * @property {(level: number) => void} [onAudioLevel] - Callback for local audio level (optional)
 * @property {(error: Error) => void} [onError] - Callback for errors
 * @property {(message: any) => void} [onTextMessage] - Callback for non-audio JSON messages
 */

// Helper function to convert Float32Array to Base64 string
function float32ArrayToBase64(buffer) {
	let binary = ""
	const bytes = new Uint8Array(buffer.buffer)
	const len = bytes.byteLength
	for (let i = 0; i < len; i++) {
		binary += String.fromCharCode(bytes[i])
	}
	return window.btoa(binary)
}

// Helper function to convert Base64 string to Float32Array
function base64ToFloat32Array(base64) {
	try {
		const binary_string = window.atob(base64)
		const len = binary_string.length
		const bytes = new Uint8Array(len)
		for (let i = 0; i < len; i++) {
			bytes[i] = binary_string.charCodeAt(i)
		}
		// Important: Ensure the byte array length is a multiple of 4 for Float32
		if (bytes.buffer.byteLength % 4 !== 0) {
			console.warn(
				"Received Base64 audio data with length not multiple of 4. Padding or truncation might occur."
			)
			// Optionally, handle padding or return null/error
			// For simplicity, we proceed, but this might cause issues.
		}
		return new Float32Array(bytes.buffer)
	} catch (e) {
		console.error("Error decoding Base64 to Float32Array:", e)
		return null
	}
}

export class WebSocketAudioClient {
	/** @type {WebSocket | null} */
	ws = null
	/** @type {MediaStream | null} */
	mediaStream = null
	/** @type {AudioContext | null} */
	audioContext = null
	/** @type {MediaStreamAudioSourceNode | null} */
	sourceNode = null
	/** @type {AudioWorkletNode | null} */
	workletNode = null
	/** @type {AnalyserNode | null} */
	analyser = null // For audio level analysis
	/** @type {Uint8Array | null} */
	dataArray = null // For audio level analysis
	/** @type {number | null} */
	animationFrameId = null // For audio level analysis
	/** @type {WebSocketClientOptions} */
	options = {}
	/** @type {string | null} */
	websocketId = null
	/** @type {boolean} */
	isConnected = false
	/** @type {string} */
	serverUrl = "ws://localhost:8000/voice/websocket/offer" // Use the documented endpoint

	/**
	 * @param {WebSocketClientOptions} options
	 */
	constructor(options = {}) {
		this.options = options
		this.websocketId = "ws_" + Math.random().toString(36).substring(2, 9) // Generate unique ID
		console.log(
			`[WebSocketAudioClient] Initialized with ID: ${this.websocketId}`
		)
	}

	async connect() {
		if (this.isConnected || this.ws) {
			console.warn(
				"[WebSocketAudioClient] Already connected or connecting."
			)
			return
		}
		console.log("[WebSocketAudioClient] Attempting to connect...")

		try {
			// 1. Get user media (microphone)
			try {
				// Use desired constraints, 16kHz is often good for STT
				const constraints = {
					audio: {
						sampleRate: 16000,
						echoCancellation: true,
						noiseSuppression: true
					},
					video: false
				}
				this.mediaStream =
					await navigator.mediaDevices.getUserMedia(constraints)
				console.log("[WebSocketAudioClient] Got user media.")
				this.mediaStream.getAudioTracks().forEach((track) => {
					console.log(
						"[WebSocketAudioClient] Actual track settings:",
						track.getSettings()
					)
				})
			} catch (mediaError) {
				console.error(
					"[WebSocketAudioClient] Media access error:",
					mediaError
				)
				this.handleError(
					new Error(
						"Microphone access failed. Please check permissions and devices."
					)
				)
				return
			}

			// 2. Setup Audio Context and Worklet
			await this.setupAudioProcessing()

			// 3. Establish WebSocket Connection
			this.ws = new WebSocket(this.serverUrl)
			this.setupWebSocketHandlers()
		} catch (error) {
			console.error(
				"[WebSocketAudioClient] Connection setup error:",
				error
			)
			this.handleError(
				error instanceof Error ? error : new Error(String(error))
			)
			this.disconnect() // Ensure cleanup on error
		}
	}

	async setupAudioProcessing() {
		if (!this.mediaStream) throw new Error("Media stream is not available.")
		if (this.audioContext) return // Already set up

		try {
			this.audioContext = new (window.AudioContext ||
				window.webkitAudioContext)({
				sampleRate: 16000 // Match the sample rate requested from getUserMedia
			})
			console.log(
				`[WebSocketAudioClient] AudioContext created with sample rate: ${this.audioContext.sampleRate}`
			)

			// --- Audio Worklet Setup ---
			console.log("[WebSocketAudioClient] Adding AudioWorklet module...")
			// Adjust the path based on where you place audio-processor.js
			// In Next.js, placing it in /public makes it available at the root URL
			await this.audioContext.audioWorklet.addModule(
				"/audio-processor.js"
			)
			console.log("[WebSocketAudioClient] AudioWorklet module added.")

			this.workletNode = new AudioWorkletNode(
				this.audioContext,
				"audio-processor"
			)
			console.log("[WebSocketAudioClient] AudioWorkletNode created.")

			this.sourceNode = this.audioContext.createMediaStreamSource(
				this.mediaStream
			)
			this.sourceNode.connect(this.workletNode)

			// Don't connect worklet to destination unless you want local echo
			// this.workletNode.connect(this.audioContext.destination);

			// Handle messages (audio chunks) from the worklet
			this.workletNode.port.onmessage = (event) => {
				if (this.ws && this.ws.readyState === WebSocket.OPEN) {
					const pcmFloat32Data = event.data // This is Float32Array
					// Encode as Base64 and send
					const base64Audio = float32ArrayToBase64(pcmFloat32Data)
					const message = JSON.stringify({
						event: "media",
						media: { payload: base64Audio }
					})
					// console.log(`[WebSocketAudioClient] Sending media chunk, base64 length: ${base64Audio.length}`); // DEBUG
					this.ws.send(message)
				}
			}
			console.log(
				"[WebSocketAudioClient] AudioWorklet processing setup complete."
			)
			// --- End Audio Worklet Setup ---

			// Optional: Setup analysis node (connect source to analyser too)
			if (this.options.onAudioLevel) {
				this.analyser = this.audioContext.createAnalyser()
				this.analyser.fftSize = 256
				this.analyser.smoothingTimeConstant = 0.3
				this.sourceNode.connect(this.analyser) // Connect source to analyser
				const bufferLength = this.analyser.frequencyBinCount
				this.dataArray = new Uint8Array(bufferLength)
				this.startAnalysisLoop()
				console.log("[WebSocketAudioClient] Audio analysis setup.")
			}
		} catch (error) {
			console.error(
				"[WebSocketAudioClient] Error setting up audio processing:",
				error
			)
			throw new Error("Failed to setup audio processing.")
		}
	}

	setupWebSocketHandlers() {
		if (!this.ws) return

		this.ws.onopen = () => {
			console.log("[WebSocketAudioClient] WebSocket connection opened.")
			// Send the start message required by fastrtc
			const startMessage = JSON.stringify({
				event: "start",
				websocket_id: this.websocketId
			})
			console.log(
				"[WebSocketAudioClient] Sending start message:",
				startMessage
			)
			this.ws?.send(startMessage)

			this.isConnected = true
			if (this.options.onConnected) this.options.onConnected()

			// Resume AudioContext if suspended (often needed after user interaction)
			if (this.audioContext && this.audioContext.state === "suspended") {
				this.audioContext.resume().then(() => {
					console.log("[WebSocketAudioClient] AudioContext resumed.")
				})
			}
		}

		this.ws.onmessage = (event) => {
			// Assume server sends JSON containing Base64 audio or text messages
			try {
				const data = JSON.parse(event.data)
				// console.log("[WebSocketAudioClient] Received message:", data); // Debug

				if (data.type === "audio_chunk" && data.data) {
					// Check fastrtc source for actual type! Guessed 'audio_chunk'
					const base64Audio = data.data
					const audioChunk = base64ToFloat32Array(base64Audio)
					if (audioChunk && this.options.onAudioChunkReceived) {
						this.options.onAudioChunkReceived(audioChunk)
					}
				} else if (
					data.type &&
					data.type !== "ping" &&
					data.type !== "pong"
				) {
					// Handle other message types (log, error, warnings, potentially text transcript?)
					if (this.options.onTextMessage) {
						this.options.onTextMessage(data)
					} else {
						console.log(
							"[WebSocketAudioClient] Received text/status message:",
							data
						)
					}
				}
			} catch (e) {
				console.error(
					"[WebSocketAudioClient] Error processing received message:",
					e,
					"Raw data:",
					event.data
				)
				// Handle potential binary audio data if server sends it raw (less likely with fastrtc JSON structure)
				if (
					event.data instanceof Blob ||
					event.data instanceof ArrayBuffer
				) {
					console.warn(
						"[WebSocketAudioClient] Received unexpected binary message."
					)
					// Try processing as binary if needed
				}
			}
		}

		this.ws.onerror = (error) => {
			console.error("[WebSocketAudioClient] WebSocket error:", error)
			this.handleError(new Error("WebSocket connection error."))
			// The disconnect method will be called by onclose usually
		}

		this.ws.onclose = (event) => {
			console.log(
				`[WebSocketAudioClient] WebSocket closed: Code=${event.code}, Reason=${event.reason}`
			)
			this.isConnected = false
			// Don't automatically call disconnect here, let the main app logic decide
			if (this.options.onDisconnected) {
				this.options.onDisconnected()
			}
			this.cleanupAudioProcessing() // Clean up audio resources on close
			this.ws = null // Clear reference
		}
	}

	// Audio Analysis Loop (similar to WebRTCClient)
	startAnalysisLoop() {
		if (
			this.animationFrameId ||
			!this.analyser ||
			!this.dataArray ||
			!this.options.onAudioLevel
		)
			return
		let lastUpdateTime = 0
		const throttleInterval = 100

		const analyze = () => {
			if (!this.analyser || !this.dataArray) {
				this.animationFrameId = null
				return
			}
			this.analyser.getByteFrequencyData(this.dataArray)
			const currentTime = performance.now()
			if (currentTime - lastUpdateTime > throttleInterval) {
				let sum = 0
				for (let i = 0; i < this.dataArray.length; i++) {
					sum += this.dataArray[i]
				}
				const averageLevel = sum / this.dataArray.length / 128.0
				this.options.onAudioLevel(Math.min(averageLevel * 1.5, 1.0))
				lastUpdateTime = currentTime
			}
			this.animationFrameId = requestAnimationFrame(analyze)
		}
		this.animationFrameId = requestAnimationFrame(analyze)
	}

	stopAnalysis() {
		if (this.animationFrameId !== null) {
			cancelAnimationFrame(this.animationFrameId)
			this.animationFrameId = null
		}
		this.analyser = null
		this.dataArray = null
	}

	cleanupAudioProcessing() {
		console.log(
			"[WebSocketAudioClient] Cleaning up audio processing nodes..."
		)
		this.stopAnalysis() // Stop analysis loop

		// Disconnect nodes in reverse order
		if (this.workletNode) {
			this.workletNode.port.close() // Close the message port
			try {
				if (this.sourceNode) this.workletNode.disconnect() // Disconnect worklet if source exists
			} catch (e) {
				console.warn("Error disconnecting worklet node", e)
			}
			this.workletNode = null
		}
		if (this.sourceNode) {
			try {
				// Disconnect from worklet (already done) and analyser
				if (this.analyser) this.sourceNode.disconnect(this.analyser)
			} catch (e) {
				console.warn("Error disconnecting source node", e)
			}
			this.sourceNode = null
		}
		if (this.mediaStream) {
			this.mediaStream.getTracks().forEach((track) => track.stop())
			console.log("[WebSocketAudioClient] Media tracks stopped.")
			this.mediaStream = null
		}
		// Close AudioContext - might be better to keep it if app needs it later
		// if (this.audioContext && this.audioContext.state !== 'closed') {
		//     this.audioContext.close().then(() => console.log('[WebSocketAudioClient] AudioContext closed.'));
		//     this.audioContext = null;
		// }
		console.log("[WebSocketAudioClient] Audio processing cleanup finished.")
	}

	disconnect() {
		console.log("[WebSocketAudioClient] Disconnecting...")
		this.cleanupAudioProcessing()

		if (this.ws) {
			// Prevent onclose handler from triggering reconnect logic if it exists
			this.ws.onclose = null
			this.ws.onerror = null
			this.ws.close(1000, "Client disconnecting normally") // Normal closure
			this.ws = null // Clear reference immediately
		}

		this.isConnected = false
		// Ensure onDisconnected is called if not already triggered by onclose
		if (this.options.onDisconnected) {
			// Check if already called by onclose handler? Needs careful state mgmt
			// For simplicity, call it again, the receiver should be idempotent.
			this.options.onDisconnected()
		}
		console.log("[WebSocketAudioClient] Disconnected.")
	}

	setMuted(muted) {
		if (this.mediaStream) {
			this.mediaStream.getAudioTracks().forEach((track) => {
				track.enabled = !muted
			})
			console.log(`[WebSocketAudioClient] Mic track enabled: ${!muted}`)
		}
	}

	/** @param {Error} error */
	handleError(error) {
		console.error("[WebSocketAudioClient] Error:", error)
		if (this.options.onError) {
			this.options.onError(error)
		}
		// Optionally trigger disconnect on certain errors
		// this.disconnect();
	}
}
