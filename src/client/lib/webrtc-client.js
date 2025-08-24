export class WebRTCClient {
	constructor(options = {}) {
		this.peerConnection = null
		this.mediaStream = null
		this.dataChannel = null
		this.options = options
		this.audioContext = null
		this.analyser = null
		this.dataArray = null
		this.animationFrameId = null
		this.serverUrl =
			process.env.NEXT_PUBLIC_APP_SERVER_URL || "http://localhost:5000"
		this.iceServers = options.iceServers || [] // Store iceServers from options

		// --- MODIFICATION: Add a queue for ICE candidates ---
		this.iceCandidateQueue = []
		// --- MODIFICATION: Add a timer for handling temporary disconnections ---
		this.disconnectTimer = null
		console.log("[WebRTCClient] Initialized with options:", options)
	}

	async connect(deviceId, authToken, rtcToken) {
		if (!authToken) {
			throw new Error("Authentication token is required to connect.")
		}
		if (!rtcToken) {
			throw new Error("RTC token is required to connect.")
		}

		// --- MODIFICATION: Reset the queue on new connection ---
		this.iceCandidateQueue = []
		console.log(
			`[WebRTCClient] Connecting with deviceId: ${deviceId}, rtcToken: ${rtcToken}`
		)

		try {
			this.peerConnection = new RTCPeerConnection({
				iceServers: this.iceServers
			})
			console.log(
				"[WebRTCClient] PeerConnection created with ICE servers:",
				this.iceServers
			)

			// --- MODIFICATION: Queue ICE candidates instead of sending immediately ---
			this.peerConnection.onicecandidate = (event) => {
				if (event.candidate) {
					this.iceCandidateQueue.push(event.candidate)
				}
			}

			this.peerConnection.oniceconnectionstatechange = (event) => {
				const state = this.peerConnection.iceConnectionState
				console.log(
					`[WebRTCClient] oniceconnectionstatechange: ${state}`
				)
			}

			this.peerConnection.onconnectionstatechange = (event) => {
				const state = this.peerConnection.connectionState
				console.log(`[WebRTCClient] onconnectionstatechange: ${state}`)

				// --- START MODIFICATION: Handle temporary disconnections ---
				// If connection is established or re-established, clear any pending disconnect timer.
				if (state === "connected") {
					if (this.disconnectTimer) {
						console.log(
							"[WebRTCClient] Connection re-established, clearing disconnect timer."
						)
						clearTimeout(this.disconnectTimer)
						this.disconnectTimer = null
					}
					this.options.onConnected?.()
				}

				// When disconnected, it might be temporary. Start a timer to see if it recovers.
				if (state === "disconnected") {
					console.warn(
						"WebRTC connection is temporarily disconnected. Starting a 300-second timer to see if it recovers..."
					)
					if (!this.disconnectTimer) {
						this.disconnectTimer = setTimeout(() => {
							console.error(
								"WebRTC connection did not recover from 'disconnected' state within the time limit. Forcibly disconnecting."
							)
							this.disconnect()
						}, 300000) // 300-second grace period for recovery, should be enough for TTS
					}
				}

				// If the connection fails or is closed, it's permanent. Disconnect immediately.
				if (state === "failed" || state === "closed") {
					if (this.disconnectTimer) {
						clearTimeout(this.disconnectTimer)
						this.disconnectTimer = null
					}
					console.error(
						"Peer connection failed or was permanently closed. Cleaning up."
					)
					this.disconnect()
				}
				// --- END MODIFICATION ---
			}

			const audioConstraints = {
				...(deviceId && { deviceId: { exact: deviceId } }),
				noiseSuppression: false,
				echoCancellation: false
			}
			console.log(
				"[WebRTCClient] Requesting user media with constraints:",
				audioConstraints
			)
			this.mediaStream = await navigator.mediaDevices.getUserMedia({
				audio: audioConstraints
			})
			console.log("[WebRTCClient] Got user media stream.")

			this.setupAudioAnalysis()

			this.mediaStream.getTracks().forEach((track) => {
				if (this.peerConnection) {
					console.log(
						"[WebRTCClient] Adding audio track to PeerConnection."
					)
					this.peerConnection.addTrack(track, this.mediaStream)
				}
			})

			this.peerConnection.addEventListener("track", (event) => {
				this.options.onAudioStream?.(event.streams[0])
			})

			this.dataChannel = this.peerConnection.createDataChannel("events")
			console.log("[WebRTCClient] Data channel created.")
			this.dataChannel.addEventListener("message", (event) => {
				try {
					this.options.onEvent?.(JSON.parse(event.data))
				} catch (error) {
					console.error("Error parsing data channel message:", error)
				}
			})

			const offer = await this.peerConnection.createOffer()
			console.log("[WebRTCClient] Creating offer...")
			await this.peerConnection.setLocalDescription(offer)
			console.log("[WebRTCClient] Local description (offer) set.")

			console.log("[WebRTCClient] Sending offer to server...")
			const response = await fetch(
				`${this.serverUrl}/voice/webrtc/offer`,
				{
					method: "POST",
					mode: "cors",
					headers: {
						"Content-Type": "application/json",
						Authorization: `Bearer ${authToken}`
					},
					body: JSON.stringify({
						sdp: offer.sdp,
						type: offer.type,
						webrtc_id: rtcToken
					})
				}
			)

			if (!response.ok) {
				const errorText = await response.text()
				throw new Error(
					`Server responded with ${response.status}: ${errorText}`
				)
			}

			const serverResponse = await response.json()
			console.log("[WebRTCClient] Received answer from server.")
			await this.peerConnection.setRemoteDescription(serverResponse)
			console.log("[WebRTCClient] Remote description (answer) set.")

			// --- MODIFICATION: Send all queued ICE candidates now ---
			console.log(
				`[WebRTCClient] Sending ${this.iceCandidateQueue.length} queued ICE candidates...`
			)
			for (const candidate of this.iceCandidateQueue) {
				fetch(`${this.serverUrl}/voice/webrtc/offer`, {
					method: "POST",
					mode: "cors",
					headers: {
						"Content-Type": "application/json",
						Authorization: `Bearer ${authToken}`
					},
					body: JSON.stringify({
						candidate: candidate.toJSON(),
						webrtc_id: rtcToken,
						type: "ice-candidate"
					})
				}).catch((e) =>
					console.error("Error sending queued ICE candidate:", e)
				)
			}
			this.iceCandidateQueue = [] // Clear the queue
			console.log("[WebRTCClient] ICE candidate queue cleared.")
		} catch (error) {
			console.error("Error connecting WebRTC:", error)
			this.disconnect()
			throw error
		}
	}

	setupAudioAnalysis() {
		if (!this.mediaStream || !this.options.onAudioLevel) return
		try {
			console.log("[WebRTCClient] Setting up audio analysis.")
			this.audioContext = new (window.AudioContext ||
				window.webkitAudioContext)()
			this.analyser = this.audioContext.createAnalyser()
			this.analyser.fftSize = 256
			const source = this.audioContext.createMediaStreamSource(
				this.mediaStream
			)
			source.connect(this.analyser)
			const bufferLength = this.analyser.frequencyBinCount
			this.dataArray = new Uint8Array(bufferLength)
			this.startAnalysis()
		} catch (error) {
			console.error("Error setting up audio analysis:", error)
		}
	}

	startAnalysis() {
		if (!this.analyser || !this.dataArray || !this.options.onAudioLevel)
			return
		let lastUpdateTime = 0
		const throttleInterval = 100 // 100ms, i.e., 10 times per second
		console.log("[WebRTCClient] Starting audio analysis loop.")

		const analyze = () => {
			if (!this.analyser || !this.dataArray) return
			this.analyser.getByteFrequencyData(this.dataArray)
			const currentTime = Date.now()
			if (currentTime - lastUpdateTime > throttleInterval) {
				let sumOfSquares = 0
				for (let i = 0; i < this.dataArray.length; i++) {
					sumOfSquares += this.dataArray[i] * this.dataArray[i]
				}
				const rms = Math.sqrt(sumOfSquares / this.dataArray.length)
				// Normalize RMS (0-255) to a 0-1 scale, with 128 as a midpoint.
				// Multiply by 1.5 to boost the signal for better visualization.
				const normalizedLevel = (rms / 128) * 1.5
				this.options.onAudioLevel(Math.min(normalizedLevel, 1.0)) // Clamp at 1.0
				lastUpdateTime = currentTime
			}
			this.animationFrameId = requestAnimationFrame(analyze)
		}
		this.animationFrameId = requestAnimationFrame(analyze)
	}

	stopAnalysis() {
		if (this.animationFrameId) {
			console.log("[WebRTCClient] Stopping audio analysis loop.")
			cancelAnimationFrame(this.animationFrameId)
			this.animationFrameId = null
		}
		if (this.audioContext) {
			this.audioContext.close().catch(console.error)
			this.audioContext = null
		}
		this.analyser = null
		this.dataArray = null
	}

	disconnect() {
		// --- MODIFICATION: Make disconnect idempotent ---
		// If peerConnection is already null, we've already cleaned up.
		if (this.peerConnection === null) {
			console.log("[WebRTCClient] Already disconnected, skipping redundant disconnect call.");
			return;
		}
		console.log("[WebRTCClient] Disconnecting...")
		if (this.disconnectTimer) {
			clearTimeout(this.disconnectTimer)
			this.disconnectTimer = null
		}

		this.stopAnalysis()
		this.mediaStream?.getTracks().forEach((track) => track.stop())
		this.mediaStream = null
		this.peerConnection?.close()
		this.peerConnection = null
		this.dataChannel = null
		this.options.onDisconnected?.()
		console.log("[WebRTCClient] Disconnected and cleaned up.")
	}
}
