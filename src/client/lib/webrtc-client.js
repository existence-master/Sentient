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

		try {
			this.peerConnection = new RTCPeerConnection({
				iceServers: this.iceServers
			})

			// --- MODIFICATION: Queue ICE candidates instead of sending immediately ---
			this.peerConnection.onicecandidate = (event) => {
				if (event.candidate) {
					this.iceCandidateQueue.push(event.candidate)
				}
			}

			this.peerConnection.oniceconnectionstatechange = (event) => {
				const state = this.peerConnection.iceConnectionState
			}

			this.peerConnection.onconnectionstatechange = (event) => {
				const state = this.peerConnection.connectionState

				// --- START MODIFICATION: Handle temporary disconnections ---
				// If connection is established or re-established, clear any pending disconnect timer.
				if (state === "connected") {
					if (this.disconnectTimer) {
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
			this.mediaStream = await navigator.mediaDevices.getUserMedia({
				audio: audioConstraints
			})

			this.setupAudioAnalysis()

			this.mediaStream.getTracks().forEach((track) => {
				if (this.peerConnection) {
					this.peerConnection.addTrack(track, this.mediaStream)
				}
			})

			this.peerConnection.addEventListener("track", (event) => {
				this.options.onAudioStream?.(event.streams[0])
			})

			this.dataChannel = this.peerConnection.createDataChannel("events")
			this.dataChannel.addEventListener("message", (event) => {
				try {
					const message = JSON.parse(event.data)
					this.options.onEvent?.(message)
				} catch (error) {
					console.error("Error parsing data channel message:", error)
				}
			})

			const offer = await this.peerConnection.createOffer()
			await this.peerConnection.setLocalDescription(offer)

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
			await this.peerConnection.setRemoteDescription(serverResponse)

			// --- MODIFICATION: Send all queued ICE candidates now ---
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

		} catch (error) {
			console.error("Error connecting WebRTC:", error)
			this.disconnect()
			throw error
		}
	}

	setupAudioAnalysis() {
		if (!this.mediaStream || !this.options.onAudioLevel) return
		try {
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
	}
}
