export class WebRTCClient {
	peerConnection = null
	mediaStream = null
	dataChannel = null
	options
	audioContext = null
	analyser = null
	dataArray = null
	animationFrameId = null
	// ADDED: Keep track of the audio track sender for muting
	audioSender = null
	// ADDED: Keep track of mute state internally
	isMuted = false

	constructor(options = {}) {
		this.options = options
		this.isMuted = options.initialMuteState || false // Initialize mute state
	}

	async connect() {
		try {
			this.peerConnection = new RTCPeerConnection()

			// --- Get user media ---
			try {
				// ADDED: Construct audio constraints based on options
				const audioConstraints = this.options.selectedDeviceId
					? { deviceId: { exact: this.options.selectedDeviceId } }
					: true // Use default if no device selected

				console.log(`[WebRTC] Requesting media with constraints:`, {
					audio: audioConstraints
				})
				this.mediaStream = await navigator.mediaDevices.getUserMedia({
					audio: audioConstraints // Use selected device if available
				})
				console.log("[WebRTC] Got user media stream.")

				// ADDED: Set initial track enabled state based on initial mute state
				this.mediaStream.getAudioTracks().forEach((track) => {
					track.enabled = !this.isMuted
					console.log(
						`[WebRTC] Initial track enabled state: ${track.enabled} (muted: ${this.isMuted})`
					)
				})
			} catch (mediaError) {
				console.error("[WebRTC] Media error:", mediaError)
				if (mediaError.name === "NotAllowedError") {
					throw new Error(
						"Microphone access denied. Please allow microphone access and try again."
					)
				} else if (mediaError.name === "NotFoundError") {
					throw new Error(
						`Microphone ${this.options.selectedDeviceId ? `(ID: ${this.options.selectedDeviceId})` : ""} not found. Please check connection or selection.`
					)
				} else if (mediaError.name === "OverconstrainedError") {
					throw new Error(
						`Could not satisfy constraints for microphone ${this.options.selectedDeviceId ? `(ID: ${this.options.selectedDeviceId})` : ""}. It might not support required features or is unavailable.`
					)
				} else {
					throw mediaError // Re-throw other errors
				}
			}

			this.setupAudioAnalysis() // Setup analysis after getting the stream

			// Add tracks to the peer connection and find the audio sender
			this.audioSender = null // Reset sender
			this.mediaStream.getTracks().forEach((track) => {
				if (this.peerConnection && track.kind === "audio") {
					// Store the sender for mute/unmute controls
					this.audioSender = this.peerConnection.addTrack(
						track,
						this.mediaStream
					)
					console.log("[WebRTC] Added audio track and stored sender.")
				} else if (this.peerConnection) {
					// Add other tracks if any (e.g., video, though unlikely here)
					this.peerConnection.addTrack(track, this.mediaStream)
				}
			})
			if (!this.audioSender) {
				console.warn(
					"[WebRTC] Audio sender was not found after adding tracks."
				)
			}

			// --- Event Listeners ---
			this.peerConnection.addEventListener("track", (event) => {
				console.log("[WebRTC] Received remote track:", event.track.kind)
				if (
					this.options.onAudioStream &&
					event.streams &&
					event.streams[0]
				) {
					console.log(
						"[WebRTC] Passing remote stream to onAudioStream callback."
					)
					this.options.onAudioStream(event.streams[0])
				}
			})

			// Handle data channel if needed (optional for voice-only)
			// this.dataChannel = this.peerConnection.createDataChannel("text");
			// this.dataChannel.addEventListener("message", (event) => { /* ... */ });

			// --- Offer/Answer Exchange ---
			console.log("[WebRTC] Creating offer...")
			const offer = await this.peerConnection.createOffer()
			await this.peerConnection.setLocalDescription(offer)
			console.log("[WebRTC] Local description (offer) set.")

			const webrtcId = Math.random().toString(36).substring(7)
			console.log(
				`[WebRTC] Sending offer to backend (ID: ${webrtcId})...`
			)
			const response = await fetch(
				"http://localhost:5000/voice/webrtc/offer", // Ensure this URL is correct
				{
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						Accept: "application/json"
					},
					mode: "cors",
					// credentials: "same-origin", // Usually not needed for localhost unless specific server setup
					body: JSON.stringify({
						sdp: offer.sdp,
						type: offer.type,
						webrtc_id: webrtcId // Include a unique ID if backend needs it
					})
				}
			)

			if (!response.ok) {
				const errorText = await response.text()
				console.error(
					`[WebRTC] Backend offer request failed (${response.status}):`,
					errorText
				)
				throw new Error(
					`Backend rejected offer: ${response.status} ${errorText || response.statusText}`
				)
			}

			const serverResponse = await response.json()
			console.log("[WebRTC] Received answer from backend.")
			await this.peerConnection.setRemoteDescription(serverResponse)
			console.log("[WebRTC] Remote description (answer) set.")

			console.log("[WebRTC] Connection established successfully.")
			if (this.options.onConnected) {
				this.options.onConnected()
			}
		} catch (error) {
			console.error("[WebRTC] Error during connect():", error)
			// ADDED: Call error callback if provided
			if (this.options.onConnectError) {
				this.options.onConnectError(error)
			}
			this.disconnect() // Ensure cleanup on error
			throw error // Re-throw the error for the caller
		}
	}

	setupAudioAnalysis() {
		if (!this.mediaStream || !this.mediaStream.getAudioTracks().length) {
			console.warn(
				"[WebRTC] No audio track available in mediaStream for analysis."
			)
			return
		}

		try {
			// Close existing context if any
			if (this.audioContext && this.audioContext.state !== "closed") {
				this.audioContext.close()
			}
			this.audioContext = new AudioContext()
			this.analyser = this.audioContext.createAnalyser()
			this.analyser.fftSize = 256 // Determines detail, power of 2 (32-32768)

			// Check if the stream is active before creating source
			if (!this.mediaStream.active) {
				console.warn(
					"[WebRTC] MediaStream is not active, cannot create source for analysis."
				)
				return
			}

			const source = this.audioContext.createMediaStreamSource(
				this.mediaStream
			)
			source.connect(this.analyser)
			// Note: Do NOT connect analyser to destination if you only want analysis
			// this.analyser.connect(this.audioContext.destination); // Don't do this unless you want feedback

			const bufferLength = this.analyser.frequencyBinCount // Always fftSize / 2
			this.dataArray = new Uint8Array(bufferLength)
			console.log("[WebRTC] Audio analysis setup complete.")
			this.startAnalysis() // Start analysis immediately after setup
		} catch (error) {
			console.error("[WebRTC] Error setting up audio analysis:", error)
			this.stopAnalysis() // Ensure cleanup on setup error
		}
	}

	startAnalysis() {
		if (!this.analyser || !this.dataArray || !this.options.onAudioLevel) {
			console.log(
				"[WebRTC] Analysis cannot start: Missing analyser, dataArray, or onAudioLevel callback."
			)
			return
		}
		if (this.animationFrameId !== null) {
			console.log("[WebRTC] Analysis already running.")
			return // Prevent multiple loops
		}

		let lastUpdateTime = 0
		const throttleInterval = 50 // Update more frequently (e.g., 50ms = 20fps)

		const analyze = () => {
			// Ensure analyser is still valid before using it
			if (!this.analyser) {
				console.log(
					"[WebRTC] Analyser became null, stopping analysis loop."
				)
				this.animationFrameId = null // Ensure loop stops
				return
			}

			try {
				this.analyser.getByteFrequencyData(this.dataArray) // Use frequency data
				// Or use TimeDomain data for volume: this.analyser.getByteTimeDomainData(this.dataArray);

				const currentTime = Date.now()
				if (currentTime - lastUpdateTime > throttleInterval) {
					// Calculate average volume level (RMS might be better, but average is simpler)
					let sum = 0
					for (let i = 0; i < this.dataArray.length; i++) {
						sum += this.dataArray[i]
					}
					const average = sum / this.dataArray.length
					// Normalize the average to a 0-1 range (approximate)
					// 255 is max value for Uint8Array elements
					const normalizedLevel = Math.min(1, average / 128) // Adjust divisor for sensitivity

					this.options.onAudioLevel(normalizedLevel)
					lastUpdateTime = currentTime
				}
			} catch (e) {
				console.error("[WebRTC] Error during audio analysis:", e)
				// Potentially stop analysis on error?
				// this.stopAnalysis();
			}

			this.animationFrameId = requestAnimationFrame(analyze) // Schedule next frame
		}

		console.log("[WebRTC] Starting audio analysis loop.")
		this.animationFrameId = requestAnimationFrame(analyze) // Start the loop
	}

	stopAnalysis() {
		if (this.animationFrameId !== null) {
			console.log("[WebRTC] Stopping audio analysis loop.")
			cancelAnimationFrame(this.animationFrameId)
			this.animationFrameId = null
		}

		if (this.audioContext && this.audioContext.state !== "closed") {
			console.log("[WebRTC] Closing AudioContext.")
			this.audioContext
				.close()
				.catch((e) => console.warn("Error closing AudioContext:", e))
		}

		this.audioContext = null
		this.analyser = null
		this.dataArray = null
		console.log("[WebRTC] Audio analysis stopped and resources released.")
	}

	// ADDED: Method to toggle mute state
	toggleMute(mute) {
		// Accepts the desired *new* mute state (true = mute, false = unmute)
		if (!this.audioSender || !this.audioSender.track) {
			console.warn(
				"[WebRTC] Cannot toggle mute: Audio sender or track not available."
			)
			return
		}
		if (this.audioSender.track.readyState === "ended") {
			console.warn("[WebRTC] Cannot toggle mute: Audio track has ended.")
			return
		}

		const shouldEnable = !mute // Track is enabled if *not* muted
		if (this.audioSender.track.enabled !== shouldEnable) {
			this.audioSender.track.enabled = shouldEnable
			this.isMuted = !shouldEnable // Update internal state
			console.log(
				`[WebRTC] Mic track ${shouldEnable ? "enabled" : "disabled"} (muted: ${this.isMuted})`
			)
		} else {
			console.log(
				`[WebRTC] Mic track already in desired state (enabled: ${shouldEnable}).`
			)
		}
	}

	disconnect() {
		console.log("[WebRTC] disconnect() called.")
		this.stopAnalysis() // Stop analysis first

		if (this.mediaStream) {
			console.log("[WebRTC] Stopping media tracks.")
			this.mediaStream.getTracks().forEach((track) => track.stop())
			this.mediaStream = null
		} else {
			console.log("[WebRTC] No mediaStream to stop.")
		}

		if (this.peerConnection) {
			// Remove event listeners? Usually not strictly necessary if closing immediately.
			console.log("[WebRTC] Closing PeerConnection.")
			this.peerConnection.close()
			this.peerConnection = null
		} else {
			console.log("[WebRTC] No PeerConnection to close.")
		}

		this.dataChannel = null // Assuming no data channel needed or handled elsewhere
		this.audioSender = null // Clear sender reference
		this.isMuted = this.options.initialMuteState || false // Reset mute state

		// Call the disconnected callback if provided
		if (this.options.onDisconnected) {
			console.log("[WebRTC] Calling onDisconnected callback.")
			this.options.onDisconnected()
		}
		console.log("[WebRTC] Disconnect process complete.")
	}

	// ADDED: Static method to enumerate devices without needing an instance
	static async enumerateAudioInputDevices() {
		try {
			if (
				!navigator.mediaDevices ||
				!navigator.mediaDevices.enumerateDevices
			) {
				console.warn("[WebRTC] enumerateDevices() not supported.")
				return []
			}
			const devices = await navigator.mediaDevices.enumerateDevices()
			const audioInputDevices = devices.filter(
				(device) => device.kind === "audioinput"
			)
			console.log(
				"[WebRTC] Enumerated audio input devices:",
				audioInputDevices
			)
			return audioInputDevices.map((device) => ({
				deviceId: device.deviceId,
				label:
					device.label ||
					`Microphone ${audioInputDevices.indexOf(device) + 1}` // Provide default label if missing
			}))
		} catch (error) {
			console.error("[WebRTC] Error enumerating audio devices:", error)
			return []
		}
	}
}
