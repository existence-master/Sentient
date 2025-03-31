// src/interface/lib/WebRTCClient.js

/**
 * @typedef {Object} WebRTCClientOptions
 * @property {() => void} [onConnected]
 * @property {() => void} [onDisconnected]
 * @property {(stream: MediaStream) => void} [onAudioStream] - Callback for remote audio stream
 * @property {(level: number) => void} [onAudioLevel] - Callback for local audio level
 * @property {(error: Error) => void} [onError] - Callback for errors
 */

export class WebRTCClient {
	/** @type {RTCPeerConnection | null} */
	peerConnection = null
	/** @type {MediaStream | null} */
	mediaStream = null
	/** @type {WebRTCClientOptions} */
	options = {}
	/** @type {AudioContext | null} */
	audioContext = null
	/** @type {AnalyserNode | null} */
	analyser = null
	/** @type {Uint8Array | null} */
	dataArray = null
	/** @type {number | null} */
	animationFrameId = null
	/** @type {string | null} */
	webrtcId = null
	/** @type {boolean} */
	isConnected = false
	/** @type {MediaStreamAudioSourceNode | null} */
	audioSourceNode = null // To disconnect analyser

	/**
	 * @param {WebRTCClientOptions} options
	 */
	constructor(options = {}) {
		this.options = options
		this.webrtcId = "webrtc_" + Math.random().toString(36).substring(2, 9) // Generate unique ID
		console.log(`[WebRTCClient] Initialized with ID: ${this.webrtcId}`)
	}

	async connect() {
		if (this.isConnected || this.peerConnection) {
			console.warn("[WebRTCClient] Already connected or connecting.")
			return
		}
		console.log("[WebRTCClient] Attempting to connect...")

		try {
			// 1. Get user media (microphone)
			try {
				this.mediaStream = await navigator.mediaDevices.getUserMedia({
					audio: {
						// Consider adding noiseSuppression and echoCancellation
						noiseSuppression: true,
						echoCancellation: true
					},
					video: false
				})
				console.log(
					"[WebRTCClient] MediaStream tracks:",
					this.mediaStream.getTracks()
				)
			} catch (mediaError) {
				console.error("[WebRTCClient] Media access error:", mediaError)
				let message = "Failed to get microphone access."
				if (mediaError.name === "NotAllowedError") {
					message =
						"Microphone access denied. Please allow access in browser settings."
				} else if (mediaError.name === "NotFoundError") {
					message =
						"No microphone detected. Please connect a microphone."
				}
				this.handleError(new Error(message))
				this.disconnect() // Ensure cleanup on media error
				return // Stop connection attempt
			}

			// 2. Setup Peer Connection
			const configuration = {
				iceServers: [{ urls: "stun:stun.l.google.com:19302" }] // Use a public STUN server
			}
			this.peerConnection = new RTCPeerConnection(configuration)
			this.isConnected = false // Mark as not yet fully connected

			// 3. Setup Event Listeners for PeerConnection
			this.peerConnection.onicecandidate = (event) => {
				if (event.candidate) {
					console.log(
						"[WebRTCClient] ICE candidate:",
						event.candidate
					)
				} else {
					console.log(
						"[WebRTCClient] ICE candidate gathering complete."
					)
				}
			}

			this.peerConnection.onconnectionstatechange = () => {
				console.log(
					`[WebRTCClient] Connection state changed: ${this.peerConnection.connectionState}`
				)
				if (!this.peerConnection) return
				console.log(
					`[WebRTCClient] Connection state changed: ${this.peerConnection.connectionState}`
				)
				switch (this.peerConnection.connectionState) {
					case "connected":
						if (!this.isConnected) {
							this.isConnected = true
							console.log(
								"[WebRTCClient] Successfully connected."
							)
							if (this.options.onConnected)
								this.options.onConnected()
							this.setupAudioAnalysis() // Start analysis only when connected
						}
						break
					case "disconnected":
					case "failed":
						this.handleError(
							new Error(
								`Connection ${this.peerConnection.connectionState}`
							)
						)
						this.disconnect()
						break
					case "closed":
						this.isConnected = false
						if (this.options.onDisconnected)
							this.options.onDisconnected()
						break
				}
			}

			this.peerConnection.ontrack = (event) => {
				console.log("[WebRTCClient] Remote track received.")
				if (
					event.track.kind === "audio" &&
					this.options.onAudioStream
				) {
					// Pass the whole stream containing the track
					this.options.onAudioStream(event.streams[0])
				}
			}

			// 4. Add Local Tracks
			this.mediaStream.getTracks().forEach((track) => {
				this.peerConnection?.addTrack(track, this.mediaStream)
				console.log("[WebRTCClient] Local audio track added.")
			})

			// 5. Create Offer and Send to Backend
			const offer = await this.peerConnection.createOffer()
			await this.peerConnection.setLocalDescription(offer)
			console.log("[WebRTCClient] Local SDP:", offer.sdp)

			console.log("[WebRTCClient] Sending offer to backend...")
			// Use the FastRTC endpoint: /<mount_path>/offer
			const response = await fetch(
				"http://localhost:5000/voice/webrtc/offer",
				{
					// Correct endpoint
					method: "POST",
					headers: {
						"Content-Type": "application/json"
					},
					body: JSON.stringify({
						sdp: offer.sdp,
						type: offer.type,
						webrtc_id: this.webrtcId // Send the unique ID
						// FastRTC might expect other fields? Check its source/docs if needed.
					})
				}
			)

			this.peerConnection.getStats(null).then((stats) => {
				stats.forEach((report) => {
					if (
						report.type === "outbound-rtp" &&
						report.kind === "audio"
					) {
						console.log(
							"[WebRTCClient] Audio bytes sent:",
							report.bytesSent
						)
					}
				})
			})

			setInterval(() => {
				if (this.peerConnection && this.isConnected) {
					this.peerConnection.getStats(null).then((stats) => {
						stats.forEach((report) => {
							if (
								report.type === "outbound-rtp" &&
								report.kind === "audio"
							) {
								console.log(
									"[WebRTCClient] Audio bytes sent:",
									report.bytesSent
								)
							}
						})
					})
				}
			}, 5000)

			if (!response.ok) {
				const errorText = await response.text()
				throw new Error(
					`Backend offer exchange failed: ${response.status} ${errorText}`
				)
			}

			const answer = await response.json()
			console.log("[WebRTCClient] Received answer from backend.")

			if (!answer || !answer.sdp || !answer.type) {
				throw new Error("Invalid answer received from backend")
			}

			// 6. Set Remote Description
			await this.peerConnection.setRemoteDescription(
				new RTCSessionDescription(answer)
			)
			console.log("[WebRTCClient] Remote description set.")
			// Connection state change listener will handle final 'connected' state
		} catch (error) {
			console.error("[WebRTCClient] Connection error:", error)
			this.handleError(error)
			this.disconnect()
		}
	}

	setupAudioAnalysis() {
		if (!this.mediaStream || this.audioContext) return // Only setup once
		console.log("[WebRTCClient] Setting up audio analysis...")
		try {
			this.audioContext = new (window.AudioContext ||
				window.webkitAudioContext)()
			this.analyser = this.audioContext.createAnalyser()
			this.analyser.fftSize = 256 // Frequency data detail
			this.analyser.smoothingTimeConstant = 0.3 // Smoothing

			// Connect microphone stream to analyser
			this.audioSourceNode = this.audioContext.createMediaStreamSource(
				this.mediaStream
			)
			this.audioSourceNode.connect(this.analyser)

			// Buffer for frequency data
			const bufferLength = this.analyser.frequencyBinCount
			this.dataArray = new Uint8Array(bufferLength)

			this.startAnalysisLoop()
			console.log("[WebRTCClient] Audio analysis setup complete.")
		} catch (error) {
			console.error(
				"[WebRTCClient] Error setting up audio analysis:",
				error
			)
			this.handleError(new Error("Failed to setup audio analysis"))
			// Don't disconnect here, connection might still work
			this.stopAnalysis() // Clean up analysis parts
		}
	}

	startAnalysisLoop() {
		if (
			this.animationFrameId ||
			!this.analyser ||
			!this.dataArray ||
			!this.options.onAudioLevel
		)
			return

		let lastUpdateTime = 0
		const throttleInterval = 100 // ms - adjust for performance/responsiveness

		const analyze = () => {
			if (!this.analyser || !this.dataArray) {
				// Check if analyser still exists
				this.animationFrameId = null // Stop loop if analyser is gone
				return
			}
			this.analyser.getByteFrequencyData(this.dataArray) // Use frequency data

			const currentTime = performance.now()
			if (currentTime - lastUpdateTime > throttleInterval) {
				let sum = 0
				for (let i = 0; i < this.dataArray.length; i++) {
					sum += this.dataArray[i]
				}
				// Normalize average volume to 0-1 range
				// (Frequency data isn't direct volume, but correlates with energy)
				const averageLevel = sum / this.dataArray.length / 128.0 // Rough normalization
				this.options.onAudioLevel(Math.min(averageLevel * 1.5, 1.0)) // Amplify slightly and clamp
				lastUpdateTime = currentTime
			}

			this.animationFrameId = requestAnimationFrame(analyze)
		}

		this.animationFrameId = requestAnimationFrame(analyze)
	}

	stopAnalysis() {
		console.log("[WebRTCClient] Stopping audio analysis...")
		if (this.animationFrameId !== null) {
			cancelAnimationFrame(this.animationFrameId)
			this.animationFrameId = null
		}

		// Disconnect the analyser node
		if (this.audioSourceNode) {
			try {
				this.audioSourceNode.disconnect()
			} catch (e) {
				console.warn("Error disconnecting audio source node:", e)
			}
			this.audioSourceNode = null
		}
		this.analyser = null // Allow garbage collection

		// Close AudioContext - careful, can affect other audio on the page
		// It's often better to keep it alive if the app might use audio again.
		// Let's comment out closing for now, unless resource leak is proven.
		/*
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close().then(() => {
                 console.log('[WebRTCClient] AudioContext closed.');
                 this.audioContext = null;
            }).catch(e => console.error("Error closing AudioContext:", e));
        }
        */
		this.dataArray = null
	}

	disconnect() {
		console.log("[WebRTCClient] Disconnecting...")
		this.stopAnalysis() // Stop analysis first

		// Stop media tracks
		if (this.mediaStream) {
			this.mediaStream.getTracks().forEach((track) => track.stop())
			console.log("[WebRTCClient] Media tracks stopped.")
			this.mediaStream = null
		}

		// Close PeerConnection
		if (this.peerConnection) {
			// Remove listeners to prevent errors during closing
			this.peerConnection.onicecandidate = null
			this.peerConnection.onconnectionstatechange = null
			this.peerConnection.ontrack = null
			this.peerConnection.close()
			console.log("[WebRTCClient] PeerConnection closed.")
			this.peerConnection = null
		}

		this.isConnected = false
		if (this.options.onDisconnected) {
			this.options.onDisconnected()
		}
		console.log("[WebRTCClient] Disconnected.")
	}

	/**
	 * @param {boolean} muted
	 */
	setMuted(muted) {
		if (this.mediaStream) {
			this.mediaStream.getAudioTracks().forEach((track) => {
				track.enabled = !muted
			})
			console.log(`[WebRTCClient] Mic track enabled: ${!muted}`)
		}
	}

	/**
	 * @param {Error} error
	 */
	handleError(error) {
		console.error("[WebRTCClient] Error:", error)
		if (this.options.onError) {
			this.options.onError(error)
		}
	}
}
