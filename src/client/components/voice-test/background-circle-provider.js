"use client"

// MODIFIED: Removed initialMuteState from props, removed related useEffects and state
import {
	useState,
	useEffect,
	useRef,
	useCallback,
	useImperativeHandle,
	forwardRef
} from "react"
import { VoiceBlobs } from "@components/voice-visualization/VoiceBlobs"
import { WebRTCClient } from "@utils/WebRTCClient"
import React from "react"

// Component definition wrapped in forwardRef
// REMOVED: initialMuteState, selectedDeviceId props
const BackgroundCircleProviderComponent = (
	{ onStatusChange, connectionStatusProp },
	ref
) => {
	const [webrtcClient, setWebrtcClient] = useState(null)
	const [isConnected, setIsConnected] = useState(false)
	const [internalStatus, setInternalStatus] = useState("disconnected")
	const [audioLevel, setAudioLevel] = useState(0)
	const audioRef = useRef(null)
	// REMOVED: isMuted internal state

	// Effect to sync internal status from prop (no changes needed here)
	useEffect(() => {
		if (connectionStatusProp !== internalStatus) {
			console.log(
				`Provider: Syncing internal status from prop: ${connectionStatusProp}`
			)
			setInternalStatus(connectionStatusProp)
			setIsConnected(connectionStatusProp === "connected")
			if (connectionStatusProp === "disconnected") {
				setAudioLevel(0)
			}
		}
	}, [connectionStatusProp, internalStatus])

	// REMOVED: Effect to sync mute state from prop

	// --- Callbacks for WebRTCClient --- (no changes needed here)
	const handleConnected = useCallback(() => {
		console.log("Provider: WebRTC Connected")
		setIsConnected(true)
		setInternalStatus("connected")
		onStatusChange?.("connected")
	}, [onStatusChange])

	const handleDisconnected = useCallback(() => {
		console.log("Provider: WebRTC Disconnected")
		setIsConnected(false)
		setAudioLevel(0)
		setInternalStatus("disconnected")
		onStatusChange?.("disconnected")
	}, [onStatusChange])

	const handleConnectError = useCallback(
		(error) => {
			console.error("Provider: WebRTC Connection Error", error)
			setIsConnected(false)
			setAudioLevel(0)
			setInternalStatus("disconnected")
			onStatusChange?.("disconnected")
		},
		[onStatusChange]
	)

	const handleAudioStream = useCallback((stream) => {
		console.log("Provider: Received remote audio stream")
		if (audioRef.current) {
			audioRef.current.srcObject = stream
			audioRef.current
				.play()
				.catch((e) => console.warn("Remote audio playback failed:", e))
		}
	}, [])

	const handleAudioLevel = useCallback((level) => {
		setAudioLevel((prev) => prev * 0.7 + level * 0.3)
	}, [])

	// --- WebRTC Client Initialization ---
	useEffect(() => {
		// MODIFIED: Removed initialMuteState and selectedDeviceId from options passed to client
		console.log(
			"Provider: Initializing WebRTCClient instance with basic options"
		)
		const client = new WebRTCClient({
			onConnected: handleConnected,
			onDisconnected: handleDisconnected,
			onConnectError: handleConnectError,
			onAudioStream: handleAudioStream,
			onAudioLevel: handleAudioLevel
			// initialMuteState: isMuted, // REMOVED
			// selectedDeviceId: selectedDeviceId // REMOVED
		})
		setWebrtcClient(client)

		// Cleanup
		return () => {
			console.log(
				"Provider: Disconnecting WebRTC client on component unmount"
			)
			client.disconnect()
		}
		// MODIFIED: Removed selectedDeviceId and isMuted from dependencies
	}, [
		handleConnected,
		handleDisconnected,
		handleConnectError,
		handleAudioStream,
		handleAudioLevel
	])

	// --- Expose connect/disconnect methods via ref ---
	useImperativeHandle(ref, () => ({
		connect: async () => {
			// Connect method remains
			if (webrtcClient && internalStatus === "disconnected") {
				console.log("Provider: connect() called via ref")
				setInternalStatus("connecting")
				onStatusChange?.("connecting")
				try {
					// MODIFIED: Removed logging related to deviceId
					// console.log(`Provider: Attempting connection with deviceId: ${selectedDeviceId}`);
					await webrtcClient.connect() // Calls the reverted client's connect
				} catch (error) {
					console.error(
						"Provider: Error during connect() call:",
						error
					)
					throw error
				}
			} else {
				console.warn(
					"Provider: connect() called but client not ready or already connecting/connected."
				)
			}
		},
		disconnect: () => {
			// Disconnect method remains
			if (webrtcClient && internalStatus !== "disconnected") {
				console.log("Provider: disconnect() called via ref")
				webrtcClient.disconnect()
			} else {
				console.warn(
					"Provider: disconnect() called but client not connected or already disconnecting."
				)
			}
		}
		// REMOVED: toggleMute method is no longer exposed
		// REMOVED: enumerateDevices method is no longer exposed
	}))

	// --- Render Logic ---
	return (
		<div className="relative w-full h-full flex items-center justify-center">
			{/* VoiceBlobs rendering remains the same */}
			<VoiceBlobs
				audioLevel={audioLevel}
				isActive={internalStatus === "connected"}
				isConnecting={internalStatus === "connecting"}
			/>
			<audio ref={audioRef} hidden />
		</div>
	)
}

// Wrap the component function with forwardRef
const ForwardedBackgroundCircleProvider = forwardRef(
	BackgroundCircleProviderComponent
)

// Add display name
ForwardedBackgroundCircleProvider.displayName = "BackgroundCircleProvider"

// Export the forwardRef-wrapped component as default
export default ForwardedBackgroundCircleProvider

// Keep named export if needed (exports the unwrapped component)
export { BackgroundCircleProviderComponent as BackgroundCircleProvider }
