"use client"

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
const BackgroundCircleProviderComponent = (
	{
		onStatusChange,
		connectionStatusProp,
		initialMuteState,
		selectedDeviceId
	},
	ref
) => {
	// ADDED: initialMuteState, selectedDeviceId props
	const [webrtcClient, setWebrtcClient] = useState(null)
	const [isConnected, setIsConnected] = useState(false)
	const [internalStatus, setInternalStatus] = useState("disconnected")
	const [audioLevel, setAudioLevel] = useState(0)
	const audioRef = useRef(null)
	// ADDED: State to track mute status internally, synchronized with parent
	const [isMuted, setIsMuted] = useState(initialMuteState || false)

	// Effect to sync internal status from prop
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

	// ADDED: Effect to sync internal mute state from prop (if parent changes it)
	useEffect(() => {
		if (initialMuteState !== undefined && initialMuteState !== isMuted) {
			console.log(
				`Provider: Syncing mute state from prop: ${initialMuteState}`
			)
			setIsMuted(initialMuteState)
			// Also apply to webrtcClient if it exists
			webrtcClient?.toggleMute(initialMuteState)
		}
	}, [initialMuteState, isMuted, webrtcClient]) // Include webrtcClient dependency

	// --- Callbacks for WebRTCClient ---
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
		console.log(
			"Provider: Initializing WebRTCClient instance with options:",
			{ initialMuteState, selectedDeviceId }
		)
		// MODIFIED: Pass down initialMuteState and selectedDeviceId to the client constructor
		const client = new WebRTCClient({
			onConnected: handleConnected,
			onDisconnected: handleDisconnected,
			onConnectError: handleConnectError,
			onAudioStream: handleAudioStream,
			onAudioLevel: handleAudioLevel,
			initialMuteState: isMuted, // Pass current internal mute state
			selectedDeviceId: selectedDeviceId // Pass selected device ID
		})
		setWebrtcClient(client)

		// Cleanup
		return () => {
			console.log(
				"Provider: Disconnecting WebRTC client on component unmount"
			)
			client.disconnect()
		}
		// MODIFIED: Added selectedDeviceId and isMuted to dependencies
	}, [
		handleConnected,
		handleDisconnected,
		handleConnectError,
		handleAudioStream,
		handleAudioLevel,
		selectedDeviceId,
		isMuted
	])

	// --- Expose connect/disconnect/toggleMute methods via ref ---
	useImperativeHandle(ref, () => ({
		connect: async () => {
			if (webrtcClient && internalStatus === "disconnected") {
				console.log("Provider: connect() called via ref")
				setInternalStatus("connecting")
				onStatusChange?.("connecting")
				try {
					// MODIFIED: Ensure client uses latest selectedDeviceId - RECREATE client instance on connect?
					// Alternative: Pass options directly to connect if client supports it
					// For simplicity now, assume client uses options passed during construction
					console.log(
						`Provider: Attempting connection with deviceId: ${selectedDeviceId}`
					)
					await webrtcClient.connect()
				} catch (error) {
					console.error(
						"Provider: Error during connect() call:",
						error
					)
					// Error handled by onConnectError callback
					throw error // Re-throw for parent
				}
			} else {
				console.warn(
					"Provider: connect() called but client not ready or already connecting/connected."
				)
			}
		},
		disconnect: () => {
			if (webrtcClient && internalStatus !== "disconnected") {
				console.log("Provider: disconnect() called via ref")
				webrtcClient.disconnect()
			} else {
				console.warn(
					"Provider: disconnect() called but client not connected or already disconnecting."
				)
			}
		},
		// ADDED: Expose toggleMute method
		toggleMute: (newMuteState) => {
			if (webrtcClient) {
				console.log(
					`Provider: toggleMute(${newMuteState}) called via ref`
				)
				webrtcClient.toggleMute(newMuteState)
				setIsMuted(newMuteState) // Update internal state as well
			} else {
				console.warn(
					"Provider: toggleMute called but webrtcClient is not available."
				)
			}
		},
		// ADDED: Expose device enumeration (delegating to static method)
		enumerateDevices: async () => {
			return WebRTCClient.enumerateAudioInputDevices()
		}
	}))

	// --- Render Logic ---
	return (
		<div className="relative w-full h-full flex items-center justify-center">
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

// Export unwrapped component type if needed (less common)
export { BackgroundCircleProviderComponent as BackgroundCircleProvider }
