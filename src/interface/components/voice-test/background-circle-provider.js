"use client"

import { useState, useEffect, useRef, useCallback } from "react"
// REMOVED: BackgroundCircles import
// ADDED: VoiceBlobs import
import { VoiceBlobs } from "@components/voice-visualization/VoiceBlobs"
import { AIVoiceInput } from "@components/voice-test/ui/ai-voice-input"
import { WebRTCClient } from "@utils/WebRTCClient"
import React from "react"

export function BackgroundCircleProvider() {
	// REMOVED: currentVariant state
	const [isConnected, setIsConnected] = useState(false)
	const [webrtcClient, setWebrtcClient] = useState(null)
	const [audioLevel, setAudioLevel] = useState(0) // Audio level from 0 to 1
	const audioRef = useRef(null) // For playing back audio from server (if needed)

	// Memoize callbacks to prevent unnecessary re-renders
	const handleConnected = useCallback(() => setIsConnected(true), [])
	const handleDisconnected = useCallback(() => {
		setIsConnected(false)
		setAudioLevel(0) // Reset audio level on disconnect
	}, [])

	const handleAudioStream = useCallback((stream) => {
		// This handles the audio *coming back* from the server, if any
		if (audioRef.current) {
			audioRef.current.srcObject = stream
		}
	}, [])

	const handleAudioLevel = useCallback((level) => {
		// Apply some smoothing to the audio level to prevent jerky movements
		// Adjust smoothing factor (e.g., 0.7) as needed
		setAudioLevel((prev) => prev * 0.7 + level * 0.3)
	}, [])

	// REMOVED: variants and changeVariant logic

	useEffect(() => {
		// Initialize WebRTC client with memoized callbacks
		const client = new WebRTCClient({
			onConnected: handleConnected,
			onDisconnected: handleDisconnected,
			onAudioStream: handleAudioStream, // Handles incoming audio from backend
			onAudioLevel: handleAudioLevel // Receives mic level from client
		})
		setWebrtcClient(client)

		// Cleanup function: disconnect client when component unmounts
		return () => {
			console.log("Disconnecting WebRTC client on component unmount")
			client.disconnect()
		}
		// Dependencies ensure effect runs only when callbacks change (which they shouldn't due to useCallback)
	}, [
		handleConnected,
		handleDisconnected,
		handleAudioStream,
		handleAudioLevel
	])

	const handleStart = () => {
		console.log("Attempting to connect WebRTC client...")
		webrtcClient?.connect()
	}

	const handleStop = () => {
		console.log("Disconnecting WebRTC client...")
		webrtcClient?.disconnect()
	}

	return (
		// MODIFIED: Removed onClick handler, full height/width container
		<div className="relative w-full h-full flex items-center justify-center">
			{/* MODIFIED: Render VoiceBlobs instead of BackgroundCircles */}
			<VoiceBlobs audioLevel={audioLevel} isActive={isConnected} />
			{/* AIVoiceInput remains in the center overlay */}
			<div className="absolute inset-0 flex items-center justify-center z-10">
				{" "}
				{/* Ensure button is above blobs */}
				<AIVoiceInput
					onStart={handleStart}
					onStop={handleStop}
					isConnected={isConnected}
				/>
			</div>
			{/* Audio element for playback from server */}
			<audio ref={audioRef} autoPlay hidden />
		</div>
	)
}

export default BackgroundCircleProvider // MODIFIED: Export default

// REMOVED: COLOR_VARIANTS definition
