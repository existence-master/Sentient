"use client"
import React, { useState, useEffect, useRef, useCallback } from "react"
import ChatBubble from "@components/ChatBubble"
import ToolResultBubble from "@components/ToolResultBubble"
import Sidebar from "@components/Sidebar"
import {
	IconSend,
	IconRefresh,
	IconPlayerPlayFilled,
	IconLoader,
	IconMicrophone,
	IconMicrophoneOff,
	IconPhoneOff,
	IconPlayerStopFilled
} from "@tabler/icons-react"
import toast from "react-hot-toast"
import { WebRTCClient } from "@utils/WebRTCClient" // Adjust path as needed
import "webrtc-adapter" // Polyfill

// Debounce function
function debounce(func, wait) {
	let timeout
	return function executedFunction(...args) {
		const later = () => {
			clearTimeout(timeout)
			func(...args)
		}
		clearTimeout(timeout)
		timeout = setTimeout(later, wait)
	}
}

const Chat = () => {
	// Existing State
	const [messages, setMessages] = useState([])
	const [input, setInput] = useState("")
	const [userDetails, setUserDetails] = useState(null)
	const [thinking, setThinking] = useState(false) // For text chat thinking indicator
	const [serverStatus, setServerStatus] = useState(true) // Keep for general server check?
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	const [currentModel, setCurrentModel] = useState("")
	const [isLoading, setIsLoading] = useState(true) // For initial history load

	// Voice Mode State
	const [isVoiceMode, setIsVoiceMode] = useState(false)
	const [voiceState, setVoiceState] = useState("idle") // idle, connecting, listening, thinking, speaking, error
	const [isMuted, setIsMuted] = useState(false)
	const [isConnecting, setIsConnecting] = useState(false) // Explicit connecting state for UI

	// Refs
	const textareaRef = useRef(null)
	const chatEndRef = useRef(null)
	const textChatListenersAdded = useRef(false)
	const webRTCClientRef = useRef(null) // Ref for WebRTCClient instance
	const remoteAudioRef = useRef(null) // Ref for the <audio> element
	const wsRef = useRef(null) // Ref for the general WebSocket connection
	const wsListenersAdded = useRef(false) // Track WebSocket listeners separately

	// --- Fetching Data ---
	const fetchChatHistory = useCallback(async () => {
		// Fetch history regardless of mode, maybe filter display later?
		setIsLoading(true)
		console.log("Fetching chat history...")
		try {
			const response = await window.electron?.invoke("fetch-chat-history")
			if (response.status === 200) {
				setMessages(response.messages || []) // Ensure messages is always array
				console.log(
					"Chat history fetched:",
					(response.messages || []).length,
					"messages"
				)
			} else {
				toast.error(
					`Error fetching chat history: ${response.message || response.status}`
				)
				console.error("Error fetching chat history:", response)
			}
		} catch (error) {
			toast.error(`Error fetching chat history: ${error.message}`)
			console.error("Error fetching chat history:", error)
		} finally {
			setIsLoading(false)
		}
	}, []) // useCallback ensures stable reference

	const fetchUserDetails = useCallback(async () => {
		try {
			const response = await window.electron?.invoke("get-profile")
			setUserDetails(response)
		} catch (error) {
			toast.error(`Error fetching user details: ${error.message}`)
			console.error("Error fetching user details:", error)
		}
	}, [])

	const fetchCurrentModel = useCallback(async () => {
		// Replace with actual model fetching if dynamic
		setCurrentModel("llama3.2:3b / Orpheus") // Indicate both models
	}, [])

	// Initial data fetch
	useEffect(() => {
		fetchUserDetails()
		fetchCurrentModel()
		fetchChatHistory()
	}, [fetchUserDetails, fetchCurrentModel, fetchChatHistory]) // Add deps

	// Scroll to bottom for text chat
	useEffect(() => {
		if (!isVoiceMode && messages.length > 0) {
			// Debounce scrolling slightly to prevent jerky behavior during rapid updates
			const debouncedScroll = debounce(() => {
				chatEndRef.current?.scrollIntoView({
					behavior: "smooth",
					block: "end"
				})
			}, 100) // 100ms delay
			debouncedScroll()
		}
	}, [messages, isVoiceMode])

	// --- WebSocket Connection for State Updates & Notifications ---
	const connectWebSocket = useCallback(() => {
		if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
			console.log("WebSocket already connected.")
			return
		}
		console.log("Connecting WebSocket for notifications/state...")
		// Use ws:// or wss:// based on backend protocol if needed
		const wsUrl = `ws://localhost:5000/ws?clientId=electron-chat-${Date.now()}`
		wsRef.current = new WebSocket(wsUrl)

		wsRef.current.onopen = () => {
			console.log("WebSocket connected.")
			wsListenersAdded.current = true // Mark listeners as active for this connection
			// Start ping interval
			wsRef.current.pingInterval = setInterval(() => {
				if (
					wsRef.current &&
					wsRef.current.readyState === WebSocket.OPEN
				) {
					wsRef.current.send(JSON.stringify({ type: "ping" }))
				}
			}, 30000) // Send ping every 30s
		}

		wsRef.current.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data)
				// console.log("WebSocket message received:", data); // Log all messages for debug

				// Handle Voice State Updates
				if (data.type === "voice_state") {
					const newState = data.state
					if (
						["listening", "thinking", "speaking", "error"].includes(
							newState
						)
					) {
						console.log(
							`[WebSocket] Voice state update: ${newState}`
						)
						setVoiceState(newState)
					}
				}
				// Handle Task Notifications (keep existing logic if needed)
				else if (data.type === "task_completed") {
					console.log(
						`[WebSocket] Task Completed: ${data.description}`
					)
					toast.success(`Task completed: ${data.description}`)
					fetchChatHistory() // Refresh chat on task completion
				} else if (data.type === "task_error") {
					console.error(
						`[WebSocket] Task Error: ${data.description} - ${data.error}`
					)
					toast.error(
						`Task error: ${data.description} - ${data.error}`
					)
					fetchChatHistory() // Refresh chat even on error
				}
				// Handle Memory Notifications (keep existing logic if needed)
				else if (data.type === "memory_operation_completed") {
					console.log(
						`[WebSocket] Memory Op Completed: ${data.operation_id}`
					)
					// Maybe a subtle toast or log?
					// toast.success(`Memory updated: ${JSON.stringify(data.fact)}`);
				} else if (data.type === "memory_operation_error") {
					console.error(`[WebSocket] Memory Op Error: ${data.error}`)
					toast.error(`Memory update failed: ${data.error}`)
				}

				// Handle pong
				else if (data.type === "pong") {
					// console.log("WebSocket pong received."); // Optional log
				}
			} catch (error) {
				console.error(
					"Error processing WebSocket message:",
					error,
					"Raw data:",
					event.data
				)
			}
		}

		wsRef.current.onerror = (error) => {
			console.error("WebSocket error:", error)
			toast.error("Notification connection error.")
			// Attempt to reconnect after a delay
			if (wsRef.current && wsRef.current.pingInterval)
				clearInterval(wsRef.current.pingInterval)
			setTimeout(connectWebSocket, 5000) // Reconnect after 5s
		}

		wsRef.current.onclose = (event) => {
			console.log("WebSocket closed:", event.code, event.reason)
			wsListenersAdded.current = false // Mark listeners inactive
			if (wsRef.current && wsRef.current.pingInterval)
				clearInterval(wsRef.current.pingInterval)
			wsRef.current = null
			// Only reconnect if not closed intentionally (e.g., code 1000 is normal closure)
			if (event.code !== 1000) {
				toast.info(
					"Notification connection closed. Attempting to reconnect..."
				)
				setTimeout(connectWebSocket, 5000) // Reconnect after 5s
			}
		}
	}, [fetchChatHistory]) // Add fetchChatHistory dependency

	// Effect to establish WebSocket connection on mount
	useEffect(() => {
		connectWebSocket()
		// Cleanup on unmount
		return () => {
			if (wsRef.current) {
				console.log("Closing WebSocket connection on unmount.")
				wsRef.current.onclose = null // Prevent reconnect on intentional close
				if (wsRef.current.pingInterval)
					clearInterval(wsRef.current.pingInterval)
				wsRef.current.close(1000, "Component unmounting")
				wsRef.current = null
				wsListenersAdded.current = false
			}
		}
	}, [connectWebSocket]) // Run only on mount/unmount

	// --- Text Chat IPC Listeners ---
	useEffect(() => {
		if (window.electron && !textChatListenersAdded.current) {
			console.log("Setting up IPC listeners for text chat")
			const handleMessageStream = ({
				messageId,
				token,
				done,
				memoryUsed,
				agentsUsed,
				internetUsed,
				proUsed
			}) => {
				setThinking(false) // Stop text thinking indicator
				setMessages((prev) => {
					const existingMsgIndex = prev.findIndex(
						(msg) => msg.id === messageId && !msg.isUser
					)
					if (existingMsgIndex !== -1) {
						const updatedMessages = [...prev]
						updatedMessages[existingMsgIndex] = {
							...updatedMessages[existingMsgIndex],
							message:
								updatedMessages[existingMsgIndex].message +
								token,
							...(done && {
								memoryUsed:
									memoryUsed ??
									updatedMessages[existingMsgIndex]
										.memoryUsed,
								agentsUsed:
									agentsUsed ??
									updatedMessages[existingMsgIndex]
										.agentsUsed,
								internetUsed:
									internetUsed ??
									updatedMessages[existingMsgIndex]
										.internetUsed
							})
						}
						return updatedMessages
					} else if (token && !done) {
						// Only add new if it has token and not just the final empty done message
						return [
							...prev,
							{
								id: messageId,
								message: token,
								isUser: false,
								memoryUsed: false,
								agentsUsed: false,
								internetUsed: false,
								timestamp: new Date().toISOString(),
								type: "assistant"
							}
						]
					}
					return prev // No change
				})
				// Handle credit deduction if needed (check existing logic)
				if (done && proUsed) {
					console.log(
						"[IPC] Pro feature used, credits might need update."
					)
					// window.electron?.invoke("decrement-credits");
				}
				if (done) {
					console.log("[IPC] Stream finished for message:", messageId)
					// Optionally fetch history again if needed, but updates should be complete
					// fetchChatHistory();
				}
			}

			if (typeof window.electron.onMessageStream === "function") {
				const removeListener =
					window.electron.onMessageStream(handleMessageStream)
				textChatListenersAdded.current = true
				return () => {
					console.log("Cleaning up IPC message stream listener")
					if (typeof removeListener === "function") removeListener()
					textChatListenersAdded.current = false
				}
			} else {
				console.warn(
					"window.electron.onMessageStream is not available."
				)
			}
		}
	}, [])

	// --- Text Chat Functions ---
	const handleInputChange = (e) => {
		const value = e.target.value
		setInput(value)
		if (textareaRef.current) {
			textareaRef.current.style.height = "auto"
			textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
		}
	}

	const sendMessage = useCallback(async () => {
		if (input.trim() === "" || thinking) return
		const timestamp = new Date().toISOString()
		const newMessage = {
			message: input,
			isUser: true,
			id: `user-${Date.now()}`,
			timestamp: timestamp,
			type: "user",
			isVisible: true
		}
		setMessages((prev) => [...prev, newMessage])
		const messageToSend = input
		setInput("")
		setThinking(true) // Start text thinking indicator

		if (textareaRef.current) textareaRef.current.style.height = "auto"

		try {
			console.log("[IPC] Sending message:", messageToSend)
			// Use the globally assumed active chat ID or implement logic to select/pass ID
			const response = await window.electron?.invoke("send-message", {
				input: messageToSend,
				chat_id: "" // No need to pass the chat ID
			})
			console.log("[IPC] send-message response:", response)
			if (response.status !== 200) {
				toast.error(
					`Failed to send message: ${response.message || "Unknown error"}`
				)
				setMessages((prev) =>
					prev.filter((msg) => msg.id !== newMessage.id)
				) // Remove optimistic message
				setThinking(false)
			}
			// Thinking state is stopped by the stream listener
		} catch (error) {
			toast.error(`Error sending message: ${error.message}`)
			console.error("Error sending message:", error)
			setMessages((prev) =>
				prev.filter((msg) => msg.id !== newMessage.id)
			)
			setThinking(false)
		}
	}, [input, thinking]) // Add dependencies

	const clearChatHistory = useCallback(async () => {
		console.log("Attempting to clear chat history...")
		setMessages([]) // Optimistic UI update
		try {
			const response = await fetch(
				`http://localhost:5000/clear-chat-history`,
				{ method: "POST" }
			)
			if (!response.ok) {
				const errorData = await response.json().catch(() => ({}))
				throw new Error(
					`Failed to clear: ${response.status} ${errorData.message || ""}`
				)
			}
			toast.success("Chat history cleared.")
			console.log("Chat history cleared successfully.")
			// No need to fetch history again, it's empty
		} catch (error) {
			toast.error(`Error clearing chat history: ${error.message}`)
			console.error("Error clearing chat history:", error)
			fetchChatHistory() // Re-fetch if clearing failed to restore state
		}
	}, [fetchChatHistory])

	// --- Voice Chat Functions ---

	const handleWebRTCAudioStream = useCallback((stream) => {
		console.log("[WebRTC] Received remote audio stream.")
		if (remoteAudioRef.current) {
			remoteAudioRef.current.srcObject = stream
			remoteAudioRef.current
				.play()
				.catch((e) => console.error("Error playing remote audio:", e))
		}
	}, [])

	const handleWebRTCConnected = useCallback(() => {
		console.log("[WebRTC] Connection established callback triggered.")
		setIsConnecting(false)
		setVoiceState("listening") // <--- SET STATE DIRECTLY HERE
	}, [])

	const handleWebRTCDisconnected = useCallback(() => {
		console.log("[WebRTC] Disconnected callback triggered.")
		setIsConnecting(false)
		setIsVoiceMode(false) // Turn off voice mode UI on disconnect
		setVoiceState("idle")
		// webRTCClientRef.current = null; // Client instance is cleaned up in effect
	}, [])

	const handleWebRTCError = useCallback((error) => {
		console.error("[WebRTC] Error callback triggered:", error)
		toast.error(`Voice connection error: ${error.message}`)
		setIsConnecting(false)
		setIsVoiceMode(false) // Turn off voice mode UI on error
		setVoiceState("error")
	}, [])

	// Debounced audio level handler for smoother UI updates (optional)
	const handleWebRTCAudioLevel = useCallback(
		debounce((level) => {
			// You can use this 'level' (0-1) for UI visualization if needed
			// console.log("Audio Level:", level);
		}, 50),
		[]
	) // Update visualization every 50ms

	const startVoiceMode = useCallback(async () => {
		if (
			isConnecting ||
			(webRTCClientRef.current && webRTCClientRef.current.isConnected)
		) {
			console.warn("Already connecting or connected to voice.")
			return
		}
		console.log("Starting Voice Mode...")
		setIsConnecting(true)
		setIsVoiceMode(true)
		setVoiceState("connecting") // Initial state

		// Stop text input thinking indicator if it was active
		setThinking(false)

		// Instantiate and connect WebRTC Client
		if (!webRTCClientRef.current) {
			webRTCClientRef.current = new WebRTCClient({
				onConnected: handleWebRTCConnected,
				onDisconnected: handleWebRTCDisconnected,
				onAudioStream: handleWebRTCAudioStream,
				onAudioLevel: handleWebRTCAudioLevel,
				onError: handleWebRTCError
			})
		}

		try {
			await webRTCClientRef.current.connect()
			// Connection success is handled by the onConnected callback
		} catch (error) {
			// Error is handled by the onError callback
			console.error("Error initiating WebRTC connection:", error)
			// Ensure UI resets if connect throws synchronously (though onError should handle it)
			setIsConnecting(false)
			setIsVoiceMode(false)
			setVoiceState("error")
		}
	}, [
		isConnecting,
		handleWebRTCConnected,
		handleWebRTCDisconnected,
		handleWebRTCAudioStream,
		handleWebRTCAudioLevel,
		handleWebRTCError
	]) // Add dependencies

	const stopVoiceMode = useCallback(() => {
		console.log("Stopping Voice Mode...")
		if (webRTCClientRef.current) {
			webRTCClientRef.current.disconnect() // Disconnect will trigger callbacks
		}
		// Reset states immediately for faster UI feedback
		setIsConnecting(false)
		setIsVoiceMode(false)
		setVoiceState("idle")
		webRTCClientRef.current = null // Clear the ref after calling disconnect
	}, [])

	// Effect to cleanup WebRTC client on unmount if still active
	useEffect(() => {
		return () => {
			if (webRTCClientRef.current) {
				console.log("Cleaning up WebRTC client on component unmount.")
				webRTCClientRef.current.disconnect()
				webRTCClientRef.current = null
			}
		}
	}, [])

	const toggleMute = useCallback(() => {
		if (
			!isVoiceMode ||
			!webRTCClientRef.current ||
			!webRTCClientRef.current.isConnected
		)
			return
		const nextMutedState = !isMuted
		setIsMuted(nextMutedState)
		webRTCClientRef.current.setMuted(nextMutedState) // Call client method
		console.log(`Mic ${nextMutedState ? "muted" : "unmuted"}`)
	}, [isVoiceMode, isMuted])

	// --- Render Logic ---

	const renderVoiceMode = () => (
		<div className="flex-grow w-full flex flex-col justify-center items-center gap-8 text-white px-4">
			{/* Status Display */}
			<div className="text-center h-20">
				{" "}
				{/* Fixed height to prevent layout shifts */}
				<p className="text-5xl md:text-6xl font-semibold mb-3 capitalize">
					{voiceState}
				</p>
				<div className="h-6">
					{" "}
					{/* Placeholder for subtitle text */}
					{voiceState === "connecting" && (
						<IconLoader className="w-8 h-8 text-lightblue animate-spin mx-auto" />
					)}
					{voiceState === "listening" && (
						<p className="text-lg text-gray-400 animate-pulse">
							Listening...
						</p>
					)}
					{voiceState === "thinking" && (
						<IconLoader className="w-8 h-8 text-lightblue animate-spin mx-auto" />
					)}
					{voiceState === "speaking" && (
						<p className="text-lg text-gray-400">Speaking...</p>
					)}
					{voiceState === "error" && (
						<p className="text-lg text-red-500">
							Connection error.
						</p>
					)}
				</div>
			</div>

			{/* Controls */}
			<div className="flex gap-6 mt-8">
				<button
					onClick={toggleMute}
					className={`p-4 rounded-full transition-colors duration-200 ease-in-out ${
						isMuted
							? "bg-red-600 hover:bg-red-700"
							: "bg-gray-600 hover:bg-gray-500"
					}`}
					aria-label={
						isMuted ? "Unmute Microphone" : "Mute Microphone"
					}
					disabled={
						voiceState === "connecting" || voiceState === "error"
					} // Disable mute when not connected
				>
					{isMuted ? (
						<IconMicrophoneOff className="w-8 h-8 text-white" />
					) : (
						<IconMicrophone className="w-8 h-8 text-white" />
					)}
				</button>
				<button
					onClick={stopVoiceMode}
					className="p-4 rounded-full bg-red-600 hover:bg-red-700 transition-colors duration-200 ease-in-out"
					aria-label="End Voice Call"
				>
					<IconPhoneOff className="w-8 h-8 text-white" />
				</button>
			</div>
			{/* Hidden Audio Element for Playback */}
			<audio ref={remoteAudioRef} hidden />
		</div>
	)

	const renderTextMode = () => (
		<div className="flex-grow w-full flex flex-col overflow-hidden px-4 md:px-10 pb-4">
			{" "}
			{/* Changed structure for flex grow */}
			{/* Chat Messages Area */}
			<div className="flex-grow overflow-y-auto p-1 md:p-4 bg-matteblack rounded-xl no-scrollbar mb-3">
				{isLoading ? (
					<div className="flex justify-center items-center h-full">
						<IconLoader className="w-8 h-8 text-white animate-spin" />
					</div>
				) : messages.length === 0 ? (
					<div className="font-Poppins h-full flex flex-col justify-center items-center text-gray-500">
						<p className="text-3xl md:text-4xl text-white mb-4 text-center">
							Send a message or start voice chat!
						</p>
					</div>
				) : (
					messages
						.filter((msg) => msg.isVisible !== false)
						.map(
							(
								msg // Filter hidden messages
							) => (
								<div
									key={msg.id || msg.timestamp}
									className={`flex ${msg.isUser ? "justify-end" : "justify-start"} my-2`}
								>
									{msg.type === "tool_result" ? (
										<ToolResultBubble
											task={msg.task || "Task"}
											result={msg.message}
											memoryUsed={msg.memoryUsed}
											agentsUsed={msg.agentsUsed}
											internetUsed={msg.internetUsed}
										/>
									) : (
										<ChatBubble
											message={msg.message}
											isUser={msg.isUser}
											memoryUsed={msg.memoryUsed}
											agentsUsed={msg.agentsUsed}
											internetUsed={msg.internetUsed}
											timestamp={msg.timestamp}
										/>
									)}
								</div>
							)
						)
				)}
				{thinking && ( // Show thinking indicator for text chat
					<div className="flex justify-start items-center gap-2 mt-3 animate-pulse ml-4">
						<div className="bg-gray-600 rounded-full h-3 w-3"></div>
						<div className="bg-gray-600 rounded-full h-3 w-3"></div>
						<div className="bg-gray-600 rounded-full h-3 w-3"></div>
					</div>
				)}
				<div ref={chatEndRef} />
			</div>
			{/* Model Info */}
			<p className="text-gray-400 font-Poppins text-xs md:text-sm mb-2 px-1">
				Model:{" "}
				<span className="text-lightblue">
					{currentModel || "Loading..."}
				</span>
			</p>
			{/* Input Area */}
			<div className="relative flex flex-row gap-4 w-full px-4 py-1 bg-matteblack border border-gray-600 rounded-lg z-10">
				{" "}
				{/* Adjusted border */}
				<textarea
					ref={textareaRef}
					value={input}
					onChange={handleInputChange}
					onKeyDown={(e) => {
						if (e.key === "Enter" && !e.shiftKey) {
							e.preventDefault()
							sendMessage()
						}
					}}
					className="flex-grow p-3 pr-36 bg-transparent text-base md:text-lg text-white focus:outline-none resize-none no-scrollbar overflow-y-auto" // Adjusted padding/size
					placeholder="Start typing..."
					style={{ maxHeight: "150px", minHeight: "24px" }} // Adjusted height limits
					rows={1}
					disabled={thinking}
				/>
				<div className="absolute right-2 bottom-1 flex flex-row items-center gap-2">
					<button
						onClick={isVoiceMode ? stopVoiceMode : startVoiceMode}
						className={`p-2.5 hover-button scale-100 hover:scale-105 cursor-pointer rounded-full text-white ${isVoiceMode ? "bg-red-600 hover:bg-red-700" : "bg-gray-600 hover:bg-gray-500"} disabled:opacity-50 transition-all duration-200 ease-in-out`}
						disabled={isConnecting} // Disable while connecting
						aria-label={
							isVoiceMode ? "Stop Voice Chat" : "Start Voice Chat"
						}
					>
						{isConnecting ? (
							<IconLoader className="w-4 h-4 animate-spin" />
						) : isVoiceMode ? (
							<IconPlayerStopFilled className="w-4 h-4" />
						) : (
							<IconMicrophone className="w-4 h-4" />
						)}
					</button>
					<button
						onClick={sendMessage}
						className="p-2.5 hover-button scale-100 hover:scale-105 cursor-pointer rounded-full text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-50 transition-colors duration-200 ease-in-out"
						disabled={thinking || input.trim() === ""}
						aria-label="Send Message"
					>
						<IconSend className="w-4 h-4" />
					</button>
					<button
						onClick={clearChatHistory}
						className="p-2.5 rounded-full hover-button scale-100 cursor-pointer hover:scale-105 text-white bg-gray-600 hover:bg-gray-500 transition-colors duration-200 ease-in-out"
						aria-label="Clear Chat History"
					>
						<IconRefresh className="w-4 h-4" />
					</button>
				</div>
			</div>
		</div>
	)

	return (
		<div className="h-screen bg-matteblack flex relative overflow-hidden">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
			/>
			{/* Main Content Area */}
			<div className="flex-grow flex flex-col items-start h-full bg-matteblack">
				{isVoiceMode ? renderVoiceMode() : renderTextMode()}
			</div>
		</div>
	)
}

export default Chat
