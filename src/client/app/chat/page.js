"use client"
import React, {
	useState,
	useEffect,
	useRef,
	useCallback
	// No longer need forwardRef here
} from "react"
import ChatBubble from "@components/ChatBubble"
import ToolResultBubble from "@components/ToolResultBubble"
import Sidebar from "@components/Sidebar"
import TopControlBar from "@components/TopControlBar"
import {
	IconSend,
	IconRefresh,
	IconLoader,
	IconPhone,
	IconPhoneOff
	// REMOVED: Mute/unmute icons are no longer needed here
	// IconMicrophone,
	// IconMicrophoneOff,
} from "@tabler/icons-react"
import toast from "react-hot-toast"

// Import the DEFAULT export (forwardRef-wrapped component)
import BackgroundCircleProvider from "@components/voice-test/background-circle-provider"

const Chat = () => {
	// --- State Variables ---
	const [messages, setMessages] = useState([])
	const [input, setInput] = useState("")
	const [userDetails, setUserDetails] = useState("")
	const [thinking, setThinking] = useState(false)
	const [serverStatus, setServerStatus] = useState(true)
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	const [currentModel, setCurrentModel] = useState("")
	const [chatMode, setChatMode] = useState("voice")
	const [isLoading, setIsLoading] = useState(() => chatMode === "text")
	const [connectionStatus, setConnectionStatus] = useState("disconnected")
	// REMOVED: isMuted state
	// REMOVED: callDuration state
	const [audioInputDevices, setAudioInputDevices] = useState([])
	const [selectedAudioInputDevice, setSelectedAudioInputDevice] = useState("")

	// --- Refs ---
	const textareaRef = useRef(null)
	const chatEndRef = useRef(null)
	const eventListenersAdded = useRef(false)
	const backgroundCircleProviderRef = useRef(null)
	const ringtoneAudioRef = useRef(null)
	const connectedAudioRef = useRef(null)
	// REMOVED: timerIntervalRef

	// --- Handlers ---
	const handleInputChange = (e) => {
		const value = e.target.value
		setInput(value)
		if (textareaRef.current) {
			textareaRef.current.style.height = "auto"
			textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
		}
	}

	const handleToggleMode = (targetMode) => {
		if (targetMode === chatMode) return
		setChatMode(targetMode)
		if (targetMode === "text" && connectionStatus !== "disconnected") {
			handleStopVoice()
		}
	}

	// REMOVED: formatDuration helper

	// --- Connection Status and Timer Handling ---
	const handleStatusChange = useCallback((status) => {
		console.log("Connection status changed:", status)
		setConnectionStatus(status)

		// Stop ringing sound
		if (status !== "connecting" && ringtoneAudioRef.current) {
			ringtoneAudioRef.current.pause()
			ringtoneAudioRef.current.currentTime = 0
		}

		// Play connected sound
		if (status === "connected") {
			// REMOVED: Reset duration
			if (connectedAudioRef.current) {
				connectedAudioRef.current.volume = 0.4 // Keep volume adjustment
				connectedAudioRef.current
					.play()
					.catch((e) =>
						console.error("Error playing connected sound:", e)
					)
			}
			// REMOVED: Timer starting logic
		} else {
			// REMOVED: Timer clearing logic
			// REMOVED: Reset duration logic
		}
	}, [])

	// --- Voice Control Handlers ---
	const handleStartVoice = async () => {
		// MODIFIED: No longer needs to pass deviceId to connect
		if (
			connectionStatus !== "disconnected" ||
			!backgroundCircleProviderRef.current
		)
			return
		console.log("ChatPage: handleStartVoice called")
		setConnectionStatus("connecting")
		if (ringtoneAudioRef.current) {
			ringtoneAudioRef.current.volume = 0.3
			ringtoneAudioRef.current.loop = true
			ringtoneAudioRef.current
				.play()
				.catch((e) => console.error("Error playing ringtone:", e))
		}
		try {
			// Call connect without arguments
			await backgroundCircleProviderRef.current?.connect()
		} catch (error) {
			console.error("ChatPage: Error starting voice connection:", error)
			toast.error(
				`Failed to connect: ${error.message || "Unknown error"}`
			)
			handleStatusChange("disconnected")
		}
	}

	const handleStopVoice = () => {
		if (
			connectionStatus === "disconnected" ||
			!backgroundCircleProviderRef.current
		)
			return
		console.log("ChatPage: handleStopVoice called")
		backgroundCircleProviderRef.current?.disconnect()
	}

	// REMOVED: handleToggleMute handler

	// MODIFIED: Device change handler - only updates state
	const handleDeviceChange = (event) => {
		const deviceId = event.target.value
		console.log("ChatPage: Selected audio input device changed:", deviceId)
		setSelectedAudioInputDevice(deviceId)
		// Inform user that a reconnect is needed for the change to take effect
		toast.success(
			"Microphone selection changed. Please restart the call to use the new device.",
			{ duration: 4000 }
		)
		// REMOVED: Automatic reconnect logic
	}

	// --- Data Fetching and IPC ---
	const fetchChatHistory = async () => {
		try {
			const response = await window.electron?.invoke("fetch-chat-history")
			if (response?.status === 200) {
				console.log(
					"fetchChatHistory: Received history",
					response.messages?.length
				)
				setMessages(response.messages || [])
			} else {
				console.error(
					"fetchChatHistory: Error status from IPC:",
					response?.status
				)
				toast.error("Error fetching chat history.")
				setMessages([])
			}
		} catch (error) {
			console.error("fetchChatHistory: Exception caught:", error)
			toast.error("Error fetching chat history.")
			setMessages([])
		} finally {
			console.log(
				"fetchChatHistory: Setting isLoading to false in finally block."
			)
			setIsLoading(false)
		}
	}

	const fetchUserDetails = async () => {
		try {
			const response = await window.electron?.invoke("get-profile")
			setUserDetails(response)
		} catch (error) {
			toast.error("Error fetching user details.")
		}
	}

	const fetchCurrentModel = async () => {
		setCurrentModel("llama3.2:3b")
	}

	const setupIpcListeners = () => {
		if (!eventListenersAdded.current && window.electron) {
			const handleMessageStream = ({ messageId, token }) => {
				setMessages((prev) => {
					const messageIndex = prev.findIndex(
						(msg) => msg.id === messageId
					)
					if (messageIndex === -1) {
						return [
							...prev,
							{
								id: messageId,
								message: token,
								isUser: false,
								memoryUsed: false,
								agentsUsed: false,
								internetUsed: false,
								type: "text"
							}
						]
					}
					return prev.map((msg, index) =>
						index === messageIndex
							? { ...msg, message: msg.message + token }
							: msg
					)
				})
			}
			window.electron.onMessageStream(handleMessageStream)
			eventListenersAdded.current = true
		}
	}

	const sendMessage = async () => {
		if (input.trim() === "" || chatMode !== "text") return

		const newMessage = {
			message: input,
			isUser: true,
			id: Date.now(),
			type: "text"
		}
		setMessages((prev) => [...prev, newMessage])
		setInput("")
		if (textareaRef.current) {
			textareaRef.current.style.height = "auto"
		}
		setThinking(true)
		setupIpcListeners()

		try {
			const response = await window.electron?.invoke("send-message", {
				input: newMessage.message
			})
			if (response.status === 200) {
				console.log(
					"Message send invoked, waiting for stream/completion."
				)
			} else {
				toast.error("Failed to send message via IPC.")
				setThinking(false)
			}
		} catch (error) {
			toast.error("Error sending message.")
			setThinking(false)
		} finally {
			setThinking(false)
			fetchChatHistory() // Fetch history after sending
		}
	}

	const clearChatHistory = async () => {
		try {
			const response = await window.electron?.invoke("clear-chat-history")
			if (response.status === 200) {
				setMessages([])
				if (chatMode === "text") setInput("")
				toast.success("Chat history cleared.")
			} else {
				toast.error("Failed to clear chat history via IPC.")
			}
		} catch (error) {
			toast.error("Error clearing chat history.")
		}
	}

	const reinitiateServer = async () => {
		setServerStatus(false)
		toast.loading("Restarting server...")
		try {
			const response = await window.electron?.invoke("reinitiate-server")
			if (response.status === 200) {
				toast.dismiss()
				toast.success("Server restarted. Fetching history...")
				if (chatMode === "text") {
					await fetchChatHistory()
				}
			} else {
				toast.dismiss()
				toast.error("Failed to restart server.")
			}
		} catch (error) {
			toast.dismiss()
			toast.error("Error restarting the server.")
		} finally {
			setServerStatus(true)
		}
	}

	// --- Effects ---
	// Initial setup effect
	useEffect(() => {
		console.log("ChatPage: Initial Mount Effect - chatMode:", chatMode)
		fetchUserDetails()
		fetchCurrentModel()

		// ADDED: Fetch audio input devices on mount directly using navigator
		const getDevices = async () => {
			try {
				if (
					!navigator.mediaDevices ||
					!navigator.mediaDevices.enumerateDevices
				) {
					console.warn("ChatPage: enumerateDevices() not supported.")
					setAudioInputDevices([])
					return
				}
				const devices = await navigator.mediaDevices.enumerateDevices()
				const audioInputDevices = devices.filter(
					(device) => device.kind === "audioinput"
				)

				if (audioInputDevices.length > 0) {
					console.log(
						"ChatPage: Fetched audio devices:",
						audioInputDevices
					)
					setAudioInputDevices(
						audioInputDevices.map((d) => ({
							// Store only needed info
							deviceId: d.deviceId,
							label:
								d.label ||
								`Microphone ${audioInputDevices.indexOf(d) + 1}`
						}))
					)
					// Set default selected device only if not already set
					if (!selectedAudioInputDevice) {
						const defaultDevice =
							audioInputDevices.find(
								(d) => d.deviceId === "default"
							) || audioInputDevices[0]
						if (defaultDevice) {
							setSelectedAudioInputDevice(defaultDevice.deviceId)
							console.log(
								"ChatPage: Set default audio input device:",
								defaultDevice.deviceId
							)
						}
					}
				} else {
					console.log("ChatPage: No audio input devices found.")
					setAudioInputDevices([])
					setSelectedAudioInputDevice("")
				}
			} catch (error) {
				console.error("ChatPage: Error fetching audio devices:", error)
				toast.error("Could not get microphone list.")
			}
		}
		getDevices() // Call the function

		// Fetch history only if starting in text mode
		if (chatMode === "text") {
			console.log(
				"ChatPage: Initial Mount - Fetching history (text mode)."
			)
			fetchChatHistory()
		} else {
			console.log(
				"ChatPage: Initial Mount - Setting isLoading false (voice mode)."
			)
			setIsLoading(false) // Ensure loader is off if starting in voice mode
		}
		setupIpcListeners()

		// Cleanup
		return () => {
			console.log("ChatPage: Unmount Cleanup")
			eventListenersAdded.current = false
			// REMOVED: Timer cleanup
			if (
				backgroundCircleProviderRef.current &&
				connectionStatus !== "disconnected"
			) {
				console.log("ChatPage: Disconnecting voice on unmount")
				backgroundCircleProviderRef.current.disconnect()
			}
			if (ringtoneAudioRef.current) {
				ringtoneAudioRef.current.pause()
				ringtoneAudioRef.current.currentTime = 0
			}
			if (connectedAudioRef.current) {
				connectedAudioRef.current.pause()
				connectedAudioRef.current.currentTime = 0
			}
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [])

	// Effect for scrolling and fetching history on mode switch
	useEffect(() => {
		console.log(
			"ChatPage: Mode/Messages Effect - chatMode:",
			chatMode,
			"isLoading:",
			isLoading
		)
		if (chatMode === "text") {
			if (chatEndRef.current) {
				chatEndRef.current.scrollIntoView({ behavior: "smooth" })
			}
			// Fetch logic
			if (!isLoading) {
				console.log(
					"ChatPage: Switched to text mode, fetching history."
				)
				fetchChatHistory()
			}
		}
	}, [chatMode]) // Correct dependencies

	// Effect for textarea resize
	useEffect(() => {
		if (chatMode === "text" && textareaRef.current) {
			handleInputChange({ target: textareaRef.current })
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [chatMode, input])

	// --- Component Return (JSX) ---
	return (
		<div className="h-screen bg-matteblack relative overflow-hidden dark">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
			/>

			<TopControlBar
				chatMode={chatMode}
				onToggleMode={handleToggleMode}
			/>

			{/* Main Content Area */}
			<div className="absolute inset-0 flex flex-col justify-center items-center h-full w-full bg-matteblack z-10 pt-20">
				{/* Top Right Buttons */}
				<div className="absolute top-5 right-5 z-20 flex gap-3">
					<button
						onClick={reinitiateServer}
						className="p-3 hover-button rounded-full text-white cursor-pointer"
						title="Restart Server"
					>
						{!serverStatus ? (
							<IconLoader className="w-4 h-4 text-white animate-spin" />
						) : (
							<IconRefresh className="w-4 h-4 text-white" />
						)}
					</button>
				</div>

				{/* Conditional Content Container */}
				<div className="w-full h-full flex flex-col items-center justify-center p-5 text-white">
					{isLoading ? ( // Simple loading check
						<div className="flex justify-center items-center h-full w-full">
							<IconLoader className="w-10 h-10 text-white animate-spin" />
						</div>
					) : chatMode === "text" ? (
						// --- Text Chat UI ---
						<div className="w-full max-w-4xl h-full flex flex-col">
							{/* Message Display */}
							<div className="grow overflow-y-auto p-4 rounded-xl no-scrollbar mb-4 flex flex-col gap-4">
								{messages.length === 0 && !thinking ? (
									<div className="font-Poppins h-full flex flex-col justify-center items-center text-gray-400">
										{" "}
										<p className="text-3xl text-white mb-4">
											{" "}
											Send a message to start{" "}
										</p>{" "}
									</div>
								) : (
									messages.map((msg) => (
										<div
											key={msg.id || Math.random()}
											className={`flex ${msg.isUser ? "justify-end" : "justify-start"} w-full`}
										>
											{" "}
											{msg.type === "tool_result" ? (
												<ToolResultBubble
													task={msg.task}
													result={msg.message}
													memoryUsed={msg.memoryUsed}
													agentsUsed={msg.agentsUsed}
													internetUsed={
														msg.internetUsed
													}
												/>
											) : (
												<ChatBubble
													message={msg.message}
													isUser={msg.isUser}
													memoryUsed={msg.memoryUsed}
													agentsUsed={msg.agentsUsed}
													internetUsed={
														msg.internetUsed
													}
												/>
											)}{" "}
										</div>
									))
								)}
								{thinking && (
									<div className="flex justify-start w-full mt-2">
										{" "}
										<div className="flex items-center gap-2 p-3 bg-gray-700 rounded-lg">
											{" "}
											<div className="bg-gray-400 rounded-full h-2 w-2 animate-pulse delay-75"></div>{" "}
											<div className="bg-gray-400 rounded-full h-2 w-2 animate-pulse delay-150"></div>{" "}
											<div className="bg-gray-400 rounded-full h-2 w-2 animate-pulse delay-300"></div>{" "}
										</div>{" "}
									</div>
								)}
								<div ref={chatEndRef} />
							</div>
							{/* Input Area */}
							<div className="w-full flex flex-col items-center">
								<p className="text-gray-400 font-Poppins text-xs mb-2">
									Check out our{" "}
									<a
										href="https://sentient-2.gitbook.io/docs"
										className="text-lightblue hover:text-lightblue/80"
									>
										docs
									</a>{" "}
									to see what Sentient can do.
								</p>
								<div className="relative w-full flex flex-row gap-4 items-end px-4 py-2 bg-matteblack border-[1px] border-gray-600 rounded-lg z-10">
									<textarea
										ref={textareaRef}
										value={input}
										onChange={handleInputChange}
										onKeyDown={(e) => {
											if (
												e.key === "Enter" &&
												!e.shiftKey
											) {
												e.preventDefault()
												sendMessage()
											}
										}}
										className="flex-grow p-2 pr-28 rounded-lg bg-transparent text-base text-white focus:outline-none resize-none no-scrollbar overflow-y-auto"
										placeholder="Type your message..."
										style={{
											maxHeight: "150px",
											minHeight: "24px"
										}}
										rows={1}
									/>
									<div className="absolute right-4 bottom-3 flex flex-row items-center gap-2">
										<button
											onClick={sendMessage}
											disabled={
												thinking || input.trim() === ""
											}
											className="p-2 hover-button scale-100 hover:scale-110 cursor-pointer rounded-full text-white disabled:opacity-50 disabled:cursor-not-allowed"
											title="Send Message"
										>
											{" "}
											<IconSend className="w-4 h-4 text-white" />{" "}
										</button>
										<button
											onClick={clearChatHistory}
											className="p-2 rounded-full hover-button scale-100 cursor-pointer hover:scale-110 text-white"
											title="Clear Chat History"
										>
											{" "}
											<IconRefresh className="w-4 h-4 text-white" />{" "}
										</button>
									</div>
								</div>
							</div>
						</div>
					) : (
						// --- Voice Chat UI ---
						<div className="flex flex-col items-center justify-center h-full w-full relative">
							{/* Background Blobs */}
							<BackgroundCircleProvider
								ref={backgroundCircleProviderRef}
								onStatusChange={handleStatusChange}
								connectionStatusProp={connectionStatus}
								// REMOVED: Props related to mute/device as client doesn't handle them now
								// initialMuteState={isMuted}
								// selectedDeviceId={selectedAudioInputDevice}
							/>

							{/* MODIFIED: Central Call Controls (Simpler) */}
							<div className="absolute inset-0 flex flex-col items-center justify-center z-20 pointer-events-none">
								<div className="flex flex-col items-center pointer-events-auto">
									{" "}
									{/* Removed background box for cleaner look */}
									{/* Disconnected State: Show Start Button */}
									{connectionStatus === "disconnected" && (
										<button
											onClick={handleStartVoice}
											className="p-4 bg-green-600 hover:bg-green-500 rounded-full text-white transition-colors duration-200 shadow-lg"
											title="Start Call"
										>
											<IconPhone size={32} />
										</button>
									)}
									{/* Connecting State: Show Loader */}
									{connectionStatus === "connecting" && (
										<div className="p-4 text-yellow-400">
											<IconLoader
												size={40}
												className="animate-spin"
											/>
										</div>
									)}
									{/* Connected State: Show Hang Up Button */}
									{connectionStatus === "connected" && (
										<button
											onClick={handleStopVoice}
											className="p-4 bg-red-600 hover:bg-red-500 rounded-full text-white transition-colors duration-200 shadow-lg"
											title="Hang Up"
										>
											<IconPhoneOff size={32} />
										</button>
									)}
									{/* REMOVED: Timer, Mute Button */}
								</div>
							</div>
						</div>
					)}
				</div>
			</div>

			{/* --- ADDED: Mic Selection Dropdown (Bottom Right) --- */}
			{/* Always visible when not loading */}
			{!isLoading && (
				<div className="absolute bottom-6 right-6 z-30">
					<select
						value={selectedAudioInputDevice}
						onChange={handleDeviceChange}
						className="bg-neutral-700/80 backdrop-blur-sm border border-neutral-600 text-white text-xs rounded px-3 py-2 focus:outline-none focus:border-lightblue appearance-none max-w-[200px] truncate shadow-lg"
						title="Select Microphone (Restart call to apply)" // Updated tooltip
					>
						{audioInputDevices.length === 0 ? (
							<option value="">Loading mics...</option> // Changed default message
						) : (
							audioInputDevices.map((device) => (
								<option
									key={device.deviceId}
									value={device.deviceId}
								>
									{device.label}
								</option>
							))
						)}
					</select>
				</div>
			)}

			{/* Audio elements (volume adjusted) */}
			<audio
				ref={ringtoneAudioRef}
				src="/audio/ringing.mp3"
				preload="auto"
				loop
			></audio>
			<audio
				ref={connectedAudioRef}
				src="/audio/connected.mp3"
				preload="auto"
			></audio>
		</div>
	)
}

export default Chat
