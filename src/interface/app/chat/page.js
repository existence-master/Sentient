"use client"
// ADDED: useCallback and forwardRef imports (forwardRef needed for provider)
import React, {
	useState,
	useEffect,
	useRef,
	useCallback,
	forwardRef // Keep forwardRef import
} from "react"
import ChatBubble from "@components/ChatBubble"
import ToolResultBubble from "@components/ToolResultBubble"
import Sidebar from "@components/Sidebar"
// ADDED: Import the new TopControlBar component
import TopControlBar from "@components/TopControlBar"
import {
	IconSend,
	IconRefresh,
	IconLoader,
	// ADDED: Icons for call controls
	IconPhone,
	IconPhoneOff,
	IconMicrophone,
	IconMicrophoneOff
} from "@tabler/icons-react"
import toast from "react-hot-toast"

// MODIFIED: Import the DEFAULT export from background-circle-provider
import BackgroundCircleProvider from "@components/voice-test/background-circle-provider"

// REMOVED: Separate forwardRef wrapping was incorrect here. It's done inside the provider file.

const Chat = () => {
	// --- State Variables ---
	const [messages, setMessages] = useState([])
	const [input, setInput] = useState("")
	const [userDetails, setUserDetails] = useState("")
	const [thinking, setThinking] = useState(false)
	const [serverStatus, setServerStatus] = useState(true)
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	const [currentModel, setCurrentModel] = useState("")
	const [chatMode, setChatMode] = useState("voice") // Initial mode
	// MODIFIED: Initialize isLoading based on initial chat mode
	const [isLoading, setIsLoading] = useState(() => chatMode === "text")
	const [connectionStatus, setConnectionStatus] = useState("disconnected")
	// ADDED: State for mute status
	const [isMuted, setIsMuted] = useState(false)
	// ADDED: State for call duration timer
	const [callDuration, setCallDuration] = useState(0)
	// ADDED: State for available audio input devices
	const [audioInputDevices, setAudioInputDevices] = useState([])
	// ADDED: State for the selected audio input device ID
	const [selectedAudioInputDevice, setSelectedAudioInputDevice] = useState("")

	// --- Refs ---
	const textareaRef = useRef(null)
	const chatEndRef = useRef(null)
	const eventListenersAdded = useRef(false)
	// MODIFIED: Ref for the BackgroundCircleProvider (which is forwardRef'd)
	const backgroundCircleProviderRef = useRef(null)
	const ringtoneAudioRef = useRef(null)
	const connectedAudioRef = useRef(null)
	// ADDED: Ref to store the timer interval ID
	const timerIntervalRef = useRef(null)

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

	// ADDED: Helper to format seconds into MM:SS
	const formatDuration = (seconds) => {
		const mins = Math.floor(seconds / 60)
		const secs = seconds % 60
		return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
	}

	// --- Connection Status and Timer Handling ---
	const handleStatusChange = useCallback((status) => {
		console.log("Connection status changed:", status)
		setConnectionStatus(status) // Update state

		// Stop ringing sound if not connecting
		if (status !== "connecting" && ringtoneAudioRef.current) {
			ringtoneAudioRef.current.pause()
			ringtoneAudioRef.current.currentTime = 0
		}

		// Handle connected state: play sound, start timer
		if (status === "connected") {
			setCallDuration(0) // Reset duration
			if (connectedAudioRef.current) {
				connectedAudioRef.current.volume = 0.4 // Set volume
				connectedAudioRef.current
					.play()
					.catch((e) =>
						console.error("Error playing connected sound:", e)
					)
			}
			// Clear previous timer just in case
			if (timerIntervalRef.current) {
				clearInterval(timerIntervalRef.current)
			}
			// Start new timer
			timerIntervalRef.current = setInterval(() => {
				setCallDuration((prevDuration) => prevDuration + 1)
			}, 1000)
		} else {
			// Handle disconnected or connecting state: clear timer
			if (timerIntervalRef.current) {
				clearInterval(timerIntervalRef.current)
				timerIntervalRef.current = null
			}
			// Reset duration if fully disconnected
			if (status === "disconnected") {
				setCallDuration(0)
			}
		}
	}, []) // Keep dependency array empty as it doesn't depend on component state/props directly

	// --- Voice Control Handlers ---
	const handleStartVoice = async () => {
		if (
			connectionStatus !== "disconnected" ||
			!backgroundCircleProviderRef.current
		)
			return
		console.log("ChatPage: handleStartVoice called")
		setConnectionStatus("connecting") // Set connecting state

		// Play ringing sound
		if (ringtoneAudioRef.current) {
			ringtoneAudioRef.current.volume = 0.3 // Set volume
			ringtoneAudioRef.current.loop = true
			ringtoneAudioRef.current
				.play()
				.catch((e) => console.error("Error playing ringtone:", e))
		}
		try {
			// Call connect method via ref
			await backgroundCircleProviderRef.current?.connect()
		} catch (error) {
			// Handle connection errors (e.g., permissions)
			console.error("ChatPage: Error starting voice connection:", error)
			toast.error(
				`Failed to connect: ${error.message || "Unknown error"}`
			)
			handleStatusChange("disconnected") // Reset status via the callback
		}
	}

	const handleStopVoice = () => {
		if (
			connectionStatus === "disconnected" ||
			!backgroundCircleProviderRef.current
		)
			return
		console.log("ChatPage: handleStopVoice called")
		backgroundCircleProviderRef.current?.disconnect() // Call disconnect via ref
		// Status change and timer clearing handled by handleStatusChange triggered by provider's callback
	}

	// ADDED: Handler for toggling mute state
	const handleToggleMute = () => {
		const newMuteState = !isMuted
		setIsMuted(newMuteState)
		// Call the toggleMute method exposed by the provider via ref
		backgroundCircleProviderRef.current?.toggleMute(newMuteState)
		console.log("ChatPage: Toggled mute to:", newMuteState)
	}

	// ADDED: Handler for changing the selected audio input device
	const handleDeviceChange = (event) => {
		const deviceId = event.target.value
		console.log("ChatPage: Selected audio input device changed:", deviceId)
		setSelectedAudioInputDevice(deviceId)
		// If currently connected, inform user and potentially reconnect
		if (connectionStatus === "connected") {
			toast.error("Mic change needs reconnect. Restarting call...")
			// Simple reconnect: Stop then start after short delay
			handleStopVoice()
			setTimeout(() => {
				// Ensure state is ready before starting again
				if (connectionStatus === "disconnected") {
					handleStartVoice()
				} else {
					// If disconnect didn't happen quickly enough, wait and try again or handle error
					console.warn(
						"Delaying startVoice after device change as disconnect hasn't completed yet."
					)
					setTimeout(handleStartVoice, 300) // Longer delay
				}
			}, 150) // Short delay to allow disconnect to process
		}
	}

	// --- Data Fetching and IPC --- (Existing logic mostly okay)
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
			setIsLoading(false) // Always clear loading state
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
	// Initial setup effect (Corrected Structure)
	useEffect(() => {
		console.log("ChatPage: Initial Mount Effect - chatMode:", chatMode)
		fetchUserDetails()
		fetchCurrentModel()

		// ADDED: Fetch audio input devices on mount
		const getDevices = async () => {
			try {
				// Attempt to get devices via ref if available, fallback to static method
				let devices = []
				if (backgroundCircleProviderRef.current?.enumerateDevices) {
					devices =
						await backgroundCircleProviderRef.current.enumerateDevices()
				} else if (BackgroundCircleProvider.enumerateDevices) {
					// Check if static method exists on component type
					devices = await BackgroundCircleProvider.enumerateDevices()
				} else {
					// Fallback if neither ref nor static method is ready/available
					// Note: WebRTCClient itself has the static method, could call directly if imported
					// import { WebRTCClient } from '@utils/WebRTCClient'; // Would need this import
					// devices = await WebRTCClient.enumerateAudioInputDevices();
					console.warn(
						"Could not access enumerateDevices via provider ref or static property yet."
					)
				}

				if (devices && devices.length > 0) {
					console.log("ChatPage: Fetched audio devices:", devices)
					setAudioInputDevices(devices)
					// Set default selected device
					const defaultDevice =
						devices.find((d) => d.deviceId === "default") ||
						devices[0]
					if (defaultDevice && !selectedAudioInputDevice) {
						// Set only if not already set
						setSelectedAudioInputDevice(defaultDevice.deviceId)
						console.log(
							"ChatPage: Set default audio input device:",
							defaultDevice.deviceId
						)
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

		// Cleanup function for the effect
		return () => {
			console.log("ChatPage: Unmount Cleanup")
			eventListenersAdded.current = false
			// Clear timer interval
			if (timerIntervalRef.current) {
				clearInterval(timerIntervalRef.current)
			}
			// Disconnect voice if active
			if (
				backgroundCircleProviderRef.current &&
				connectionStatus !== "disconnected"
			) {
				console.log("ChatPage: Disconnecting voice on unmount")
				backgroundCircleProviderRef.current.disconnect()
			}
			// Stop and reset sounds
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
	}, []) // Dependency array IS correctly empty here for mount/unmount logic

	// Effect for scrolling and fetching history on mode switch (Corrected Structure)
	useEffect(() => {
		console.log(
			"ChatPage: Mode/Messages Effect - chatMode:",
			chatMode,
			"isLoading:",
			isLoading
		)
		if (chatMode === "text") {
			// Scroll logic
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

	// Effect for textarea resize (Corrected Structure)
	useEffect(() => {
		if (chatMode === "text" && textareaRef.current) {
			handleInputChange({ target: textareaRef.current })
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [chatMode, input]) // Correct dependencies

	// --- Component Return (JSX) ---
	return (
		<div className="h-screen bg-matteblack relative overflow-hidden dark">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
			/>
			{/* Render TopControlBar - Render immediately */}
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
					{/* Loading State */}
					{isLoading ? (
						<div className="flex justify-center items-center h-full w-full">
							<IconLoader className="w-10 h-10 text-white animate-spin" />
						</div>
					) : chatMode === "text" ? (
						// Text Chat Mode UI
						<div className="w-full max-w-4xl h-full flex flex-col">
							{/* Message Display Area */}
							<div className="grow overflow-y-auto p-4 rounded-xl no-scrollbar mb-4 flex flex-col gap-4">
								{messages.length === 0 && !thinking ? (
									<div className="font-Poppins h-full flex flex-col justify-center items-center text-gray-400">
										<p className="text-3xl text-white mb-4">
											{" "}
											Send a message to start{" "}
										</p>
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
									{" "}
									Model:{" "}
									<span className="text-lightblue">
										{" "}
										{currentModel}{" "}
									</span>{" "}
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
						// Voice Chat Mode UI
						<div className="flex flex-col items-center justify-center h-full w-full relative">
							{/* Background Blobs */}
							<BackgroundCircleProvider
								ref={backgroundCircleProviderRef}
								onStatusChange={handleStatusChange}
								connectionStatusProp={connectionStatus}
								initialMuteState={isMuted}
								selectedDeviceId={selectedAudioInputDevice}
							/>

							{/* Centered Call Controls Overlay */}
							<div className="absolute inset-0 flex flex-col items-center justify-center z-20 pointer-events-none">
								<div className="flex flex-col items-center pointer-events-auto bg-neutral-800/70 backdrop-blur-sm p-6 rounded-2xl shadow-xl">
									{/* Disconnected State */}
									{connectionStatus === "disconnected" && (
										<button
											onClick={handleStartVoice}
											className="p-4 bg-green-600 hover:bg-green-500 rounded-full text-white transition-colors duration-200"
											title="Start Call"
										>
											<IconPhone size={28} />
										</button>
									)}
									{/* Connecting State - Display handled by TopControlBar or potentially here */}
									{connectionStatus === "connecting" && (
										<div className="flex items-center justify-center p-4 text-yellow-400">
											<IconLoader
												size={28}
												className="animate-spin"
											/>
											<span className="ml-2 text-sm">
												Connecting...
											</span>
										</div>
									)}
									{/* Connected State */}
									{connectionStatus === "connected" && (
										<div className="flex flex-col items-center gap-4">
											{/* Duration */}
											<div className="text-sm font-mono text-neutral-300 mb-2">
												{formatDuration(callDuration)}
											</div>
											{/* Controls Row */}
											<div className="flex items-center justify-center gap-4">
												{/* Mute/Unmute */}
												<button
													onClick={handleToggleMute}
													className={`p-3 rounded-full transition-colors duration-200 ${isMuted ? "bg-red-600 hover:bg-red-500" : "bg-neutral-600 hover:bg-neutral-500"} text-white`}
													title={
														isMuted
															? "Unmute"
															: "Mute"
													}
												>
													{isMuted ? (
														<IconMicrophoneOff
															size={20}
														/>
													) : (
														<IconMicrophone
															size={20}
														/>
													)}
												</button>
												{/* Mic Select */}
												<select
													value={
														selectedAudioInputDevice
													}
													onChange={
														handleDeviceChange
													}
													className="bg-neutral-700 border border-neutral-600 text-white text-xs rounded px-2 py-1 focus:outline-none focus:border-lightblue appearance-none max-w-[150px] truncate"
													title="Select Microphone"
												>
													{audioInputDevices.length ===
														0 && (
														<option value="">
															No mics found
														</option>
													)}
													{audioInputDevices.map(
														(device) => (
															<option
																key={
																	device.deviceId
																}
																value={
																	device.deviceId
																}
															>
																{" "}
																{
																	device.label
																}{" "}
															</option>
														)
													)}
												</select>
												{/* Hang Up */}
												<button
													onClick={handleStopVoice}
													className="p-4 bg-red-600 hover:bg-red-500 rounded-full text-white transition-colors duration-200"
													title="Hang Up"
												>
													<IconPhoneOff size={28} />
												</button>
											</div>
										</div>
									)}
								</div>
							</div>
						</div>
					)}
				</div>
			</div>{" "}
			{/* Audio elements */}
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
} // THIS CLOSING BRACE was likely the cause of the syntax errors

export default Chat // Export statement should be outside the component function
