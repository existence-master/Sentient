"use client"
import React, { useState, useEffect, useRef } from "react"
import ChatBubble from "@components/ChatBubble"
import ToolResultBubble from "@components/ToolResultBubble"
import Sidebar from "@components/Sidebar"
import {
	IconSend,
	IconRefresh,
	IconLoader,
	IconMicrophone, // Import microphone icon
	IconKeyboard // Import keyboard icon
} from "@tabler/icons-react"
import toast from "react-hot-toast"

// Import components for Voice Mode
import { BackgroundCircleProvider } from "@components/voice-test/background-circle-provider"
import { ThemeToggle } from "@components/voice-test/ui/theme-toggle"
import { ResetChat } from "@components/voice-test/ui/reset-chat"

const Chat = () => {
	// --- State Variables ---
	const [messages, setMessages] = useState([])
	const [input, setInput] = useState("")
	const [userDetails, setUserDetails] = useState("")
	const [thinking, setThinking] = useState(false)
	const [serverStatus, setServerStatus] = useState(true)
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	const [currentModel, setCurrentModel] = useState("")
	const [isLoading, setIsLoading] = useState(true)
	const [chatMode, setChatMode] = useState("voice") // 'text' or 'voice', default 'voice'

	// --- Refs ---
	const textareaRef = useRef(null)
	const chatEndRef = useRef(null)
	const eventListenersAdded = useRef(false)

	// --- Handlers ---
	const handleInputChange = (e) => {
		const value = e.target.value
		setInput(value)
		// Auto-resize textarea only if it exists (i.e., in text mode)
		if (textareaRef.current) {
			textareaRef.current.style.height = "auto"
			textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
		}
	}

	const handleToggleMode = () => {
		setChatMode((prevMode) => (prevMode === "text" ? "voice" : "text"))
	}

	// --- Data Fetching and IPC ---
	const fetchChatHistory = async () => {
		// Only set loading for the initial fetch, not subsequent background fetches
		if (messages.length === 0) {
			setIsLoading(true)
		}
		try {
			const response = await window.electron?.invoke("fetch-chat-history")
			if (response.status === 200) setMessages(response.messages)
			else toast.error("Error fetching chat history.")
		} catch (error) {
			toast.error("Error fetching chat history.")
		} finally {
			// Only stop the main loading indicator on the initial fetch
			if (isLoading) {
				setIsLoading(false)
			}
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
		// Replace with actual logic if needed
		setCurrentModel("phi4-mini")
	}

	const setupIpcListeners = () => {
		if (!eventListenersAdded.current && window.electron) {
			const handleMessageStream = ({ messageId, token }) => {
				setMessages((prev) => {
					const messageIndex = prev.findIndex(
						(msg) => msg.id === messageId
					)
					if (messageIndex === -1) {
						// Ensure new messages are only added if we are in text mode or just switched from it
						// Or handle based on specific logic if voice can also trigger text stream updates
						return [
							...prev,
							{
								id: messageId,
								message: token,
								isUser: false,
								memoryUsed: false,
								agentsUsed: false,
								internetUsed: false,
								type: "text" // Assuming stream is text
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
			// Cleanup function might be needed if listeners should be removed/re-added on mode switch
			// return () => { window.electron.removeListener(...) } // Needs specific API if exists
		}
		// Consider if cleanup is needed when component unmounts or mode changes
	}

	const sendMessage = async () => {
		if (input.trim() === "" || chatMode !== "text") return // Only send in text mode

		const newMessage = {
			message: input,
			isUser: true,
			id: Date.now(),
			type: "text"
		}
		setMessages((prev) => [...prev, newMessage])
		setInput("")
		if (textareaRef.current) {
			// Reset textarea height
			textareaRef.current.style.height = "auto"
		}
		setThinking(true)

		setupIpcListeners() // Ensure listeners are set up before sending

		try {
			const response = await window.electron?.invoke("send-message", {
				input: newMessage.message
			})
			// The response might indicate completion, but streaming handles actual message display.
			// Fetching history might overwrite streamed content, consider flow carefully.
			// Maybe fetch history only *after* streaming seems complete, or rely solely on stream?
			// For now, keeping fetchChatHistory after potential stream completion for robustness.
			if (response.status === 200) {
				// Let the stream handle the AI response display. Fetching might be redundant or cause jumps.
				// await fetchChatHistory(); // Re-evaluate if this is needed here or after stream ends
				console.log(
					"Message send invoked, waiting for stream/completion."
				)
			} else {
				toast.error("Failed to send message via IPC.")
				setThinking(false) // Stop thinking if initial send fails
			}
		} catch (error) {
			toast.error("Error sending message.")
			setThinking(false) // Stop thinking on error
		} finally {
			// Thinking state should ideally be turned off when the *stream* ends, not immediately here.
			// For now, let's keep it simple and turn it off here, assuming stream follows quickly.
			// A more robust solution would involve an 'end-of-stream' event from IPC.
			setThinking(false) // Simplified: turn off thinking after invoke call
		}
	}

	const clearChatHistory = async () => {
		// Confirmation might be good here
		try {
			// Assuming clear should work regardless of mode, using IPC if available
			const response = await window.electron?.invoke("clear-chat-history")
			if (response.status === 200) {
				setMessages([])
				// Optionally clear input if in text mode
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
		toast.loading("Restarting server...") // Give user feedback
		try {
			// Assuming reinitiate might clear history or require a refresh
			const response = await window.electron?.invoke("reinitiate-server") // Use a dedicated IPC call if possible
			if (response.status === 200) {
				toast.dismiss()
				toast.success("Server restarted. Fetching history...")
				await fetchChatHistory() // Fetch fresh history after restart
			} else {
				toast.dismiss()
				toast.error("Failed to restart server.")
			}
		} catch (error) {
			toast.dismiss()
			toast.error("Error restarting the server.")
		} finally {
			setServerStatus(true) // Set status back regardless of success/failure for now
		}
	}

	// --- Effects ---
	useEffect(() => {
		fetchUserDetails()
		fetchCurrentModel()
		fetchChatHistory() // Initial fetch
		setupIpcListeners() // Setup listeners on mount

		// Cleanup listeners on unmount
		// This assumes a simple on/off, if electron API allows removal, use that.
		return () => {
			eventListenersAdded.current = false // Reset flag
			// Add specific listener removal logic here if available from electron preload script
		}
	}, []) // Run once on mount

	useEffect(() => {
		// Scroll to bottom only in text mode when messages change
		if (chatMode === "text") {
			chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
		}
	}, [messages, chatMode]) // Add chatMode dependency

	// Optional: Background refresh (consider if needed with IPC streaming)
	// useEffect(() => {
	// 	const intervalId = setInterval(fetchChatHistory, 60000) // Refresh every 60 seconds
	// 	return () => clearInterval(intervalId)
	// }, [])

	// Adjust textarea height when switching to text mode and input changes
	useEffect(() => {
		if (chatMode === "text" && textareaRef.current) {
			handleInputChange({ target: textareaRef.current }) // Trigger resize based on current content
		}
	}, [chatMode, input]) // Rerun when mode changes or input changes in text mode

	return (
		<div className="h-screen bg-matteblack flex relative overflow-hidden">
			{" "}
			{/* Prevent body scroll */}
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
			/>
			{/* Main Content Area */}
			<div className="flex-grow flex flex-col justify-center items-center h-full bg-matteblack relative">
				{" "}
				{/* Use flex-grow and center content */}
				{/* Top Right Buttons (Always Visible) */}
				<div className="absolute top-5 right-5 z-20 flex gap-3">
					{/* Server Re-initiate Button */}
					<button
						onClick={reinitiateServer}
						className="p-3 hover-button rounded-full text-white cursor-pointer"
						title="Restart Server"
					>
						{!serverStatus ? (
							<IconLoader className="w-4 h-4 text-white animate-spin" />
						) : (
							// Using Refresh icon might be more intuitive for restarting
							<IconRefresh className="w-4 h-4 text-white" />
						)}
					</button>
					{/* Theme Toggle (Show based on Voice Mode context or always?) */}
					{/* Let's place it here for consistency */}
					<ThemeToggle />
				</div>
				{/* Conditional Content: Loading, Text Chat, or Voice Chat */}
				<div className="w-full h-full flex flex-col items-center justify-center p-5 pt-20">
					{" "}
					{/* Added padding top */}
					{isLoading ? (
						// Loading State
						<div className="flex justify-center items-center h-full w-full">
							<IconLoader className="w-10 h-10 text-white animate-spin" />
						</div>
					) : chatMode === "text" ? (
						// Text Chat Mode UI
						<div className="w-full max-w-4xl h-full flex flex-col">
							{" "}
							{/* Max width for readability */}
							{/* Message Display Area */}
							<div className="grow overflow-y-auto p-4 rounded-xl no-scrollbar mb-4 flex flex-col gap-4">
								{messages.length === 0 ? (
									<div className="font-Poppins h-full flex flex-col justify-center items-center text-gray-500">
										<p className="text-3xl text-white mb-4">
											Send a message to start
										</p>
									</div>
								) : (
									messages.map((msg) => (
										<div
											key={msg.id || Math.random()} // Use a more stable key if possible
											className={`flex ${msg.isUser ? "justify-end" : "justify-start"} w-full`}
										>
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
											)}
										</div>
									))
								)}
								{/* Thinking Indicator */}
								{thinking && (
									<div className="flex justify-start w-full mt-2">
										<div className="flex items-center gap-2 p-3 bg-gray-700 rounded-lg">
											<div className="bg-gray-400 rounded-full h-2 w-2 animate-pulse delay-75"></div>
											<div className="bg-gray-400 rounded-full h-2 w-2 animate-pulse delay-150"></div>
											<div className="bg-gray-400 rounded-full h-2 w-2 animate-pulse delay-300"></div>
										</div>
									</div>
								)}
								<div ref={chatEndRef} /> {/* Scroll target */}
							</div>
							{/* Model Info and Input Area Container */}
							<div className="w-full flex flex-col items-center">
								<p className="text-gray-400 font-Poppins text-xs mb-2">
									Model:{" "}
									<span className="text-lightblue">
										{currentModel}
									</span>
								</p>
								{/* Input Area */}
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
										className="flex-grow p-2 pr-28 rounded-lg bg-transparent text-base text-white focus:outline-none resize-none no-scrollbar overflow-y-auto" // Adjusted padding-right
										placeholder="Type your message..."
										style={{
											maxHeight: "150px",
											minHeight: "24px"
										}} // Adjusted heights
										rows={1}
									/>
									{/* Buttons inside input area */}
									<div className="absolute right-4 bottom-3 flex flex-row items-center gap-2">
										<button
											onClick={sendMessage}
											disabled={
												thinking || input.trim() === ""
											}
											className="p-2 hover-button scale-100 hover:scale-110 cursor-pointer rounded-full text-white disabled:opacity-50 disabled:cursor-not-allowed"
											title="Send Message"
										>
											<IconSend className="w-4 h-4 text-white" />
										</button>
										<button
											onClick={clearChatHistory}
											className="p-2 rounded-full hover-button scale-100 cursor-pointer hover:scale-110 text-white"
											title="Clear Chat History"
										>
											<IconRefresh className="w-4 h-4 text-white" />
										</button>
									</div>
								</div>
							</div>
						</div>
					) : (
						// Voice Chat Mode UI
						<div className="flex flex-col items-center justify-center h-full w-full relative">
							<BackgroundCircleProvider />
							{/* ResetChat for voice mode - placed bottom center or corner */}
							<div className="absolute bottom-24 left-1/2 transform -translate-x-1/2 z-10">
								<ResetChat />
							</div>
							{/* ThemeToggle is now top right */}
						</div>
					)}
				</div>
				{/* Mode Toggle Button (Always Visible except loading) */}
				{!isLoading && (
					<button
						onClick={handleToggleMode}
						className="absolute bottom-6 right-6 p-3 hover-button scale-100 hover:scale-110 cursor-pointer rounded-full text-white z-20" // Ensure high z-index
						title={
							chatMode === "text"
								? "Switch to Voice Mode"
								: "Switch to Text Mode"
						}
					>
						{chatMode === "text" ? (
							<IconMicrophone className="w-5 h-5 text-white" />
						) : (
							<IconKeyboard className="w-5 h-5 text-white" />
						)}
					</button>
				)}
			</div>{" "}
			{/* End Main Content Area */}
		</div> // End Root Div
	)
}

export default Chat
