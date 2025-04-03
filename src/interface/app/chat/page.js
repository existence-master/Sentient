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

// Import the provider which renders VoiceBlobs
import { BackgroundCircleProvider } from "@components/voice-test/background-circle-provider"
// Removed ResetChat as it wasn't in your latest provided code for this page

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
				await fetchChatHistory()
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
	useEffect(() => {
		fetchUserDetails()
		fetchCurrentModel()
		fetchChatHistory()
		setupIpcListeners()
		return () => {
			eventListenersAdded.current = false
		}
	}, [])

	useEffect(() => {
		if (chatMode === "text") {
			chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
		}
	}, [messages, chatMode])

	useEffect(() => {
		if (chatMode === "text" && textareaRef.current) {
			handleInputChange({ target: textareaRef.current })
		}
	}, [chatMode, input])

	return (
		// MODIFIED: Removed flex, added relative positioning for absolute children
		<div className="h-screen bg-matteblack relative overflow-hidden dark">
			{/* Sidebar remains, its own z-index will handle overlay */}
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
			/>
			{/* MODIFIED: Main Content Area is now absolutely positioned to fill screen */}
			{/* It has a lower z-index (z-10) than the active sidebar (z-40) */}
			<div className="absolute inset-0 flex flex-col justify-center items-center h-full w-full bg-matteblack z-10">
				{/* Top Right Buttons (Positioned relative to this absolute container) */}
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

				{/* Conditional Content Container (centers its children within the absolute container) */}
				<div className="w-full h-full flex flex-col items-center justify-center p-5 pt-20 text-white">
					{isLoading ? (
						<div className="flex justify-center items-center h-full w-full">
							<IconLoader className="w-10 h-10 text-white animate-spin" />
						</div>
					) : chatMode === "text" ? (
						// Text Chat Mode UI (remains the same)
						<div className="w-full max-w-4xl h-full flex flex-col">
							{/* Message Display Area */}
							<div className="grow overflow-y-auto p-4 rounded-xl no-scrollbar mb-4 flex flex-col gap-4">
								{messages.length === 0 ? (
									<div className="font-Poppins h-full flex flex-col justify-center items-center text-gray-400">
										<p className="text-3xl text-white mb-4">
											Send a message to start
										</p>
									</div>
								) : (
									messages.map((msg) => (
										<div
											key={msg.id || Math.random()}
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
										className="flex-grow p-2 pr-28 rounded-lg bg-transparent text-base text-white focus:outline-none resize-none no-scrollbar overflow-y-auto"
										placeholder="Type your message..."
										style={{
											maxHeight: "150px",
											minHeight: "24px"
										}}
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
						// Container remains centered within the main absolute container
						<div className="flex flex-col items-center justify-center h-full w-full relative">
							{/* BackgroundCircleProvider renders VoiceBlobs, which will now be screen-centered */}
							<BackgroundCircleProvider />
						</div>
					)}
				</div>

				{/* Mode Toggle Button (Positioned relative to the absolute main container) */}
				{/* It needs a higher z-index than the main content (z-10) but lower than sidebar (z-40) */}
				{!isLoading && (
					<button
						onClick={handleToggleMode}
						// MODIFIED: Increased z-index to z-20 to be above main content but below sidebar
						className="absolute bottom-6 right-6 p-3 hover-button scale-100 hover:scale-110 cursor-pointer rounded-full text-white z-20"
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
