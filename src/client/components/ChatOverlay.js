"use client"
import React, { useState, useEffect, useRef, useCallback } from "react"
import {
	IconSend,
	IconLoader,
	IconPlayerStopFilled,
	IconWorldSearch,
	IconCloud,
	IconNews,
	IconMap,
	IconShoppingCart,
	IconX
} from "@tabler/icons-react"
import toast from "react-hot-toast"
import { cn } from "@utils/cn"
import { Tooltip } from "react-tooltip"
import { motion, AnimatePresence } from "framer-motion"
import ChatBubble from "@components/ChatBubble"

// Simplified Switch component from old chat page
const Switch = ({ checked, onCheckedChange }) => (
	<button
		type="button"
		role="switch"
		aria-checked={checked}
		onClick={() => onCheckedChange(!checked)}
		className={cn(
			// Removed focus ring overrides to use global style
			"relative inline-flex h-[18px] w-[34px] flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out",
			checked ? "bg-[var(--color-accent-blue)]" : "bg-neutral-600"
		)}
	>
		<span
			aria-hidden="true"
			className={cn(
				"pointer-events-none inline-block h-[14px] w-[14px] transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out",
				checked ? "translate-x-[16px]" : "translate-x-0"
			)}
		/>
	</button>
)

const ChatOverlay = ({ onClose }) => {
	const [messages, setMessages] = useState([])
	const [input, setInput] = useState("")
	const [thinking, setThinking] = useState(false)
	const [isInternetEnabled, setInternetEnabled] = useState(false)
	const [isWeatherEnabled, setWeatherEnabled] = useState(false)
	const [isNewsEnabled, setNewsEnabled] = useState(false)
	const [isMapsEnabled, setMapsEnabled] = useState(false)
	const [isShoppingEnabled, setShoppingEnabled] = useState(false)
	const [stepEvents, setStepEvents] = useState([]) // NEW: for backend step streaming
	const textareaRef = useRef(null)
	const chatEndRef = useRef(null)
	const abortControllerRef = useRef(null) // To abort fetch requests
	const scrollContainerRef = useRef(null) // For scrolling
	// Reset state when overlay is closed
	// This effect now correctly handles cleanup on unmount
	useEffect(() => {
		return () => {
			if (abortControllerRef.current) {
				abortControllerRef.current.abort()
			}
		}
	}, [])

	const handleInputChange = (e) => {
		const value = e.target.value
		setInput(value)
		if (textareaRef.current) {
			textareaRef.current.style.height = "auto"
			textareaRef.current.style.height = `${Math.min(
				textareaRef.current.scrollHeight,
				200
			)}px`
		}
	}

	const sendMessage = async () => {
		if (input.trim() === "") return
		const newUserMessage = {
			role: "user",
			content: input,
			id: Date.now()
		}

		// Add new message and create history for API call
		const updatedMessages = [...messages, newUserMessage]
		setMessages(updatedMessages)
		setStepEvents([]) // NEW: clear step log for new session

		setInput("")
		if (textareaRef.current) textareaRef.current.style.height = "auto"
		setThinking(true)
		abortControllerRef.current = new AbortController()

		// Format for backend
		const apiMessages = updatedMessages.map(({ role, content }) => ({
			role,
			content
		}))

		try {
			// Make the streaming call to our Next.js API route.
			const response = await fetch(
				"/api/chat/message", // Use the API route which proxies to the backend
				{
					method: "POST",
					headers: {
						"Content-Type": "application/json"
					},
					body: JSON.stringify({
						// The API route will forward these to the backend
						messages: apiMessages,
						enable_internet: isInternetEnabled,
						enable_weather: isWeatherEnabled,
						enable_news: isNewsEnabled,
						enable_maps: isMapsEnabled,
						enable_shopping: isShoppingEnabled
					}),
					signal: abortControllerRef.current.signal
				}
			)
			if (!response.ok || !response.body) {
				const errorData = await response
					.json()
					.catch(() => ({ message: "An unknown error occurred" }))
				throw new Error(
					errorData.message || "Failed to get streaming response"
				)
			}
			const reader = response.body.getReader()
			const decoder = new TextDecoder()
			let assistantMessageId = null
			let buffer = ""
			while (true) {
				const { value, done } = await reader.read()
				if (done) break
				buffer += decoder.decode(value, { stream: true })
				let newlineIndex
				while ((newlineIndex = buffer.indexOf("\n")) !== -1) {
					const line = buffer.slice(0, newlineIndex)
					buffer = buffer.slice(newlineIndex + 1)
					if (line.trim() === "") continue // Skip empty lines

					// Log the raw line received from the stream
					console.log("[ChatStream] Received raw line:", line)

					try {
						const parsed = JSON.parse(line)
						// Log the parsed JavaScript object
						console.log("[ChatStream] Parsed data:", parsed)
						if (parsed.type === "error") {
							toast.error(`An error occurred: ${parsed.message}`)
							continue
						}
						if (parsed.type === "step") {
							setStepEvents((prev) => [...prev, parsed])
							continue
						}
						if (parsed.type === "assistantStream") {
							const token = parsed.token || parsed.message || ""
							assistantMessageId = parsed.messageId
							setMessages((prev) => {
								const existingMsgIndex = prev.findIndex(
									(msg) => msg.id === assistantMessageId
								)
								if (existingMsgIndex !== -1) {
									return prev.map((msg, index) =>
										index === existingMsgIndex
											? {
												...msg,
												content: msg.content + token
											}
										: msg
									)
								} else {
									return [
										...prev,
										{
											id: assistantMessageId,
											role: "assistant",
											content: token,
											memoryUsed: parsed.memoryUsed,
											agentsUsed: parsed.agentsUsed,
											internetUsed: parsed.internetUsed
										}
									]
								}
							})
						}
					} catch (e) {
						console.error(
							"Error parsing stream data:",
							e,
							"Line:",
							line
						)
					}
				}
			}
		} catch (error) {
			if (error.name !== "AbortError")
				toast.error(`Error: ${error.message}`)
		} finally {
			setThinking(false)
		}
	}

	const handleStopStreaming = () => {
		if (abortControllerRef.current) {
			abortControllerRef.current.abort()
		}
	}

	useEffect(() => {
		// Auto-scroll to bottom, using smooth behavior
		if (chatEndRef.current)
			chatEndRef.current.scrollIntoView({ behavior: "smooth" })
	}, [messages, thinking]) // Rerun whenever messages change or thinking state changes
	const handleBackdropClick = (e) => {
		if (e.target === e.currentTarget) {
			onClose()
		}
	}

	return (
		<div
			onClick={handleBackdropClick}
			className="fixed inset-0 bg-black/50 backdrop-blur-md z-50 flex items-center justify-center p-4"
		>
			<motion.div
				initial={{ opacity: 0, y: 50, scale: 0.9 }}
				animate={{ opacity: 1, y: 0, scale: 1 }}
				exit={{ opacity: 0, y: 50, scale: 0.9 }}
				transition={{ duration: 0.3, ease: "easeInOut" }}
				className="bg-[var(--color-primary-background)] border border-[var(--color-primary-surface-elevated)] rounded-2xl w-full max-w-3xl h-[85vh] flex flex-col shadow-2xl"
			>
				<Tooltip id="chat-overlay-tooltip" />
				<header className="flex justify-between items-center p-4 border-b border-[var(--color-primary-surface)] flex-shrink-0">
					<h2 className="text-xl font-semibold text-[var(--color-text-primary)]">
						Chat with Sentient
					</h2>
					<button
						onClick={onClose}
						className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
					>
						<IconX />
					</button>
				</header>

				{/* NEW: Step progress log */}
				{stepEvents.length > 0 && (
					<div className="p-4 border-b border-[var(--color-primary-surface-elevated)] bg-[var(--color-primary-surface)]/50">
						<h4 className="text-sm font-semibold text-[var(--color-accent-blue)] mb-2">Backend Process Steps</h4>
						<ul className="space-y-1 text-xs">
							{stepEvents.map((step, idx) => (
								<li key={idx} className="flex items-center gap-2">
									<span className={`w-2 h-2 rounded-full ${step.status === 'done' ? 'bg-green-400' : step.status === 'in_progress' ? 'bg-yellow-400 animate-pulse' : 'bg-red-400'}`}></span>
									<span className="font-medium text-[var(--color-text-primary)]">{step.step}</span>
									<span className="text-[var(--color-text-secondary)]">[{new Date(step.timestamp).toLocaleTimeString()}]</span>
								</li>
							))}
						</ul>
					</div>
				)}

				<div
					ref={scrollContainerRef}
					className="flex-1 overflow-y-auto p-4 flex flex-col gap-4 custom-scrollbar"
				>
					{messages.length === 0 && !thinking ? (
						<div className="flex-1 flex flex-col justify-center items-center text-gray-400">
							<p className="text-3xl text-[var(--color-text-primary)] mb-4 font-Inter text-center">
								How can I help you today?
							</p>
						</div>
					) : (
						messages.map((msg, i) => (
							<div
								key={msg.id || i}
								className={`flex w-full ${
									msg.role === "user"
										? "justify-end"
										: "justify-start"
								}`}
							>
								<ChatBubble
									message={msg.content}
									isUser={msg.role === "user"}
									memoryUsed={msg.memoryUsed}
									agentsUsed={msg.agentsUsed}
									internetUsed={msg.internetUsed}
								/>
							</div>
						))
					)}
					<AnimatePresence>
						{thinking && (
							<div className="flex justify-start w-full mt-2">
								<motion.div
									initial={{ opacity: 0, y: 10 }}
									animate={{ opacity: 1, y: 0 }}
									exit={{ opacity: 0 }}
									className="flex items-center gap-2 p-3 bg-gray-700 rounded-lg"
								>
									<div className="bg-gray-400 rounded-full h-2 w-2 animate-pulse delay-75"></div>
									<div className="bg-gray-400 rounded-full h-2 w-2 animate-pulse delay-150"></div>
									<div className="bg-gray-400 rounded-full h-2 w-2 animate-pulse delay-300"></div>
								</motion.div>
							</div>
						)}
					</AnimatePresence>
					<div ref={chatEndRef} />{" "}
					{/* This empty div is the target for scrolling */}
				</div>

				<div className="p-4 border-t border-[var(--color-primary-surface)]">
					<div className="flex items-center flex-wrap justify-center gap-4 mb-3 text-xs text-[var(--color-text-secondary)]">
						{/* Tool Toggles Here */}
						<label
							className="flex items-center gap-1.5 cursor-pointer hover:text-white transition-colors"
							data-tooltip-id="chat-overlay-tooltip"
							data-tooltip-content="Allows the AI to search the web for real-time information."
						>
							<IconWorldSearch
								size={16}
								className={
									isInternetEnabled
										? "text-[var(--color-accent-blue)]"
										: ""
								}
							/>
							<span>Internet</span>
							<Switch
								checked={isInternetEnabled}
								onCheckedChange={setInternetEnabled}
							/>
						</label>
						<label
							className="flex items-center gap-1.5 cursor-pointer hover:text-white transition-colors"
							data-tooltip-id="chat-overlay-tooltip"
							data-tooltip-content="Enables weather forecasts for any location."
						>
							<IconCloud
								size={16}
								className={
									isWeatherEnabled
										? "text-[var(--color-accent-blue)]"
										: ""
								}
							/>
							<span>Weather</span>
							<Switch
								checked={isWeatherEnabled}
								onCheckedChange={setWeatherEnabled}
							/>
						</label>
						<label
							className="flex items-center gap-1.5 cursor-pointer hover:text-white transition-colors"
							data-tooltip-id="chat-overlay-tooltip"
							data-tooltip-content="Enables fetching the latest news headlines."
						>
							<IconNews
								size={16}
								className={
									isNewsEnabled
										? "text-[var(--color-accent-blue)]"
										: ""
								}
							/>
							<span>News</span>
							<Switch
								checked={isNewsEnabled}
								onCheckedChange={setNewsEnabled}
							/>
						</label>
						<label
							className="flex items-center gap-1.5 cursor-pointer hover:text-white transition-colors"
							data-tooltip-id="chat-overlay-tooltip"
							data-tooltip-content="Enables location-based information and directions."
						>
							<IconMap
								size={16}
								className={
									isMapsEnabled
										? "text-[var(--color-accent-blue)]"
										: ""
								}
							/>
							<span>Maps</span>
							<Switch
								checked={isMapsEnabled}
								onCheckedChange={setMapsEnabled}
							/>
						</label>
						<label
							className="flex items-center gap-1.5 cursor-pointer hover:text-white transition-colors"
							data-tooltip-id="chat-overlay-tooltip"
							data-tooltip-content="Enables product search and price comparisons."
						>
							<IconShoppingCart
								size={16}
								className={
									isShoppingEnabled
										? "text-[var(--color-accent-blue)]"
										: ""
								}
							/>
							<span>Shopping</span>
							<Switch
								checked={isShoppingEnabled}
								onCheckedChange={setShoppingEnabled}
							/>
						</label>
					</div>
					<div className="relative w-full flex flex-row gap-4 items-end px-4 py-2 bg-[var(--color-primary-background)] border-[1px] border-[var(--color-primary-surface-elevated)] rounded-lg">
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
							className="flex-grow p-2 pr-28 rounded-lg bg-transparent text-base text-[var(--color-text-primary)] focus:outline-none focus:ring-0 resize-none no-scrollbar overflow-y-auto"
							placeholder="Type your message..."
							style={{ maxHeight: "150px", minHeight: "24px" }}
							rows={1}
						/>
						<div className="absolute right-4 bottom-3 flex flex-row items-center gap-2">
							{thinking ? (
								<button
									onClick={handleStopStreaming}
									className="p-2 hover-button scale-100 hover:scale-110 cursor-pointer rounded-full text-white bg-[var(--color-accent-red)] hover:bg-[var(--color-accent-red-hover)]"
									data-tooltip-id="chat-overlay-tooltip"
									data-tooltip-content="Stop Generation"
								>
									<IconPlayerStopFilled className="w-4 h-4 text-white" />
								</button>
							) : (
								<button
									onClick={sendMessage}
									disabled={input.trim() === ""}
									className="p-2 hover-button scale-100 hover:scale-110 cursor-pointer rounded-full text-white disabled:opacity-50 disabled:cursor-not-allowed bg-[var(--color-accent-blue)]"
									data-tooltip-id="chat-overlay-tooltip"
									data-tooltip-content="Send Message"
								>
									<IconSend className="w-4 h-4 text-white" />
								</button>
							)}
						</div>
					</div>
				</div>
			</motion.div>
		</div>
	)
}

export default ChatOverlay
