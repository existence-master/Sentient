"use client"
import React, { useState, useEffect, useRef } from "react"
import ChatBubble from "@components/ChatBubble"
import ToolResultBubble from "@components/ToolResultBubble"
import Sidebar from "@components/Sidebar"
import {
	IconSend,
	IconRefresh,
	IconPlayerPlayFilled,
	IconLoader
} from "@tabler/icons-react"
import toast from "react-hot-toast"


const Chat = () => {
	const [messages, setMessages] = useState([])
	const [input, setInput] = useState("")
	const [userDetails, setUserDetails] = useState("")
	const [thinking, setThinking] = useState(false)
	const [serverStatus, setServerStatus] = useState(true)
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	const [currentModel, setCurrentModel] = useState("")
	const [isLoading, setIsLoading] = useState(true) // New loading state

	const textareaRef = useRef(null)
	const chatEndRef = useRef(null)
	const eventListenersAdded = useRef(false)

	const handleInputChange = (e) => {
		const value = e.target.value
		setInput(value)
		textareaRef.current.style.height = "auto"
		textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
	}

	const fetchChatHistory = async () => {
		setIsLoading(true)
		try {
			const response = await window.electron?.invoke("fetch-chat-history")
			if (response.status === 200) setMessages(response.messages)
			else toast.error("Error fetching chat history.")
		} catch (error) {
			toast.error("Error fetching chat history.")
		} finally {
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
		setCurrentModel("llama3.2:3b") // Placeholder, adjust as needed
	}

	useEffect(() => {
		fetchUserDetails()
		fetchCurrentModel()
		fetchChatHistory()
	}, [])

	useEffect(() => {
		chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
	}, [messages])

	const setupIpcListeners = () => {
		if (!eventListenersAdded.current) {
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
								internetUsed: false
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
			return () => {
				eventListenersAdded.current = false
			}
		}
	}

	const sendMessage = async () => {
		if (input.trim() === "") return
		const newMessage = { message: input, isUser: true, id: Date.now() }
		setMessages((prev) => [...prev, newMessage])
		setInput("")
		setThinking(true)

		setupIpcListeners()

		try {
			const response = await window.electron?.invoke("send-message", {
				input: newMessage.message
			})
			if (response.status === 200) {
				await fetchChatHistory()
				setThinking(false)
			} else {
				toast.error("Failed to send message.")
			}
		} catch (error) {
			toast.error("Error sending message.")
		} finally {
			setThinking(false)
		}
	}

	const clearChatHistory = async () => {
		try {
			const response = await fetch(
				`http://localhost:5000/clear-chat-history`,
				{ method: "POST" }
			)
			if (!response.ok) throw new Error("Failed to clear chat history")
			setMessages([])
			toast.success("Chat history cleared.")
		} catch (error) {
			toast.error("Error clearing chat history.")
		}
	}

	const reinitiateServer = async () => {
		setServerStatus(false)
		try {
			const response = await window.electron?.invoke("fetch-chat-history")
			if (response.status === 200) setMessages(response.messages)
		} catch (error) {
			toast.error("Error restarting the server.")
		} finally {
			setServerStatus(true)
		}
	}

	useEffect(() => {
		const intervalId = setInterval(fetchChatHistory, 60000) // Refresh every 5 seconds

		return () => clearInterval(intervalId) // Cleanup interval on component unmount
	}, [])

	return (
		<div className="h-screen bg-matteblack flex relative">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
			/>
			<div className="w-4/5 flex flex-col justify-center items-start h-full bg-matteblack">
				<div className="flex w-4/5 items-start z-20 justify-between mb-6">
					<button
						onClick={reinitiateServer}
						className="absolute top-10 right-10 p-3 hover-button rounded-full text-white cursor-pointer"
					>
						{serverStatus ? (
							<IconPlayerPlayFilled className="w-4 h-4 text-white" />
						) : (
							<IconLoader className="w-4 h-4 text-white animate-spin" />
						)}
					</button>
				</div>
				<div className="w-4/5 ml-10 z-10 h-full overflow-y-scroll no-scrollbar flex flex-col gap-4">
					<div className="grow overflow-y-auto p-4 bg-matteblack rounded-xl no-scrollbar">
						{isLoading ? (
							<div className="flex justify-center items-center h-full">
								<IconLoader className="w-8 h-8 text-white animate-spin" />
							</div>
						) : messages.length === 0 ? (
							<div className="font-Poppins h-full flex flex-col justify-center items-center text-gray-500">
								<p className="text-4xl text-white mb-4">
									Send a message to start a conversation
								</p>
							</div>
						) : (
							messages.map((msg) => (
								<div
									key={msg.id}
									className={`flex ${msg.isUser ? "justify-end" : "justify-start"}`}
								>
									{msg.type === "tool_result" ? (
										<ToolResultBubble
											task={msg.task}
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
										/>
									)}
								</div>
							))
						)}
						{thinking && (
							<div className="flex items-center gap-2 mt-3 animate-pulse">
								<div className="bg-gray-500 rounded-full h-3 w-3"></div>
								<div className="bg-gray-500 rounded-full h-3 w-3"></div>
								<div className="bg-gray-500 rounded-full h-3 w-3"></div>
							</div>
						)}
						<div ref={chatEndRef} />
					</div>
					<p className="text-gray-400 font-Poppins text-sm">
						Current Model:{" "}
						<span className="text-lightblue">{currentModel}</span>
					</p>
					<div className="relative mb-5 flex flex-row gap-4 w-full px-4 py-1 bg-matteblack border-[1px] border-white rounded-lg z-30">
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
							className="grow p-5 pr-24 rounded-lg bg-transparent text-lg text-white focus:outline-hidden resize-none no-scrollbar overflow-y-auto"
							placeholder="Start typing..."
							style={{ maxHeight: "200px", minHeight: "30px" }}
							rows={1}
						/>
						<div className="absolute right-4 bottom-4 flex flex-row items-center gap-3">
							<button
								onClick={sendMessage}
								className="p-3 hover-button scale-100 hover:scale-110 cursor-pointer rounded-full text-white"
							>
								<IconSend className="w-4 h-4 text-white" />
							</button>
							<button
								onClick={clearChatHistory}
								className="p-3 rounded-full hover-button scale-100 cursor-pointer hover:scale-110 text-white"
							>
								<IconRefresh className="w-4 h-4 text-white" />
							</button>
						</div>
					</div>
				</div>
			</div>
		</div>
	)
}

export default Chat