"use client"

import React, { useState, useEffect, useRef } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import ChatBubble from "@components/ChatBubble"
import Sidebar from "@components/Sidebar"
import { IconMicrophone, IconMicrophoneOff } from "@tabler/icons-react"
import toast from "react-hot-toast"
import { AudioVisualizer } from "@components/AudioResponsiveBg"

const VoiceChat = () => {
	const searchParams = useSearchParams()
	const router = useRouter()
	const chatId = searchParams.get("chatId")
	const [messages, setMessages] = useState([])
	const [recording, setRecording] = useState(false)
	const [isPlaying, setIsPlaying] = useState(false)
	const [userDetails, setUserDetails] = useState("")
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	const [processingAudio, setProcessingAudio] = useState(false)
	const [isProcessing, setIsProcessing] = useState(false)
	const mediaRecorderRef = useRef(null)
	const currentAudioRef = useRef(null)
	const chatEndRef = useRef(null)
	const audioCtx = useRef(null)
	const tempMessageIdRef = useRef(null)
	const [userAnalyser, setUserAnalyser] = useState(null)
	const [aiAnalyser, setAiAnalyser] = useState(null)

	// Initialize AudioContext
	useEffect(() => {
		const AudioContext = window.AudioContext || window.webkitAudioContext
		if (!AudioContext) {
			toast.error("AudioContext is not supported in this browser")
			return
		}

		audioCtx.current = new AudioContext()

		// Automatically resume on any user interaction
		const resumeAudio = async () => {
			try {
				if (audioCtx.current.state === "suspended") {
					await audioCtx.current.resume()
				}
			} catch (error) {
				console.error("Error resuming audio context:", error)
			}
		}

		document.addEventListener("click", resumeAudio)
		return () => document.removeEventListener("click", resumeAudio)
	}, [])

	// Fetch chat history when chatId changes
	useEffect(() => {
		if (chatId) {
			fetchChatHistory()
		}
	}, [chatId])

	// Set up navigation listener with cleanup
	useEffect(() => {
		fetchUserDetails() // Your existing function to fetch user details
		const handleNavigate = (event, newChatId) => {
			router.push(`/voice-chat?chatId=${newChatId}`)
		}

		// Set up the event listener
		window.electron.on("navigate-to-voice-chat", handleNavigate)

		// Cleanup function to remove the listener
		return () => {
			window.electron.removeListener(
				"navigate-to-voice-chat",
				handleNavigate
			)
		}
	}, [router])

	// Auto-start recording if startRecording=true in URL
	useEffect(() => {
		const shouldStart = searchParams.get("startRecording") === "true"
		if (shouldStart && chatId && !recording && !processingAudio) {
			startRecording()
		}
	}, [searchParams, chatId])

	useEffect(() => {
		const handleVoiceMessagesAdded = (
			event,
			{ chatId: receivedChatId, messages: newMessages }
		) => {
			if (receivedChatId === chatId) {
				setMessages((prev) => [...prev, ...newMessages])
			}
		}
		window.electron.on("voice-messages-added", handleVoiceMessagesAdded)
		return () =>
			window.electron.removeListener(
				"voice-messages-added",
				handleVoiceMessagesAdded
			)
	}, [chatId])

	// Scroll to bottom of chat
	useEffect(() => {
		chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
	}, [messages])

	const fetchChatHistory = async () => {
		try {
			const response = await window.electron.invoke(
				"fetch-chat-history",
				{ chatId }
			)
			if (response.status === 200) {
				setMessages(response.messages)
			}
		} catch (error) {
			toast.error("Error fetching chat history.")
		}
	}

	const fetchUserDetails = async () => {
		try {
			const response = await window.electron.invoke("get-profile")
			setUserDetails(response)
		} catch (error) {
			toast.error("Error fetching user details.")
		}
	}

	const startNewVoiceChat = async (autoStartRecording = false) => {
		try {
			const response = await window.electron.invoke("create-chat", {
				title: "Voice Chat"
			})
			if (response.status === 200) {
				const newChatId = response.chatId
				const url = `/voice-chat?chatId=${newChatId}`
				router.push(url)
			} else {
				toast.error("Failed to create new voice chat.")
			}
		} catch (error) {
			toast.error("Error creating new voice chat.")
		}
	}

	let ws

	const startRecording = async () => {
		if (!chatId) {
			toast.error("Chat ID is missing. Please start a new voice chat.")
			return
		}

		try {
			setProcessingAudio(true)
			const stream = await navigator.mediaDevices.getUserMedia({
				audio: true
			})
			await audioCtx.current.resume()

			// Set up user analyser
			const source = audioCtx.current.createMediaStreamSource(stream)
			const analyser = audioCtx.current.createAnalyser()
			analyser.fftSize = 256
			source.connect(analyser)
			setUserAnalyser(analyser)

			const mediaRecorder = new MediaRecorder(stream)
			mediaRecorderRef.current = mediaRecorder

			// Establish WebSocket connection
			ws = new WebSocket("ws://localhost:5008/voice-chat-stream")

			ws.onopen = () => {
				console.log("WebSocket connection opened")
				ws.send(chatId) // Send chat_id first
			}

			ws.onmessage = async (event) => {
				if (typeof event.data === "string") {
					const data = JSON.parse(event.data)
					const { messages } = data
					const formattedMessages = messages.map((msg) => ({
						message: msg.message,
						isUser: msg.isUser,
						memoryUsed: msg.memoryUsed || false,
						agentsUsed: msg.agentsUsed || false,
						internetUsed: msg.internetUsed || false
					}))
					// Save to DB via IPC
					await window.electron.invoke("add-voice-messages", {
						chatId,
						messages: formattedMessages
					})
					setIsProcessing(false)
				} else if (event.data instanceof Blob) {
					console.log("Received audio response")
					const audioUrl = URL.createObjectURL(event.data)
					const audioElement = new Audio(audioUrl)
					currentAudioRef.current = audioElement
					const source =
						audioCtx.current.createMediaElementSource(audioElement)
					const analyser = audioCtx.current.createAnalyser()
					analyser.fftSize = 256
					source.connect(analyser)
					analyser.connect(audioCtx.current.destination)
					setAiAnalyser(analyser)
					setIsPlaying(true)
					audioElement.onended = () => {
						setIsPlaying(false)
						setAiAnalyser(null)
						setProcessingAudio(false)
					}
					audioElement.play().catch((error) => {
						console.error("Failed to play audio:", error)
						setIsPlaying(false)
						setProcessingAudio(false)
					})
				}
			}

			ws.onerror = (error) => {
				console.error("WebSocket error:", error)
				toast.error("WebSocket error occurred.")
				setProcessingAudio(false)
			}

			ws.onclose = () => {
				console.log("WebSocket connection closed")
			}

			mediaRecorder.ondataavailable = (event) => {
				if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
					console.log("Sending audio chunk via WebSocket")
					ws.send(event.data) // Send chunk as binary data
				}
			}

			mediaRecorder.onstop = () => {
				setIsProcessing(true) // Show processing message
				if (ws && ws.readyState === WebSocket.OPEN) {
					ws.send("END")
				}
			}

			mediaRecorder.start()
			setRecording(true)
			setProcessingAudio(false)
		} catch (error) {
			console.error("Error accessing microphone:", error)
			toast.error("Failed to start recording.")
			setProcessingAudio(false)
		}
	}

	const stopRecording = () => {
		if (mediaRecorderRef.current) {
			mediaRecorderRef.current.stop()
			mediaRecorderRef.current.stream
				.getTracks()
				.forEach((track) => track.stop())
			setRecording(false)
			setProcessingAudio(true)
			setUserAnalyser(null)
		}
	}

	if (!chatId) {
		return (
			<div className="h-screen w-full flex relative bg-matteblack">
				<Sidebar
					userDetails={userDetails}
					isSidebarVisible={isSidebarVisible}
					setSidebarVisible={setSidebarVisible}
					chatId={chatId}
					setChatId={(newChatId) =>
						(window.location.href = `/chat?chatId=${newChatId}`)
					}
					fromChat={true}
				/>
				<div className="w-4/5 flex flex-col justify-center items-center h-full bg-matteblack">
					<div className="flex flex-col items-center justify-center text-gray-500">
						<h1 className="text-6xl text-white font-bold mb-8 font-Poppins">
							Voice Chat
						</h1>
						<p className="text-2xl text-gray-500 mb-6 font-Poppins">
							Begin a new voice conversation
						</p>
						<button
							onClick={() => startNewVoiceChat(true)}
							disabled={processingAudio}
							className={`p-4 bg-lightblue text-white rounded-full hover:bg-blue-600 text-lg font-Poppins ${
								processingAudio
									? "opacity-50 cursor-not-allowed"
									: ""
							}`}
						>
							{processingAudio ? (
								"Starting..."
							) : (
								<>
									<IconMicrophone className="w-6 h-6 inline mr-2" />
									Start Talking
								</>
							)}
						</button>
					</div>
				</div>
			</div>
		)
	}

	return (
		<div className="h-screen bg-matteblack flex relative">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
				chatId={chatId}
				setChatId={(newChatId) =>
					(window.location.href = `/chat?chatId=${newChatId}`)
				}
				fromChat={true}
			/>
			<div className="w-4/5 flex flex-col justify-center items-start h-full bg-matteblack">
				<div className="flex justify-between w-full mb-4 px-10">
					<h2 className="text-2xl font-Poppins mt-20 text-white">
						Voice Chat
					</h2>
				</div>
				<div className="w-4/5 ml-10 z-10 h-full overflow-y-scroll no-scrollbar flex flex-col gap-4">
					<div className="grow overflow-y-auto p-4 bg-matteblack rounded-xl no-scrollbar">
						{messages.length === 0 && !isProcessing && (
							<div className="font-Poppins h-full flex flex-col justify-center items-center text-gray-500">
								<p className="text-4xl text-white mb-4">
									Start a voice conversation
								</p>
								<p className="text-2xl text-gray-500">
									Press the microphone to begin recording.
								</p>
							</div>
						)}
						{messages.map((msg) => (
							<div
								key={msg.id}
								className={`flex ${msg.isUser ? "justify-end" : "justify-start"}`}
							>
								<ChatBubble
									message={msg.message}
									isUser={msg.isUser}
									memoryUsed={msg.memoryUsed}
									agentsUsed={msg.agentsUsed}
									internetUsed={msg.internetUsed}
									isProcessing={msg.isProcessing}
								/>
							</div>
						))}
						<div ref={chatEndRef} />
					</div>
					{isProcessing && (
						<div className="flex justify-end p-4">
							<p className="text-gray-500 italic">
								Processing your audio...
							</p>
						</div>
					)}
					<div className="flex flex-col items-center mb-4">
						{isPlaying && aiAnalyser && (
							<div className="mb-4 w-full max-w-md">
								<p className="text-white text-center mb-2">
									AI Speaking
								</p>
								<AudioVisualizer
									analyser={aiAnalyser}
									isAI={true}
								/>
							</div>
						)}
						{recording && userAnalyser && (
							<div className="w-full max-w-md">
								<p className="text-white text-center mb-2">
									You are Speaking
								</p>
								<AudioVisualizer
									analyser={userAnalyser}
									isAI={false}
								/>
							</div>
						)}
					</div>
					<div className="flex justify-center w-full mb-5">
						<button
							onClick={() => {
								console.log(
									"Mic button clicked, recording:",
									recording
								)
								recording ? stopRecording() : startRecording()
							}}
							disabled={processingAudio && !recording}
							className={`p-3 hover-button rounded-full text-white cursor-pointer ${
								processingAudio && !recording
									? "opacity-50 cursor-not-allowed"
									: ""
							}`}
						>
							{recording ? (
								<IconMicrophoneOff className="w-6 h-6 text-red-500" />
							) : (
								<IconMicrophone className="w-6 h-6 text-white" />
							)}
						</button>
					</div>
				</div>
			</div>
		</div>
	)
}

export default VoiceChat
