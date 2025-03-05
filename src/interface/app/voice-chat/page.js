"use client"

import React, { useState, useEffect, useRef } from "react"
import { useSearchParams } from "next/navigation"
import ChatBubble from "@components/ChatBubble"
import Sidebar from "@components/Sidebar"
import { IconMicrophone, IconStop } from "@tabler/icons-react"
import toast from "react-hot-toast"

const VoiceChat = () => {
	const searchParams = useSearchParams()
	const chatId = searchParams.get("chatId")
	const [messages, setMessages] = useState([])
	const [recording, setRecording] = useState(false)
	const [userDetails, setUserDetails] = useState("")
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	const mediaRecorderRef = useRef(null)
	const chatEndRef = useRef(null)

	useEffect(() => {
		if (chatId) {
			fetchChatHistory()
		}
	}, [chatId])

	useEffect(() => {
		fetchUserDetails()
		window.electron.on("navigate-to-voice-chat", (newChatId) => {
			window.location.href = `/voice-chat?chatId=${newChatId}`
		})
	}, [])

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

	const startRecording = async () => {
		try {
			const stream = await navigator.mediaDevices.getUserMedia({
				audio: true
			})
			mediaRecorderRef.current = new MediaRecorder(stream)
			const chunks = []
			mediaRecorderRef.current.ondataavailable = (e) =>
				chunks.push(e.data)
			mediaRecorderRef.current.onstop = () => sendAudio(chunks)
			mediaRecorderRef.current.start()
			setRecording(true)
		} catch (error) {
			toast.error("Error starting recording.")
		}
	}

	const stopRecording = () => {
		if (mediaRecorderRef.current) {
			mediaRecorderRef.current.stop()
			mediaRecorderRef.current.stream
				.getTracks()
				.forEach((track) => track.stop())
			setRecording(false)
		}
	}

	const sendAudio = async (chunks) => {
		const audioBlob = new Blob(chunks, { type: "audio/wav" })
		const formData = new FormData()
		formData.append("audio", audioBlob, "audio.wav")
		formData.append("chat_id", chatId)

		try {
			const response = await fetch("http://localhost:5008/voice-chat", {
				method: "POST",
				body: formData
			})
			if (response.ok) {
				const audioResponseBlob = await response.blob()
				const audioUrl = URL.createObjectURL(audioResponseBlob)
				const audio = new Audio(audioUrl)
				audio.play()
				await fetchChatHistory() // Update chat history with new messages
			} else {
				toast.error("Error processing audio.")
			}
		} catch (error) {
			toast.error("Error sending audio.")
		}
	}

	if (!chatId) {
		return (
			<div className="h-screen flex justify-center items-center bg-matteblack">
				<p className="text-white text-lg">Loading...</p>
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
				<div className="w-4/5 ml-10 z-10 h-full overflow-y-scroll no-scrollbar flex flex-col gap-4">
					<div className="grow overflow-y-auto p-4 bg-matteblack rounded-xl no-scrollbar">
						{messages.length === 0 && (
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
								/>
							</div>
						))}
						<div ref={chatEndRef} />
					</div>
					<div className="flex justify-center w-full mb-5">
						<button
							onClick={recording ? stopRecording : startRecording}
							className="p-3 hover-button rounded-full text-white cursor-pointer"
						>
							{recording ? (
								<IconStop className="w-6 h-6 text-red-500" />
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
