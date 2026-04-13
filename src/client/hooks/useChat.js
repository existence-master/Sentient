"use client"

import { useState, useEffect, useRef, useCallback, useMemo } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import toast from "react-hot-toast"
import { usePostHog } from "posthog-js/react"
import { WebRTCClient } from "@lib/webrtc-client"
import { useTourStore } from "@stores/app-stores"

function usePrevious(value) {
	const ref = useRef()
	useEffect(() => {
		ref.current = value
	})
	return ref.current
}

export const useChat = () => {
	const [displayedMessages, setDisplayedMessages] = useState([])
	const [input, setInput] = useState("")
	const [isLoading, setIsLoading] = useState(true)
	const [thinking, setThinking] = useState(false)
	const textareaRef = useRef(null)
	const chatEndRef = useRef(null)
	const abortControllerRef = useRef(null)
	const scrollContainerRef = useRef(null)
	const fileInputRef = useRef(null)

	// State for infinite scroll
	const [isLoadingOlder, setIsLoadingOlder] = useState(false)
	const [hasMoreMessages, setHasMoreMessages] = useState(true)
	const [searchingForMessageId, setSearchingForMessageId] = useState(null)

	// State for UI enhancements
	const [userDetails, setUserDetails] = useState(null)
	const posthog = usePostHog()
	const [isFocused, setIsFocused] = useState(false)
	const [isWelcomeModalOpen, setIsWelcomeModalOpen] = useState(false)
	const [replyingTo, setReplyingTo] = useState(null)
	const [isOptionsOpen, setIsOptionsOpen] = useState(false)
	const [confirmClear, setConfirmClear] = useState(false)
	const [integrations, setIntegrations] = useState([])

	const searchParams = useSearchParams()
	const router = useRouter()
	const tourState = useTourStore()
	const startTour = useTourStore((s) => s.startTour)
	const nextStep = useTourStore((s) => s.nextStep)
	const nextSubStep = useTourStore((s) => s.nextSubStep)
	const setHighlightPaused = useTourStore((s) => s.setHighlightPaused)
	const chatActionsRef = useTourStore((s) => s.chatActionsRef)
	const prevTourState = usePrevious(tourState)

	// --- File Upload State ---
	const [selectedFile, setSelectedFile] = useState(null)
	const [isUploading, setIsUploading] = useState(false)
	const [uploadedFilename, setUploadedFilename] = useState(null)

	// --- Voice Mode State ---
	const [isMuted, setIsMuted] = useState(false)
	const [isVoiceMode, setIsVoiceMode] = useState(false)
	const [connectionStatus, setConnectionStatus] = useState("disconnected")
	const [audioInputDevices, setAudioInputDevices] = useState([])
	const [selectedAudioInputDevice, setSelectedAudioInputDevice] = useState("")
	const [voiceStatusText, setVoiceStatusText] = useState(
		"Click to start call"
	)
	const [statusText, setStatusText] = useState("")
	const [audioLevel, setAudioLevel] = useState(0)
	const webrtcClientRef = useRef(null)
	const ringtoneAudioRef = useRef(null)
	const connectedAudioRef = useRef(null)
	const remoteAudioRef = useRef(null)
	const voiceModeStartTimeRef = useRef(null)

	const lastSpokenTextRef = useRef("")
	const setMicrophoneEnabled = useCallback((enabled) => {
		if (webrtcClientRef.current?.mediaStream) {
			const audioTracks =
				webrtcClientRef.current.mediaStream.getAudioTracks()
			if (audioTracks.length > 0) {
				// Only change if the state is different to avoid unnecessary operations
				if (audioTracks[0].enabled !== enabled) {
					audioTracks[0].enabled = enabled
					setIsMuted(!enabled)
				}
			}
		}
	}, [])

	const fetchInitialMessages = useCallback(async () => {
		if (tourState?.isActive) {
			setIsLoading(false)
			return
		}
		setIsLoading(true)
		try {
			const res = await fetch("/api/chat/history", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ limit: 50 })
			})
			if (!res.ok) throw new Error("Failed to fetch messages")
			const data = await res.json()
			const fetchedMessages = (data.messages || []).map((m) => ({
				...m,
				id: m.message_id
			}))
			setDisplayedMessages(fetchedMessages)
			setHasMoreMessages((data.messages || []).length === 50)
		} catch (error) {
			toast.error(error.message)
		} finally {
			setIsLoading(false)
		}
	}, [tourState?.isActive])

	const fetchUserDetails = useCallback(async () => {
		try {
			const res = await fetch("/api/user/profile")
			if (res.ok) {
				const data = await res.json()
				setUserDetails(data)
			} else {
				setUserDetails({ given_name: "User" })
			}
		} catch (error) {
			console.error("Failed to fetch user details:", error)
			setUserDetails({ given_name: "User" })
		}
	}, [])

	useEffect(() => {
		// When tour becomes inactive, refetch original messages
		if (!tourState?.isActive && prevTourState?.isActive) {
			fetchInitialMessages()
		}
	}, [tourState?.isActive, prevTourState?.isActive, fetchInitialMessages])

	useEffect(() => {
		fetchInitialMessages()
		fetchUserDetails()
		return () => {
			if (abortControllerRef.current) {
				abortControllerRef.current.abort()
			}
		}
	}, [fetchInitialMessages, fetchUserDetails])

	useEffect(() => {
		if (
			searchParams.get("show_demo") === "true" &&
			startTour &&
			!tourState?.isActive
		) {
			startTour()
			router.replace("/chat", { scroll: false }) // Keep this to clean URL
		}
	}, [searchParams, router, startTour, tourState?.isActive])

	useEffect(() => {
		const messageId = searchParams.get("messageId")
		if (!messageId || displayedMessages.length === 0) return

		const element = document.getElementById(`message-${messageId}`)
		if (element) {
			// Message is already rendered, scroll to it.
			element.scrollIntoView({ behavior: "smooth", block: "center" })
			element.classList.add("highlight-message")
			setTimeout(() => {
				element.classList.remove("highlight-message")
			}, 3000)

			// Clean up state and URL
			setSearchingForMessageId(null)
			router.replace("/chat", { scroll: false })
		} else if (!searchingForMessageId) {
			// Message is not rendered, start the search process.
			setSearchingForMessageId(messageId)
		}
	}, [searchParams, displayedMessages, router, searchingForMessageId])

	useEffect(() => {
		// This effect triggers the fetching loop when a search is initiated.
		if (searchingForMessageId) {
			fetchOlderMessages()
		}
	}, [searchingForMessageId]) // Dependency on searchingForMessageId

	const sendMessage = async () => {
		if ((!input.trim() && !uploadedFilename) || thinking || isUploading)
			return

		setThinking(true)
		abortControllerRef.current = new AbortController()

		posthog?.capture("chat_message_sent", {
			message_length: input.length,
			has_file: !!uploadedFilename
		})

		let messageContent = input.trim()
		if (uploadedFilename) {
			messageContent = `(Attached file for context: ${uploadedFilename}) ${messageContent}. Use file-management MCP to read it`
		}

		const newUserMessage = {
			id: `user-${Date.now()}`,
			role: "user",
			content: messageContent,
			timestamp: new Date().toISOString(),
			...(replyingTo && { replyToId: replyingTo.id })
		}

		setStatusText("Getting ready...")
		const updatedMessages = [...displayedMessages, newUserMessage]
		setDisplayedMessages(updatedMessages)

		setInput("")
		setReplyingTo(null)
		setUploadedFilename(null) // Reset file after sending
		setSelectedFile(null)
		if (textareaRef.current) textareaRef.current.style.height = "auto"

		try {
			if (tourState?.isActive && tourState.step === 1) {
				// --- TOUR SIMULATION ---
				const subStep = tourState.subStep
				setHighlightPaused(true) // Pause highlight as soon as message is sent
				setThinking(true)

				if (subStep === 0) {
					// First message: "Hi Sentient!"
					setTimeout(() => {
						const fakeResponse = {
							id: `assistant-${Date.now()}`,
							role: "assistant",
							content: "Hey there, I'm ready to help.",
							timestamp: new Date().toISOString()
						}
						setDisplayedMessages((prev) => [...prev, fakeResponse])
						setThinking(false)
						setTimeout(() => {
							setHighlightPaused(false) // Resume highlight for next instruction
							nextSubStep()
						}, 2000) // 2 second delay to read the message
					}, 1500) // Delay for assistant to "think"
				} else if (subStep === 1) {
					// Second message: "Send an email..."
					setStatusText("Analyzing request...")
					setTimeout(() => {
						setStatusText("Using tool: gmail")
					}, 1000)

					setTimeout(() => {
						const fakeResponse = {
							id: `assistant-${Date.now()}`,
							role: "assistant",
							content:
								"Cool, I've sent that email. Is there anything else you want to do?",
							timestamp: new Date().toISOString(),
							tools: ["gmail"]
						}
						setDisplayedMessages((prev) => [...prev, fakeResponse])
						setThinking(false)
						setStatusText("")
						setTimeout(() => {
							setHighlightPaused(false) // Resume highlight for next instruction
							nextSubStep()
						}, 2000)
					}, 2500) // Delay for assistant to "work"
				} else if (subStep === 2) {
					// Third message: "Create workflow..."
					setStatusText("Analyzing request...")
					setTimeout(() => {
						setStatusText("Using tool: tasks")
					}, 1000)
					setTimeout(() => {
						const fakeResponse = {
							id: `assistant-${Date.now()}`,
							role: "assistant",
							content: "Cool, I've created the workflow.",
							timestamp: new Date().toISOString(),
							tools: ["tasks"]
						}
						setDisplayedMessages((prev) => [...prev, fakeResponse])
						setThinking(false)
						setStatusText("")
						// This is the last chat step, move to the next main step.
						setTimeout(() => {
							// No need to resume highlight, as the next step will have a new target.
							nextStep()
						}, 2000)
					}, 2500)
				}
				return // End simulation here
			}

			const response = await fetch("/api/chat/message", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					// The server only needs the new user message to save it, then it fetches its own history.
					messages: [newUserMessage]
				}),
				signal: abortControllerRef.current.signal
			})

			if (!response.ok) {
				const errorData = await response.json().catch(() => ({
					detail: `Request failed with status ${response.status}`
				}))
				const error = new Error(
					errorData.detail || "An unexpected error occurred."
				)
				error.status = response.status
				throw error
			}

			const reader = response.body.getReader()
			const decoder = new TextDecoder()
			let assistantMessageId = `assistant-${Date.now()}`

			setDisplayedMessages((prev) => [
				...prev,
				{
					id: assistantMessageId,
					role: "assistant",
					content: "",
					timestamp: new Date().toISOString(),
					tools: [],
					turn_steps: [] // --- ADDED --- Initialize turn_steps for the new message
				}
			])

			while (true) {
				const { done, value } = await reader.read()
				if (done) break

				const chunk = decoder.decode(value)
				for (const line of chunk.split("\n")) {
					if (!line.trim()) continue

					try {
						const parsed = JSON.parse(line)

						if (parsed.type === "error") {
							toast.error(`An error occurred: ${parsed.message}`)
							continue
						}

						// This is the fix: Update the temporary ID to the real one from the backend
						if (
							parsed.messageId &&
							assistantMessageId.startsWith("assistant-")
						) {
							const tempId = assistantMessageId
							assistantMessageId = parsed.messageId // Update the reference to the real ID
							setDisplayedMessages((prev) =>
								prev.map((m) =>
									m.id === tempId
										? { ...m, id: parsed.messageId }
										: m
								)
							)
						}

						// Handle status updates from the backend
						if (parsed.type === "status") {
							setStatusText(parsed.message)
							continue
						}

						// Clear status text when the actual response starts streaming
						if (parsed.type === "assistantStream" && parsed.token) {
							setStatusText("")
						}

						setDisplayedMessages((prev) =>
							prev.map((msg) => {
								if (msg.id === assistantMessageId) {
									// Only update the message when the stream is done
									if (parsed.done) {
										return {
											...msg,
											content: parsed.final_content || "", // Replace content with clean final version
											turn_steps: parsed.turn_steps || [], // Populate turn_steps
											tools: parsed.tools || [] // Update tools on final event
										}
									}
									// For intermediate chunks, we don't update the content to prevent streaming.
									// We can still update tools if they arrive early.
									return {
										...msg,
										tools: parsed.tools || msg.tools
									}
								}
								return msg
							})
						)
					} catch (parseError) {
						setDisplayedMessages((prev) =>
							prev.map((msg) => {
								if (msg.id === assistantMessageId) {
									return {
										...msg,
										content: msg.content + line
									}
								}
								return msg
							})
						)
					}
				}
			}
		} catch (error) {
			if (error.name === "AbortError") {
				toast.info("Message generation stopped.")
			} else {
				toast.error(`Error: ${error.message}`)
			}
			console.error("Fetch error:", error)
			setDisplayedMessages((prev) =>
				prev.filter((m) => m.id !== newUserMessage.id)
			)
		} finally {
			setThinking(false)
			setStatusText("")
		}
	}

	// Attach chat functions to the tour context's ref
	useEffect(() => {
		if (chatActionsRef) {
			// Attach the functions the tour needs to the ref
			chatActionsRef.current = {
				setInput: setInput,
				sendMessage: sendMessage
			}
		}
		// Cleanup function to nullify the ref when the component unmounts
		return () => {
			if (chatActionsRef) {
				chatActionsRef.current = null
			}
		}
	}, [chatActionsRef, sendMessage, setInput])

	const fetchIntegrations = useCallback(async () => {
		try {
			const res = await fetch("/api/settings/integrations", {
				method: "POST"
			})
			if (!res.ok) throw new Error("Failed to fetch integrations")
			const data = await res.json()
			setIntegrations(data.integrations || [])
		} catch (error) {
			console.error(
				"Failed to fetch integrations for tools menu:",
				error.message
			)
		}
	}, [])

	useEffect(() => {
		fetchIntegrations()
	}, [fetchIntegrations])

	const fetchOlderMessages = useCallback(async () => {
		if (
			isLoadingOlder ||
			!hasMoreMessages ||
			displayedMessages.length === 0
		) {
			return
		}

		setIsLoadingOlder(true)
		const oldestMessageTimestamp = displayedMessages[0].timestamp

		try {
			const res = await fetch(`/api/chat/history`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					limit: 50,
					before_timestamp: oldestMessageTimestamp
				})
			})
			if (!res.ok) throw new Error("Failed to fetch older messages")
			const data = await res.json()

			if (data.messages && data.messages.length > 0) {
				const scrollContainer = scrollContainerRef.current
				const oldScrollHeight = scrollContainer.scrollHeight

				const olderMessages = data.messages.map((m) => ({
					...m,
					id: m.message_id
				}))

				const isTargetInBatch = olderMessages.some(
					(m) => m.id === searchingForMessageId
				)

				setDisplayedMessages((prev) => [...olderMessages, ...prev]) // This will trigger the other useEffect to scroll
				setHasMoreMessages(data.messages.length === 50)

				if (searchingForMessageId) {
					// If we are searching and didn't find the target, and there are more messages, fetch again.
					if (!isTargetInBatch && data.messages.length === 50) {
						// We call fetchOlderMessages again, but it will use the *new* oldest timestamp
						// from the state that was just updated. We wrap in a timeout to allow React to re-render.
						setTimeout(() => fetchOlderMessages(), 100)
					}
				} else {
					// Normal infinite scroll behavior
					setTimeout(() => {
						scrollContainer.scrollTop =
							scrollContainer.scrollHeight - oldScrollHeight
					}, 0)
				}
			} else {
				setHasMoreMessages(false)
				setSearchingForMessageId(null) // Stop searching if no more messages
			}
		} catch (error) {
			toast.error(error.message)
		} finally {
			setIsLoadingOlder(false)
		}
	}, [
		isLoadingOlder,
		hasMoreMessages,
		displayedMessages,
		searchingForMessageId
	])

	useEffect(() => {
		const container = scrollContainerRef.current
		const handleScroll = () => {
			if (container && container.scrollTop === 0) {
				fetchOlderMessages()
			}
		}
		container?.addEventListener("scroll", handleScroll)
		return () => container?.removeEventListener("scroll", handleScroll)
	}, [fetchOlderMessages])

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

	const handleReply = (message) => {
		setReplyingTo(message)
		textareaRef.current?.focus()
	}

	const handleFileChange = async (event) => {
		const file = event.target.files?.[0]
		if (!file) return

		// Reset file input to allow re-uploading the same file
		event.target.value = ""

		// --- ADDED: File Type Validation ---
		const supportedExtensions = [
			".csv",
			".doc",
			".docx",
			".eml",
			".epub",
			".gif",
			".jpg",
			".jpeg",
			".json",
			".html",
			".htm",
			".msg",
			".odt",
			".pdf",
			".png",
			".pptx",
			".ps",
			".rtf",
			".tiff",
			".tif",
			".txt",
			".xlsx",
			".xls"
		]
		const fileExtension = `.${file.name.split(".").pop()?.toLowerCase()}`

		if (!supportedExtensions.includes(fileExtension)) {
			toast.error(
				`Unsupported file type: ${fileExtension}. Please upload a supported file.`
			)
			return
		}
		// --- END ADDED SECTION ---
		if (file.size > 5 * 1024 * 1024) {
			// 5MB limit
			toast.error(
				"File is too large. Please select a file smaller than 5MB."
			)
			return
		}

		setSelectedFile(file)
		setIsUploading(true)
		setUploadedFilename(null)
		const toastId = toast.loading(`Uploading ${file.name}...`)

		try {
			const formData = new FormData()
			formData.append("file", file)

			const response = await fetch("/api/files/upload", {
				method: "POST",
				body: formData
			})

			if (!response.ok) {
				const errorData = await response.json().catch(() => ({}))
				const error = new Error(errorData.error || "File upload failed")
				error.status = response.status
				throw error
			}

			const result = await response.json()
			setUploadedFilename(result.filename)
			toast.success(`${result.filename} uploaded successfully.`, {
				id: toastId
			})
		} catch (error) {
			toast.error(`Error: ${error.message}`, { id: toastId })
			setSelectedFile(null)
		} finally {
			setIsUploading(false)
		}
	}

	const handleDeleteMessage = async (messageId) => {
		const originalMessages = [...displayedMessages]
		setDisplayedMessages((prev) => prev.filter((m) => m.id !== messageId))

		try {
			const res = await fetch("/api/chat/delete", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ message_id: messageId })
			})
			if (!res.ok) {
				const errorData = await res.json()
				throw new Error(errorData.error || "Failed to delete message")
			}
			toast.success("Message deleted.")
		} catch (error) {
			toast.error(error.message)
			setDisplayedMessages(originalMessages) // Revert on error
		}
	}

	const handleClearAllMessages = async () => {
		setDisplayedMessages([])
		setIsOptionsOpen(false)
		setConfirmClear(false)
		try {
			const res = await fetch("/api/chat/delete", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ clear_all: true })
			})
			if (!res.ok) throw new Error("Failed to clear chat history")
			toast.success("Chat history cleared.")
		} catch (error) {
			toast.error(error.message)
			fetchInitialMessages() // Refetch to restore state on error
		}
	}

	const handleStopStreaming = () => {
		if (abortControllerRef.current) {
			abortControllerRef.current.abort()
			toast.info("Message generation stopped.")
		}
	}

	useEffect(() => {
		if (chatEndRef.current && !isVoiceMode) {
			// Use 'auto' for an instant scroll, which feels better when switching modes.
			chatEndRef.current.scrollIntoView({ behavior: "auto" })
		}
	}, [displayedMessages, thinking, isVoiceMode])

	const getGreeting = () => {
		const hour = new Date().getHours()
		if (hour < 12) return "Good Morning"
		if (hour < 18) return "Good Afternoon"
		return "Good Evening"
	}

	// --- Voice Mode Handlers ---
	const handleStatusChange = useCallback(
		(status) => {
			console.log(
				`[ChatPage] Voice connection status changed to: ${status}`
			)
			setConnectionStatus(status)
			if (status !== "connecting" && ringtoneAudioRef.current) {
				ringtoneAudioRef.current.pause()
				ringtoneAudioRef.current.currentTime = 0
			}
			if (status === "connected") {
				if (connectedAudioRef.current) {
					connectedAudioRef.current.volume = 0.4
					connectedAudioRef.current
						.play()
						.catch((e) =>
							console.error(
								"[ChatPage] Error playing connected sound:",
								e
							)
						)
				}
				// Add a delay to allow ICE connection to stabilize
				console.log(
					"[ChatPage] Connection established. Muting mic for 4s to stabilize."
				)
				setVoiceStatusText("Please wait a moment...")
				setMicrophoneEnabled(false) // Mute mic during stabilization
				setTimeout(() => {
					console.log(
						"[ChatPage] Stabilization complete. Unmuting mic."
					)
					setVoiceStatusText("Listening...")
					setMicrophoneEnabled(true) // Unmute after delay
				}, 4000)
			} else if (status === "disconnected") {
				setVoiceStatusText("Click to start call")
			} else if (status === "connecting") {
				setVoiceStatusText("Connecting...")
			}
		},
		[setMicrophoneEnabled]
	)

	const handleVoiceEvent = useCallback(
		(event) => {
			if (event.type === "stt_result" && event.text) {
				setDisplayedMessages((prev) => [
					...prev,
					{
						id: `user_${Date.now()}`,
						role: "user",
						content: event.text,
						timestamp: new Date().toISOString()
					}
				])
			} else if (event.type === "llm_result" && event.text) {
				lastSpokenTextRef.current = event.text // Store the text for duration calculation
				setDisplayedMessages((prev) => [
					...prev,
					{
						id: event.messageId || `assistant_${Date.now()}`,
						role: "assistant",
						content: event.text,
						timestamp: new Date().toISOString()
					}
				])
			} else if (event.type === "status") {
				console.log(`[ChatPage] Voice status update: ${event.message}`)
				if (event.message === "thinking") {
					setVoiceStatusText("Thinking...")
					setMicrophoneEnabled(false)
				} else if (event.message === "speaking") {
					setVoiceStatusText("Speaking...")
					setMicrophoneEnabled(false)
				} else if (event.message === "listening") {
					// The server sends 'listening' when it's done sending audio,
					// but client-side buffering can cause a delay. We estimate
					// the speaking duration based on the text length from the
					// `llm_result` event to avoid unmuting the mic too early.
					const textToMeasure = lastSpokenTextRef.current
					// Estimate duration: ~18 chars/sec -> ~55ms/char. Add a smaller buffer.
					const estimatedDuration = textToMeasure.length * 55 + 250 // ms
					console.log(
						`[ChatPage] Server is listening. Waiting estimated ${estimatedDuration}ms for audio buffer to clear before unmuting.`
					)

					setTimeout(() => {
						if (
							webrtcClientRef.current?.peerConnection
								?.connectionState === "connected"
						) {
							console.log(
								"[ChatPage] Estimated audio buffer clear time elapsed. Unmuting."
							)
							setVoiceStatusText("Listening...")
							setMicrophoneEnabled(true)
						} else {
							console.log(
								"[ChatPage] Estimated audio buffer clear time elapsed, but connection is no longer active. Not unmuting."
							)
						}
					}, estimatedDuration)

					// Reset for the next turn
					lastSpokenTextRef.current = ""
				} else if (event.message === "transcribing") {
					setVoiceStatusText("Transcribing...")
					setMicrophoneEnabled(false) // Mute as soon as transcription starts
				} else if (event.message === "choosing_tools")
					setVoiceStatusText("Choosing tools...")
				else if (
					event.message &&
					event.message.startsWith("using_tool_")
				) {
					const toolName = event.message
						.replace("using_tool_", "")
						.replace("_server", "")
						.replace("_mcp", "")
					setVoiceStatusText(
						`Using ${
							toolName.charAt(0).toUpperCase() + toolName.slice(1)
						}...`
					)
				}
			} else if (event.type === "error") {
				toast.error(`Voice Error: ${event.message}`)
				setVoiceStatusText("Error. Click to retry.")
			}
		},
		[setMicrophoneEnabled]
	)

	const handleAudioLevel = useCallback((level) => {
		setAudioLevel((prev) => prev * 0.7 + level * 0.3)
	}, [])

	const handleStartVoice = async () => {
		if (connectionStatus !== "disconnected") return
		console.log("[ChatPage] handleStartVoice triggered.")
		// --- ADD POSTHOG EVENT TRACKING ---
		posthog?.capture("voice_mode_activated")
		voiceModeStartTimeRef.current = Date.now() // Set start time
		// --- END POSTHOG EVENT TRACKING ---

		setConnectionStatus("connecting")
		setVoiceStatusText("Connecting...")
		try {
			// Step 1: Get the main auth token
			console.log("[ChatPage] Fetching auth token...")
			const tokenResponse = await fetch("/api/auth/token")
			if (!tokenResponse.ok) throw new Error("Could not get auth token.")
			const { accessToken } = await tokenResponse.json()
			console.log("[ChatPage] Auth token fetched.")

			// Step 2: Use the auth token to get a temporary RTC token
			const serverUrl =
				process.env.NEXT_PUBLIC_APP_SERVER_URL ||
				"http://localhost:5000"
			console.log("[ChatPage] Fetching RTC token...")
			const rtcTokenResponse = await fetch(
				`${serverUrl}/voice/initiate`,
				{
					method: "POST",
					headers: {
						Authorization: `Bearer ${accessToken}`
					}
				}
			)
			if (!rtcTokenResponse.ok) {
				const errorData = await rtcTokenResponse
					.json()
					.catch(() => ({}))
				const error = new Error(
					errorData.detail || "Could not initiate voice session."
				)
				error.status = rtcTokenResponse.status
				throw error
			}
			const { rtc_token, ice_servers } = await rtcTokenResponse.json()
			console.log(
				`[ChatPage] RTC token fetched. ICE servers count: ${ice_servers?.iceServers?.length || 0}`
			)

			// Step 3: Create and connect WebRTCClient directly
			if (webrtcClientRef.current) {
				console.log(
					"[ChatPage] Disconnecting existing WebRTC client before creating new one."
				)
				webrtcClientRef.current.disconnect()
			}
			console.log("[ChatPage] Creating new WebRTCClient.")
			const client = new WebRTCClient({
				onConnected: () => handleStatusChange("connected"),
				onDisconnected: () => handleStatusChange("disconnected"),
				onAudioStream: (stream) => {
					if (remoteAudioRef.current) {
						remoteAudioRef.current.srcObject = stream
						remoteAudioRef.current
							.play()
							.catch((e) =>
								console.error(
									"[ChatPage] Error playing remote audio:",
									e
								)
							)
					}
				},
				onAudioLevel: handleAudioLevel,
				onEvent: handleVoiceEvent,
				iceServers: ice_servers.iceServers
			})
			webrtcClientRef.current = client

			// Step 3: Play ringing and connect
			if (ringtoneAudioRef.current) {
				ringtoneAudioRef.current.volume = 0.3
				ringtoneAudioRef.current.loop = true
				ringtoneAudioRef.current
					.play()
					.catch((e) =>
						console.error("[ChatPage] Error playing ringtone:", e)
					)
			}
			console.log("[ChatPage] Calling WebRTCClient.connect()...")
			await webrtcClientRef.current.connect(
				selectedAudioInputDevice,
				accessToken,
				rtc_token
			)
		} catch (error) {
			console.error("[ChatPage] Error during handleStartVoice:", error)
			toast.error(
				`Failed to connect: ${error.message || "Unknown error"}`
			)
			handleStatusChange("disconnected")
		}
	}

	const initializeVoiceMode = async () => {
		console.log("[ChatPage] Initializing voice mode...")
		// Check if devices are already loaded to avoid re-prompting
		if (audioInputDevices.length > 0) {
			console.log(
				"[ChatPage] Audio devices already available, skipping permission prompt."
			)
			return true
		}

		try {
			if (
				!navigator.mediaDevices ||
				!navigator.mediaDevices.enumerateDevices
			) {
				toast.error("Media devices are not supported in this browser.")
				return false
			}
			// This is the permission prompt
			console.log("[ChatPage] Requesting microphone permissions...")
			await navigator.mediaDevices.getUserMedia({
				audio: {
					noiseSuppression: false,
					echoCancellation: false
				},
				video: false
			})
			console.log("[ChatPage] Microphone permission granted.")
			const devices = await navigator.mediaDevices.enumerateDevices()
			const audioInputDevices = devices.filter(
				(d) => d.kind === "audioinput"
			)
			if (audioInputDevices.length > 0) {
				console.log(
					`[ChatPage] Found ${audioInputDevices.length} audio input devices.`
				)
				setAudioInputDevices(
					audioInputDevices.map((d, i) => ({
						deviceId: d.deviceId,
						label: d.label || `Microphone ${i + 1}`
					}))
				)
				// Set default device if not already set
				if (!selectedAudioInputDevice) {
					setSelectedAudioInputDevice(audioInputDevices[0].deviceId)
				}
				return true
			} else {
				toast.error("No audio input devices found.")
				console.warn("[ChatPage] No audio input devices found.")
				return false
			}
		} catch (error) {
			console.error(
				"[ChatPage] Error during voice initialization:",
				error
			)
			toast.error("Microphone permission is required for voice mode.")
			return false
		}
	}

	const handleToggleMute = () => {
		if (webrtcClientRef.current?.mediaStream) {
			const audioTracks =
				webrtcClientRef.current.mediaStream.getAudioTracks()
			if (audioTracks.length > 0) {
				const isCurrentlyEnabled = audioTracks[0].enabled
				audioTracks[0].enabled = !isCurrentlyEnabled
				const newMutedState = !audioTracks[0].enabled
				setIsMuted(newMutedState)
				console.log(
					`[ChatPage] Toggled mute. Mic is now ${
						newMutedState ? "muted" : "unmuted"
					}.`
				)
				setVoiceStatusText(newMutedState ? "Muted" : "Listening...")
			}
		}
	}

	const handleStopVoice = () => {
		if (connectionStatus === "disconnected" || !webrtcClientRef.current) {
			return
		}
		console.log("[ChatPage] handleStopVoice triggered.")

		webrtcClientRef.current?.disconnect()

		// --- ADD POSTHOG EVENT TRACKING & USAGE UPDATE ---
		if (voiceModeStartTimeRef.current) {
			const duration_seconds = Math.round(
				(Date.now() - voiceModeStartTimeRef.current) / 1000
			)
			console.log(
				`[ChatPage] Voice mode ended. Duration: ${duration_seconds} seconds.`
			)
			posthog?.capture("voice_mode_used", { duration_seconds })

			// Send usage update to the server
			console.log("[ChatPage] Sending voice usage update to server.")
			fetch("/api/voice/update-usage", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ duration_seconds })
			}).catch((err) =>
				console.error("[ChatPage] Failed to update voice usage:", err)
			)

			voiceModeStartTimeRef.current = null // Reset after tracking
		}
		// --- END POSTHOG EVENT TRACKING ---

		// 2. Immediately stop any playing audio.
		if (ringtoneAudioRef.current) {
			ringtoneAudioRef.current.pause()
			ringtoneAudioRef.current.currentTime = 0
		}
		if (connectedAudioRef.current) {
			connectedAudioRef.current.pause()
			connectedAudioRef.current.currentTime = 0
		}

		// 3. Force the UI state back to disconnected immediately.
		setConnectionStatus("disconnected")
		setVoiceStatusText("Click to start call")
		setIsMuted(false)
	}

	const toggleVoiceMode = async () => {
		if (isVoiceMode) {
			console.log("[ChatPage] Toggling voice mode OFF.")
			handleStopVoice()
			setIsVoiceMode(false)
			fetchInitialMessages()
		} else {
			console.log("[ChatPage] Toggling voice mode ON.")
			// Switching TO voice mode, first get permissions
			const permissionsGranted = await initializeVoiceMode()
			if (permissionsGranted) {
				console.log(
					"[ChatPage] Permissions granted, activating voice mode."
				)
				setIsVoiceMode(true)
			} else {
				console.warn(
					"[ChatPage] Permissions not granted, voice mode not activated."
				)
			}
		}
	}

	useEffect(() => {
		// This cleanup now only runs when the ChatPage component unmounts.
		// The handleStopVoice function is now the primary way to disconnect.
		return () => {
			webrtcClientRef.current?.disconnect()
		}
	}, [])

	return {
		// State
		displayedMessages,
		setDisplayedMessages,
		input,
		setInput,
		isLoading,
		setIsLoading,
		thinking,
		setThinking,
		textareaRef,
		chatEndRef,
		abortControllerRef,
		scrollContainerRef,
		fileInputRef,
		isLoadingOlder,
		setIsLoadingOlder,
		hasMoreMessages,
		setHasMoreMessages,
		searchingForMessageId,
		setSearchingForMessageId,
		userDetails,
		setUserDetails,
		posthog,
		isFocused,
		setIsFocused,
		isWelcomeModalOpen,
		setIsWelcomeModalOpen,
		replyingTo,
		setReplyingTo,
		isOptionsOpen,
		setIsOptionsOpen,
		confirmClear,
		setConfirmClear,
		integrations,
		setIntegrations,
		searchParams,
		router,
		tourState,
		prevTourState,
		selectedFile,
		setSelectedFile,
		isUploading,
		setIsUploading,
		uploadedFilename,
		setUploadedFilename,
		isMuted,
		setIsMuted,
		isVoiceMode,
		setIsVoiceMode,
		connectionStatus,
		setConnectionStatus,
		audioInputDevices,
		setAudioInputDevices,
		selectedAudioInputDevice,
		setSelectedAudioInputDevice,
		voiceStatusText,
		setVoiceStatusText,
		statusText,
		setStatusText,
		audioLevel,
		setAudioLevel,
		webrtcClientRef,
		ringtoneAudioRef,
		connectedAudioRef,
		remoteAudioRef,
		voiceModeStartTimeRef,
		lastSpokenTextRef,

		// Functions
		setMicrophoneEnabled,
		fetchInitialMessages,
		fetchUserDetails,
		sendMessage,
		fetchIntegrations,
		fetchOlderMessages,
		handleInputChange,
		handleReply,
		handleFileChange,
		handleDeleteMessage,
		handleClearAllMessages,
		handleStopStreaming,
		getGreeting,
		handleStatusChange,
		handleVoiceEvent,
		handleAudioLevel,
		handleStartVoice,
		initializeVoiceMode,
		handleToggleMute,
		handleStopVoice,
		toggleVoiceMode
	}
}
