"use client"

import {
	useState,
	useEffect,
	useRef,
	useCallback,
	useMemo,
	Fragment
} from "react"
import { useSearchParams, useRouter } from "next/navigation"
import {
	IconLoader,
	IconCheck,
	IconClockHour4,
	IconMessageChatbot,
	IconTool,
	IconSparkles,
	IconX
} from "@tabler/icons-react"
import { Tooltip } from "react-tooltip"
import { motion, AnimatePresence } from "framer-motion"
import InteractiveNetworkBackground from "@components/ui/InteractiveNetworkBackground"
import React from "react"
import { Button } from "@components/ui/button"
import ChatHeader from "@components/chat/ChatHeader"
import ChatInputArea from "@components/chat/ChatInputArea"
import ChatMessageList from "@components/chat/ChatMessageList"
import VoiceModeUI from "@components/chat/VoiceModeUI"
import { usePostHog } from "posthog-js/react"
import {
	useQuery,
	useInfiniteQuery,
	useMutation,
	useQueryClient
} from "@tanstack/react-query"
import { useTourStore, useChatStore } from "@stores/app-stores"
import { toast } from "react-hot-toast"

export default function ChatPage() {
	const [input, setInput] = useState("")
	const [statusText, setStatusText] = useState("")
	const [isWelcomeModalOpen, setIsWelcomeModalOpen] = useState(false)

	const textareaRef = useRef(null)
	const chatEndRef = useRef(null)
	const abortControllerRef = useRef(null)
	const scrollContainerRef = useRef(null)
	const fileInputRef = useRef(null)
	const queryClient = useQueryClient()

	// State for tour simulation messages
	const [tourMessages, setTourMessages] = useState([])

	// State for infinite scroll
	const [searchingForMessageId, setSearchingForMessageId] = useState(null)

	// State for UI enhancements
	const posthog = usePostHog()
	const [replyingTo, setReplyingTo] = useState(null)
	const [isOptionsOpen, setIsOptionsOpen] = useState(false)
	const [confirmClear, setConfirmClear] = useState(false)

	// Zustand stores
	const {
		tourState,
		prevTourState,
		chatActionsRef,
		startTour,
		nextStep,
		nextSubStep,
		setHighlightPaused
	} = useTourStore()
	const {
		isVoiceMode,
		connectionStatus,
		isMuted,
		voiceStatusText,
		audioLevel,
		setVoiceMode,
		setConnectionStatus,
		setIsMuted,
		setVoiceStatusText,
		setAudioLevel
	} = useChatStore()

	const searchParams = useSearchParams()
	const router = useRouter()

	// --- File Upload State ---
	const [selectedFile, setSelectedFile] = useState(null)
	const [uploadedFilename, setUploadedFilename] = useState(null)

	// --- Voice Mode State ---
	const [audioInputDevices, setAudioInputDevices] = useState([])
	const [selectedAudioInputDevice, setSelectedAudioInputDevice] = useState("")
	const webrtcClientRef = useRef(null)
	const ringtoneAudioRef = useRef(null)
	const connectedAudioRef = useRef(null)
	const remoteAudioRef = useRef(null)
	const voiceModeStartTimeRef = useRef(null)

	const lastSpokenTextRef = useRef("")
	const setMicrophoneEnabled = useCallback(
		(enabled) => {
			if (webrtcClientRef.current?.mediaStream) {
				const audioTracks =
					webrtcClientRef.current.mediaStream.getAudioTracks()
				if (audioTracks.length > 0) {
					if (audioTracks[0].enabled !== enabled) {
						audioTracks[0].enabled = enabled
						setIsMuted(!enabled)
					}
				}
			}
		},
		[setIsMuted]
	)

	// --- TanStack Query Hooks ---

	const { data: userDetails } = useQuery({
		queryKey: ["userProfile"],
		queryFn: async () => {
			const res = await fetch("/api/user/profile")
			if (!res.ok) throw new Error("Failed to fetch user details")
			return res.json()
		},
		staleTime: Infinity // User profile rarely changes
	})

	const {
		data: chatHistoryData,
		fetchNextPage,
		hasNextPage,
		isFetchingNextPage: isLoadingOlder,
		isLoading: isHistoryLoading,
		isError: isHistoryError
	} = useInfiniteQuery({
		queryKey: ["chatHistory"],
		queryFn: async ({ pageParam }) => {
			const res = await fetch("/api/chat/history", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ limit: 50, before_timestamp: pageParam })
			})
			if (!res.ok) throw new Error("Failed to fetch messages")
			const pageData = await res.json()
			const messages = (pageData.messages || []).map((m) => ({
				...m,
				id: m.message_id
			}))
			return { messages, hasMore: messages.length === 50 }
		},
		initialPageParam: null,
		getNextPageParam: (lastPage) => {
			if (lastPage.hasMore && lastPage.messages.length > 0) {
				return lastPage.messages[0].timestamp
			}
			return undefined
		},
		enabled: !tourState?.isActive
	})

	const displayedMessagesFromQuery = useMemo(
		() => chatHistoryData?.pages.flatMap((page) => page.messages) ?? [],
		[chatHistoryData]
	)

	// Use tour messages if the tour is active, otherwise use data from the query
	const displayedMessages = tourState?.isActive
		? tourMessages
		: displayedMessagesFromQuery

	useEffect(() => {
		// When tour becomes inactive, refetch original messages
		if (!tourState?.isActive && prevTourState?.isActive) {
			queryClient.invalidateQueries({ queryKey: ["chatHistory"] })
		}
	}, [tourState?.isActive, prevTourState, queryClient])

	useEffect(() => {
		if (
			searchParams.get("show_demo") === "true" &&
			startTour &&
			!tourState?.isActive
		) {
			setTourMessages([]) // Reset tour messages on start
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
		if (searchingForMessageId && hasNextPage && !isLoadingOlder) {
			fetchNextPage()
		}
	}, [searchingForMessageId, hasNextPage, isLoadingOlder, fetchNextPage])

	const sendMessageMutation = useMutation({
		mutationFn: async ({
			newUserMessage,
			uploadedFilename: mutationUploadedFilename
		}) => {
			abortControllerRef.current = new AbortController()

			posthog?.capture("chat_message_sent", {
				message_length: newUserMessage.content.length,
				has_file: !!mutationUploadedFilename
			})

			let finalContent = newUserMessage.content
			if (mutationUploadedFilename) {
				finalContent = `(Attached file for context: ${mutationUploadedFilename}) ${finalContent}. Use file-management MCP to read it`
			}

			if (tourState?.isActive && tourState.step === 1) {
				// --- TOUR SIMULATION ---
				const subStep = tourState.subStep
				setHighlightPaused(true)
				setStatusText("Thinking...")
				setTourMessages((prev) => [...prev, newUserMessage])

				if (subStep === 0) {
					// First message: "Hi Sentient!"
					setTimeout(() => {
						const fakeResponse = {
							id: `assistant-tour-0`,
							role: "assistant",
							content: "Hey there, I'm ready to help.",
							timestamp: new Date().toISOString()
						}
						setTourMessages((prev) => [...prev, fakeResponse])
						setStatusText("")
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
							id: `assistant-tour-1`,
							role: "assistant",
							content:
								"Cool, I've sent that email. Is there anything else you want to do?",
							timestamp: new Date().toISOString(),
							tools: ["gmail"]
						}
						setTourMessages((prev) => [...prev, fakeResponse])
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
							id: `assistant-tour-2`,
							role: "assistant",
							content: "Cool, I've created the workflow.",
							timestamp: new Date().toISOString(),
							tools: ["tasks"]
						}
						setTourMessages((prev) => [...prev, fakeResponse])
						setStatusText("")
						// This is the last chat step, move to the next main step.
						setTimeout(() => {
							nextStep()
						}, 2000)
					}, 2500)
				}
				return // End simulation here
			}

			const messageToSend = {
				...newUserMessage,
				content: finalContent
			}
			delete messageToSend.assistantTempId

			const responsePromise = fetch("/api/chat/message", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					messages: [messageToSend]
				}),
				signal: abortControllerRef.current.signal
			})
			const response = await responsePromise
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

						if (parsed.type === "status") {
							setStatusText(parsed.message)
							continue
						}

						if (parsed.type === "assistantStream" && parsed.token) {
							setStatusText("")
						}

						queryClient.setQueryData(["chatHistory"], (oldData) => {
							if (!oldData) return oldData
							const newPages = oldData.pages.map(
								(page, pageIndex) => {
									if (
										pageIndex ===
										oldData.pages.length - 1
									) {
										const newMessages = page.messages.map(
											(msg) => {
												if (
													msg.id ===
													newUserMessage.assistantTempId
												) {
													let newId = msg.id
													if (
														parsed.messageId &&
														msg.id.startsWith(
															"assistant-"
														)
													) {
														newId = parsed.messageId
													}
													if (parsed.done) {
														return {
															...msg,
															id: newId,
															content:
																parsed.final_content ||
																"",
															turn_steps:
																parsed.turn_steps ||
																[],
															tools:
																parsed.tools ||
																[]
														}
													}
													return {
														...msg,
														id: newId,
														tools:
															parsed.tools ||
															msg.tools
													}
												}
												return msg
											}
										)
										return {
											...page,
											messages: newMessages
										}
									}
									return page
								}
							)
							return { ...oldData, pages: newPages }
						})
					} catch (parseError) {
						// This might be raw text if streaming fails to produce JSON
					}
				}
			}
		},
		onMutate: async ({ newUserMessage }) => {
			await queryClient.cancelQueries({ queryKey: ["chatHistory"] })
			const previousHistory = queryClient.getQueryData(["chatHistory"])

			queryClient.setQueryData(["chatHistory"], (oldData) => {
				const newPage = {
					messages: [
						newUserMessage,
						{
							id: newUserMessage.assistantTempId,
							role: "assistant",
							content: "",
							timestamp: new Date().toISOString(),
							tools: [],
							turn_steps: []
						}
					],
					hasMore: oldData?.pages[oldData.pages.length - 1]?.hasMore
				}

				const lastPage = oldData?.pages[oldData.pages.length - 1]
				const newPages = oldData?.pages
					? [
							...oldData.pages.slice(0, -1),
							{
								...lastPage,
								messages: [
									...lastPage.messages,
									...newPage.messages
								]
							}
						]
					: [newPage]

				return {
					...oldData,
					pages: newPages
				}
			})

			setInput("")
			setReplyingTo(null)
			setUploadedFilename(null)
			setSelectedFile(null)
			if (textareaRef.current) textareaRef.current.style.height = "auto"

			return { previousHistory }
		},
		onError: (error, variables, context) => {
			queryClient.setQueryData(["chatHistory"], context.previousHistory)
			if (error.name === "AbortError") {
				toast.info("Message generation stopped.")
			} else {
				toast.error(`Error: ${error.message}`)
			}
			console.error("Fetch error:", error)
		},
		onSettled: () => {
			setStatusText("")
			queryClient.invalidateQueries({ queryKey: ["chatHistory"] })
		}
	})

	const sendMessage = () => {
		const newUserMessage = {
			id: `user-${Date.now()}`,
			assistantTempId: `assistant-${Date.now()}`,
			role: "user",
			content: input.trim(),
			timestamp: new Date().toISOString(),
			...(replyingTo && { replyToId: replyingTo.id })
		}
		sendMessageMutation.mutate({ newUserMessage, uploadedFilename })
	}

	useEffect(() => {
		if (chatActionsRef) {
			chatActionsRef.current = {
				setInput: setInput,
				sendMessage: sendMessage
			}
		}
		return () => {
			if (chatActionsRef) {
				chatActionsRef.current = null
			}
		}
	}, [chatActionsRef, sendMessage, setInput])

	const { data: integrationsData } = useQuery({
		queryKey: ["integrations"],
		queryFn: async () => {
			const res = await fetch("/api/settings/integrations", {
				method: "POST"
			})
			if (!res.ok) throw new Error("Failed to fetch integrations")
			return res.json()
		}
	})
	const integrations = integrationsData?.integrations || []

	useEffect(() => {
		const container = scrollContainerRef.current
		const handleScroll = () => {
			if (container && container.scrollTop === 0 && hasNextPage) {
				fetchNextPage()
			}
		}
		container?.addEventListener("scroll", handleScroll)
		return () => container?.removeEventListener("scroll", handleScroll)
	}, [fetchNextPage, hasNextPage])

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

	const uploadFileMutation = useMutation({
		mutationFn: async (file) => {
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
			return response.json()
		},
		onSuccess: (result) => {
			setUploadedFilename(result.filename)
			toast.success(`${result.filename} uploaded successfully.`)
		},
		onError: (error) => {
			toast.error(`Error: ${error.message}`)
			setSelectedFile(null)
		}
	})

	const handleFileChange = (event) => {
		const file = event.target.files?.[0]
		if (!file) return

		event.target.value = ""

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
		if (file.size > 5 * 1024 * 1024) {
			// 5MB limit
			toast.error(
				"File is too large. Please select a file smaller than 5MB."
			)
			return
		}

		setSelectedFile(file)
		setUploadedFilename(null)
		uploadFileMutation.mutate(file)
	}

	const deleteMessageMutation = useMutation({
		mutationFn: (messageId) =>
			fetch("/api/chat/delete", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ message_id: messageId })
			}).then((res) => {
				if (!res.ok) throw new Error("Failed to delete message")
				return res.json()
			}),
		onMutate: async (messageId) => {
			await queryClient.cancelQueries({ queryKey: ["chatHistory"] })
			const previousHistory = queryClient.getQueryData(["chatHistory"])
			queryClient.setQueryData(["chatHistory"], (oldData) => {
				if (!oldData) return oldData
				const newPages = oldData.pages.map((page) => ({
					...page,
					messages: page.messages.filter((m) => m.id !== messageId)
				}))
				return { ...oldData, pages: newPages }
			})
			return { previousHistory }
		},
		onSuccess: () => {
			toast.success("Message deleted.")
		},
		onError: (error, variables, context) => {
			toast.error(error.message)
			queryClient.setQueryData(["chatHistory"], context.previousHistory)
		},
		onSettled: () => {
			queryClient.invalidateQueries({ queryKey: ["chatHistory"] })
		}
	})
	const handleDeleteMessage = (messageId) => {
		deleteMessageMutation.mutate(messageId)
	}

	const clearAllMessagesMutation = useMutation({
		mutationFn: () =>
			fetch("/api/chat/delete", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ clear_all: true })
			}).then((res) => {
				if (!res.ok) throw new Error("Failed to clear chat history")
				return res.json()
			}),
		onMutate: async () => {
			await queryClient.cancelQueries({ queryKey: ["chatHistory"] })
			const previousHistory = queryClient.getQueryData(["chatHistory"])
			queryClient.setQueryData(["chatHistory"], () => ({
				pages: [],
				pageParams: []
			}))
			return { previousHistory }
		},
		onSuccess: () => {
			toast.success("Chat history cleared.")
		},
		onError: (error, variables, context) => {
			toast.error(error.message)
			queryClient.setQueryData(["chatHistory"], context.previousHistory)
		}
	})

	const handleClearAllMessages = () => {
		setIsOptionsOpen(false)
		setConfirmClear(false)
		clearAllMessagesMutation.mutate()
	}

	const handleStopStreaming = () => {
		if (abortControllerRef.current) {
			abortControllerRef.current.abort()
			toast.info("Message generation stopped.")
		}
	}

	useEffect(() => {
		if (chatEndRef.current && !isVoiceMode) {
			chatEndRef.current.scrollIntoView({ behavior: "auto" })
		}
	}, [displayedMessages, sendMessageMutation.isPending, isVoiceMode])

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
		[setMicrophoneEnabled, setConnectionStatus, setVoiceStatusText]
	)

	const handleVoiceEvent = useCallback(
		(event) => {
			const addMessageToCache = (message) => {
				queryClient.setQueryData(["chatHistory"], (oldData) => {
					if (!oldData)
						return {
							pages: [{ messages: [message], hasMore: false }],
							pageParams: [null]
						}
					const lastPage = oldData.pages[oldData.pages.length - 1]
					const newPages = [
						...oldData.pages.slice(0, -1),
						{
							...lastPage,
							messages: [...lastPage.messages, message]
						}
					]
					return { ...oldData, pages: newPages }
				})
			}

			if (event.type === "stt_result" && event.text) {
				const userMessage = {
					id: `user_${Date.now()}`,
					role: "user",
					content: event.text,
					timestamp: new Date().toISOString()
				}
				addMessageToCache(userMessage)
			} else if (event.type === "llm_result" && event.text) {
				lastSpokenTextRef.current = event.text
				const assistantMessage = {
					id: event.messageId || `assistant_${Date.now()}`,
					role: "assistant",
					content: event.text,
					timestamp: new Date().toISOString()
				}
				addMessageToCache(assistantMessage)
				queryClient.invalidateQueries({ queryKey: ["chatHistory"] })
			} else if (event.type === "status") {
				console.log(`[ChatPage] Voice status update: ${event.message}`)
				if (event.message === "thinking") {
					setVoiceStatusText("Thinking...")
					setMicrophoneEnabled(false)
				} else if (event.message === "speaking") {
					setVoiceStatusText("Speaking...")
					setMicrophoneEnabled(false)
				} else if (event.message === "listening") {
					const textToMeasure = lastSpokenTextRef.current
					const estimatedDuration = textToMeasure.length * 55 + 250
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

					lastSpokenTextRef.current = ""
				} else if (event.message === "transcribing") {
					setVoiceStatusText("Transcribing...")
					setMicrophoneEnabled(false)
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
		[setMicrophoneEnabled, queryClient, setVoiceStatusText]
	)

	const handleAudioLevel = useCallback(
		(level) => {
			setAudioLevel((prev) => prev * 0.7 + level * 0.3)
		},
		[setAudioLevel]
	)

	const handleStartVoice = async () => {
		if (connectionStatus !== "disconnected") return
		console.log("[ChatPage] handleStartVoice triggered.")
		posthog?.capture("voice_mode_activated")
		voiceModeStartTimeRef.current = Date.now()

		setConnectionStatus("connecting")
		setVoiceStatusText("Connecting...")
		try {
			const tokenResponse = await fetch("/api/auth/token")
			if (!tokenResponse.ok) throw new Error("Could not get auth token.")
			const { accessToken } = await tokenResponse.json()

			const serverUrl =
				process.env.NEXT_PUBLIC_APP_SERVER_URL ||
				"http://localhost:5000"
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

			if (webrtcClientRef.current) {
				webrtcClientRef.current.disconnect()
			}
			const { WebRTCClient } = await import("@lib/webrtc-client")
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

			if (ringtoneAudioRef.current) {
				ringtoneAudioRef.current.volume = 0.3
				ringtoneAudioRef.current.loop = true
				ringtoneAudioRef.current
					.play()
					.catch((e) =>
						console.error("[ChatPage] Error playing ringtone:", e)
					)
			}
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
		if (audioInputDevices.length > 0) {
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
			await navigator.mediaDevices.getUserMedia({
				audio: {
					noiseSuppression: false,
					echoCancellation: false
				},
				video: false
			})
			const devices = await navigator.mediaDevices.enumerateDevices()
			const audioInputDevices = devices.filter(
				(d) => d.kind === "audioinput"
			)
			if (audioInputDevices.length > 0) {
				setAudioInputDevices(
					audioInputDevices.map((d, i) => ({
						deviceId: d.deviceId,
						label: d.label || `Microphone ${i + 1}`
					}))
				)
				if (!selectedAudioInputDevice) {
					setSelectedAudioInputDevice(audioInputDevices[0].deviceId)
				}
				return true
			} else {
				toast.error("No audio input devices found.")
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
				setVoiceStatusText(newMutedState ? "Muted" : "Listening...")
			}
		}
	}

	const handleStopVoice = () => {
		if (connectionStatus === "disconnected" || !webrtcClientRef.current) {
			return
		}
		webrtcClientRef.current?.disconnect()

		if (voiceModeStartTimeRef.current) {
			const duration_seconds = Math.round(
				(Date.now() - voiceModeStartTimeRef.current) / 1000
			)
			posthog?.capture("voice_mode_used", { duration_seconds })

			fetch("/api/voice/update-usage", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ duration_seconds })
			}).catch((err) =>
				console.error("[ChatPage] Failed to update voice usage:", err)
			)
			voiceModeStartTimeRef.current = null
		}

		if (ringtoneAudioRef.current) {
			ringtoneAudioRef.current.pause()
			ringtoneAudioRef.current.currentTime = 0
		}
		if (connectedAudioRef.current) {
			connectedAudioRef.current.pause()
			connectedAudioRef.current.currentTime = 0
		}

		setConnectionStatus("disconnected")
		setVoiceStatusText("Click to start call")
		setIsMuted(false)
	}

	const toggleVoiceMode = async () => {
		if (isVoiceMode) {
			setVoiceMode(false)
			handleStopVoice()
			queryClient.invalidateQueries({ queryKey: ["chatHistory"] })
		} else {
			const permissionsGranted = await initializeVoiceMode()
			if (permissionsGranted) {
				setVoiceMode(true)
			}
		}
	}

	useEffect(() => {
		return () => {
			webrtcClientRef.current?.disconnect()
		}
	}, [])

	const renderWelcomeModal = () => (
		<AnimatePresence>
			{isWelcomeModalOpen && (
				<motion.div
					initial={{ opacity: 0, backdropFilter: "blur(0px)" }}
					animate={{ opacity: 1, backdropFilter: "blur(12px)" }}
					exit={{ opacity: 0, backdropFilter: "blur(0px)" }}
					className="fixed inset-0 bg-black/70 z-[60] flex items-center justify-center p-4 md:p-6"
					onClick={() => setIsWelcomeModalOpen(false)}
				>
					<motion.div
						initial={{ opacity: 0, y: 20 }}
						animate={{ opacity: 1, y: 0 }}
						exit={{ opacity: 0, y: 20 }}
						transition={{ duration: 0.2, ease: "easeInOut" }}
						onClick={(e) => e.stopPropagation()}
						className="relative bg-neutral-900/80 backdrop-blur-2xl p-6 rounded-2xl shadow-2xl w-full max-w-2xl md:h-auto md:max-h-[85vh] h-full border border-neutral-700 flex flex-col"
					>
						<header className="flex justify-between items-center mb-6 flex-shrink-0">
							<h2 className="text-lg font-semibold text-white flex items-center gap-2">
								<IconMessageChatbot /> Welcome to Unified Chat
							</h2>
							<Button
								onClick={() => setIsWelcomeModalOpen(false)}
								variant="ghost"
								size="icon"
								className="rounded-full"
							>
								<IconX size={18} />
							</Button>
						</header>
						<main className="flex-1 overflow-y-auto custom-scrollbar pr-2 text-left space-y-6">
							<p className="text-neutral-300">
								This is your single, continuous conversation
								with me. No need to juggle multiple chats—just
								keep the dialogue flowing. Here’s how it works:
							</p>
							<div className="space-y-4">
								<div className="flex items-start gap-4">
									<IconSparkles
										size={20}
										className="text-brand-orange flex-shrink-0 mt-1"
									/>
									<div>
										<h3 className="font-semibold text-white">
											One Conversation, Infinite History
										</h3>
										<p className="text-neutral-400 text-sm mt-1">
											I remember our entire conversation,
											so you can always pick up where you
											left off.
										</p>
									</div>
								</div>
								<div className="flex items-start gap-4">
									<IconTool
										size={20}
										className="text-brand-orange flex-shrink-0 mt-1"
									/>
									<div>
										<h3 className="font-semibold text-white">
											Dynamic Tools for Any Task
										</h3>
										<p className="text-neutral-400 text-sm mt-1">
											I automatically select and use the
											right tools from your connected
											apps. Just tell me what you need,
											and I'll figure out how to get it
											done.
										</p>
									</div>
								</div>
								<div className="flex items-start gap-4">
									<IconClockHour4
										size={20}
										className="text-brand-orange flex-shrink-0 mt-1"
									/>
									<div>
										<h3 className="font-semibold text-white">
											Schedule for Later
										</h3>
										<p className="text-neutral-400 text-sm mt-1">
											Tell me to do something 'tomorrow at
											9am' or 'next Friday', and I'll
											handle it in the background, keeping
											you updated in the Tasks panel.
										</p>
									</div>
								</div>
							</div>
						</main>
						<footer className="mt-6 pt-4 border-t border-neutral-800 flex justify-end">
							<Button
								onClick={() => setIsWelcomeModalOpen(false)}
								variant="secondary"
							>
								Got it
							</Button>
						</footer>
					</motion.div>
				</motion.div>
			)}
		</AnimatePresence>
	)

	return (
		<div className="flex-1 flex h-screen text-white overflow-hidden">
			<Tooltip id="home-tooltip" place="right" style={{ zIndex: 9999 }} />
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
			{renderWelcomeModal()}
			<audio ref={remoteAudioRef} autoPlay playsInline />
			<div className="flex-1 flex flex-col overflow-hidden relative w-full pt-16 md:pt-0">
				<div className="absolute inset-0 z-[-1] network-grid-background">
					<InteractiveNetworkBackground />
				</div>
				<div className="absolute -top-[250px] left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-brand-orange/10 rounded-full blur-3xl -z-10" />

				<main
					ref={scrollContainerRef}
					className="flex-1 overflow-y-auto sm:p-0 md:px-4 pb-4 md:p-6 flex flex-col custom-scrollbar"
				>
					{isHistoryLoading ? (
						<div className="flex-1 flex justify-center items-center">
							<IconLoader className="animate-spin text-neutral-500" />
						</div>
					) : isVoiceMode ? (
						<VoiceModeUI
							{...{
								connectionStatus,
								audioLevel,
								selectedAudioInputDevice,
								setSelectedAudioInputDevice,
								audioInputDevices,
								isMuted,
								handleToggleMute,
								handleStartVoice,
								handleStopVoice,
								toggleVoiceMode,
								voiceStatusText,
								displayedMessages
							}}
						/>
					) : displayedMessages.length === 0 &&
					  !sendMessageMutation.isPending ? (
						<div className="flex-1 flex flex-col justify-center items-center p-4 md:p-6">
							<ChatHeader
								{...{
									getGreeting,
									userDetails,
									isOptionsOpen,
									setIsOptionsOpen,
									confirmClear,
									setConfirmClear,
									handleClearAllMessages
								}}
							/>
						</div>
					) : (
						<ChatMessageList
							{...{
								scrollContainerRef,
								isLoadingOlder,
								displayedMessages,
								thinking: sendMessageMutation.isPending,
								statusText,
								chatEndRef,
								handleReply,
								handleDeleteMessage
							}}
						/>
					)}
				</main>
				{!isHistoryLoading && !isVoiceMode && (
					<div className="flex-shrink-0 bg-transparent">
						<div className="relative w-full max-w-4xl mx-auto px-2 pt-2 pb-4 sm:px-6 sm:pb-6">
							<ChatInputArea
								{...{
									input,
									handleInputChange,
									sendMessage,
									textareaRef,
									uploadedFilename,
									isUploading: uploadFileMutation.isPending,
									fileInputRef,
									handleFileChange,
									integrations,
									setIsWelcomeModalOpen,
									toggleVoiceMode,
									thinking: sendMessageMutation.isPending,
									handleStopStreaming,
									replyingTo,
									setReplyingTo,
									setUploadedFilename,
									setSelectedFile
								}}
							/>
						</div>
						{displayedMessages.length === 0 &&
							!sendMessageMutation.isPending && (
								<div className="mt-12"></div>
							)}
					</div>
				)}
			</div>
		</div>
	)
}
