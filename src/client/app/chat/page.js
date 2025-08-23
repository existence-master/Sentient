"use client"

import { useState, useEffect, useRef, useCallback, useMemo } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import {
	IconSend,
	IconLoader,
	IconPlayerStopFilled,
	IconBrain,
	IconWorldSearch,
	IconFileText,
	IconArrowBackUp,
	IconX,
	IconDotsVertical,
	IconPhone,
	IconPhoneOff,
	IconWaveSine,
	IconMicrophone,
	IconMicrophoneOff,
	IconMessageOff,
	IconPaperclip,
	IconFile,
	IconPlus,
	IconTools,
	IconTool,
	IconInfoCircle,
	IconSparkles,
	IconCheck,
	IconClockHour4,
	IconMessageChatbot,
	IconMapPin,
	IconChartPie,
	IconBrandTrello,
	IconNews,
	IconBrandDiscord,
	IconBrandWhatsapp,
	IconCalendarEvent
} from "@tabler/icons-react"
import {
	IconBrandSlack,
	IconBrandNotion,
	IconBrandGithub,
	IconBrandGoogleDrive
} from "@tabler/icons-react"
import IconGoogleMail from "@components/icons/IconGoogleMail"
import toast from "react-hot-toast"
import { cn } from "@utils/cn"
import { Tooltip } from "react-tooltip"
import { motion, AnimatePresence } from "framer-motion"
import ChatBubble from "@components/ChatBubble"
import { TextLoop } from "@components/ui/TextLoop"
import InteractiveNetworkBackground from "@components/ui/InteractiveNetworkBackground"
import { TextShimmer } from "@components/ui/text-shimmer"
import Script from "next/script"
import React from "react"
import { usePostHog } from "posthog-js/react"
import SiriSpheres from "@components/voice-visualization/SiriSpheres"
import { WebRTCClient } from "@lib/webrtc-client"
import useClickOutside from "@hooks/useClickOutside"
import { usePlan } from "@hooks/usePlan"
import { POST } from "@app/api/memories/route"

const toolIcons = {
	gmail: IconGoogleMail,
	gdocs: IconFileText,
	gdrive: IconBrandGoogleDrive,
	slack: IconBrandSlack,
	notion: IconBrandNotion,
	github: IconBrandGithub,
	internet_search: IconWorldSearch,
	memory: IconBrain,
	gmaps: IconMapPin,
	quickchart: IconChartPie,
	google_search: IconWorldSearch,
	trello: IconBrandTrello,
	news: IconNews,
	discord: IconBrandDiscord,
	whatsapp: IconBrandWhatsapp,
	gcalendar_alt: IconCalendarEvent,
	default: IconTool
}

const proPlanFeatures = [
	{ name: "Text Chat", limit: "100 messages per day" },
	{ name: "Voice Chat", limit: "10 minutes per day" },
	{ name: "One-Time Tasks", limit: "20 async tasks per day" },
	{ name: "Recurring Tasks", limit: "10 active recurring workflows" },
	{ name: "Triggered Tasks", limit: "10 triggered workflows" },
	{
		name: "Parallel Agents",
		limit: "5 complex tasks per day with 50 sub agents"
	},
	{ name: "File Uploads", limit: "20 files per day" },
	{ name: "Memories", limit: "Unlimited memories" },
	{
		name: "Other Integrations",
		limit: "Notion, GitHub, Slack, Discord, Trello"
	}
]

const UpgradeToProModal = ({ isOpen, onClose }) => {
	if (!isOpen) return null

	const handleUpgrade = () => {
		const dashboardUrl = process.env.NEXT_PUBLIC_LANDING_PAGE_URL
		if (dashboardUrl) {
			window.location.href = `${dashboardUrl}/dashboard`
		}
		onClose()
	}

	return (
		<AnimatePresence>
			{isOpen && (
				<motion.div
					initial={{ opacity: 0 }}
					animate={{ opacity: 1 }}
					exit={{ opacity: 0 }}
					className="fixed inset-0 bg-black/70 backdrop-blur-md z-[100] flex items-center justify-center p-4"
					onClick={onClose}
				>
					<motion.div
						initial={{ scale: 0.95, y: 20 }}
						animate={{ scale: 1, y: 0 }}
						exit={{ scale: 0.95, y: -20 }}
						transition={{ duration: 0.2, ease: "easeInOut" }}
						onClick={(e) => e.stopPropagation()}
						className="relative bg-neutral-900/90 backdrop-blur-xl p-6 rounded-2xl shadow-2xl w-full max-w-lg border border-neutral-700 flex flex-col"
					>
						<header className="text-center mb-4">
							<h2 className="text-2xl font-bold text-white flex items-center justify-center gap-2">
								<IconSparkles className="text-brand-orange" />
								Upgrade to Pro
							</h2>
							<p className="text-neutral-400 mt-2">
								Unlock Voice Mode and other powerful features.
							</p>
						</header>
						<main className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-4 my-4">
							{proPlanFeatures.map((feature) => (
								<div
									key={feature.name}
									className="flex items-start gap-2.5"
								>
									<IconCheck
										size={18}
										className="text-green-400 flex-shrink-0 mt-0.5"
									/>
									<div>
										<p className="text-white text-sm font-medium">
											{feature.name}
										</p>
										<p className="text-neutral-400 text-xs">
											{feature.limit}
										</p>
									</div>
								</div>
							))}
						</main>
						<footer className="mt-4 flex flex-col gap-2">
							<button
								onClick={handleUpgrade}
								className="w-full py-2.5 px-5 rounded-lg bg-brand-orange hover:bg-brand-orange/90 text-brand-black font-semibold transition-colors"
							>
								Upgrade to Pro - $9/month
							</button>
							<button
								onClick={onClose}
								className="w-full py-2 px-5 rounded-lg hover:bg-neutral-800 text-sm font-medium text-neutral-400"
							>
								Not now
							</button>
						</footer>
					</motion.div>
				</motion.div>
			)}
		</AnimatePresence>
	)
}

const StorylaneDemoModal = ({ onClose }) => {
	// The script adds a global Storylane object. The button's onClick will use it.
	const embedHtml = `
        <style>
            .sl-heading-text { max-width:53%; }
            @media (max-width: 1024px) { .sl-heading-text { max-width:90%; } }
        </style>
        <div class="sl-embed-container" style="position:relative;display:flex;align-items:center;justify-content:center;border: 1px solid rgba(63,95,172,0.35);box-shadow: 0px 0px 18px rgba(26, 19, 72, 0.15);border-radius:10px">
            <div class="sl-preview-heading" style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background-color:transparent;z-index:999999;font-family:Poppins, Arial, sans-serif;font-size:clamp(20px, 2.664vw, 28px);font-weight:500;line-height:normal;text-align:center;border-radius:10px;">
                <button onclick="Storylane.Play({type: 'preview_embed',demo_type: 'image', width: 1920, height: 918, element: this, demo_url: 'https://app.storylane.io/demo/d6oo4tbg4fbb?embed=inline_overlay'})" class="sl-preview-cta" style="background-color:#9939EB;border:none;border-radius:8px;box-shadow:0px 0px 15px rgba(26, 19, 72, 0.45);color:#FFFFFF;display:inline-block;font-family:Poppins, Arial, sans-serif;font-size:clamp(16px, 1.599vw, 20px);font-weight:600;height:clamp(40px, 3.996vw, 50px);line-height:1.2;padding:0 clamp(15px, 1.776vw, 20px);text-overflow:ellipsis;transform:translateZ(0);transition:background 0.4s;white-space:nowrap;width:auto;z-index:999999;cursor:pointer">Take a Tour</button>
            </div>
            <div class="sl-embed" data-sl-demo-type="image" style="position:relative;padding-bottom:calc(47.81% + 25px);width:100%;height:0;transform:scale(1);overflow:hidden;">
                <div class="sl-preview" style="width:100%;height:100%;z-index:99999;position:absolute;background:url('https://storylane-prod-uploads.s3.us-east-2.amazonaws.com/company/company_35e9ec7f-ae05-4316-ad70-1f931aaacad6/project/project_8f4dcfca-3161-478c-834d-28eb1993ad1b/page/fFud9OKzDoQ4hDFpQaYu7.jpg') no-repeat;background-size:100% 100%;border-radius:inherit;filter:blur(2px)"></div>
                <iframe class="sl-demo" src="" name="sl-embed" allow="fullscreen" allowfullscreen style="display:none;position:absolute;top:0;left:0;width:100%;height:100%;border:none;"></iframe>
            </div>
            <iframe class="sl-demo" src="" name="sl-embed" allow="fullscreen" allowfullscreen style="display:none;position:absolute;top:0;left:0;width:100%;height:100%;border:none;"></iframe>
        </div>
    `

	return (
		<>
			<Script async src="https://js.storylane.io/js/v2/storylane.js" />
			<motion.div
				initial={{ opacity: 0 }}
				animate={{ opacity: 1 }}
				exit={{ opacity: 0 }}
				className="fixed inset-0 bg-black/70 backdrop-blur-md z-[60] flex items-center justify-center p-4"
				onClick={onClose}
			>
				<motion.div
					initial={{ scale: 0.95, y: 20 }}
					animate={{ scale: 1, y: 0 }}
					exit={{ scale: 0.95, y: -20 }}
					transition={{ duration: 0.2, ease: "easeInOut" }}
					onClick={(e) => e.stopPropagation()}
					className="relative w-full max-w-4xl"
				>
					<div dangerouslySetInnerHTML={{ __html: embedHtml }} />
					<button
						onClick={onClose}
						className="absolute -top-3 -right-3 z-[9999999] p-1.5 bg-neutral-800 text-white rounded-full hover:bg-neutral-700"
						aria-label="Close demo"
					>
						<IconX size={18} />
					</button>
				</motion.div>
			</motion.div>
		</>
	)
}

export default function ChatPage() {
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

	// State for UI enhancements
	const [userDetails, setUserDetails] = useState(null)
	const posthog = usePostHog()
	const [isFocused, setIsFocused] = useState(false)
	const [isWelcomeModalOpen, setIsWelcomeModalOpen] = useState(false)
	const [replyingTo, setReplyingTo] = useState(null)
	const [isOptionsOpen, setIsOptionsOpen] = useState(false)
	const [confirmClear, setConfirmClear] = useState(false)
	const [integrations, setIntegrations] = useState([])
	const [isToolsMenuOpen, setIsToolsMenuOpen] = useState(false)
	const toolsMenuRef = useRef(null)
	const toolsButtonRef = useRef(null)
	const [isDemoModalOpen, setDemoModalOpen] = useState(false)

	const searchParams = useSearchParams()
	const router = useRouter()
	const { isPro } = usePlan()

	// --- File Upload State ---
	const [selectedFile, setSelectedFile] = useState(null)
	const [isUploading, setIsUploading] = useState(false)
	const [uploadedFilename, setUploadedFilename] = useState(null)

	// --- Pro Feature Modal ---
	const [isUpgradeModalOpen, setUpgradeModalOpen] = useState(false)
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
	}, [])

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
		fetchInitialMessages()
		fetchUserDetails()
		return () => {
			if (abortControllerRef.current) {
				abortControllerRef.current.abort()
			}
		}
	}, [fetchInitialMessages, fetchUserDetails])

	useEffect(() => {
		if (searchParams.get("show_demo") === "true") {
			setDemoModalOpen(true)
		}
	}, [searchParams])

	const handleCloseDemo = () => {
		setDemoModalOpen(false)
		// Clean up the URL to prevent the modal from reappearing on refresh
		// Using replace to avoid adding to browser history
		router.replace("/chat", { scroll: false })
	}

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

	useClickOutside(toolsMenuRef, (event) => {
		if (
			toolsButtonRef.current &&
			!toolsButtonRef.current.contains(event.target)
		) {
			setIsToolsMenuOpen(false)
		}
	})

	const { connectedTools, builtinTools } = useMemo(() => {
		const hiddenTools = [
			"progress_updater",
			"chat_tools",
			"tasks",
			"google_search"
		]
		const connected = integrations.filter(
			(i) =>
				i.connected &&
				(i.auth_type === "oauth" || i.auth_type === "manual")
		)
		const builtin = integrations.filter(
			(i) => i.auth_type === "builtin" && !hiddenTools.includes(i.name)
		)
		return { connectedTools: connected, builtinTools: builtin }
	}, [integrations])

	const fetchOlderMessages = useCallback(async () => {
		if (
			isLoadingOlder ||
			!hasMoreMessages ||
			displayedMessages.length === 0
		)
			return

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
				setDisplayedMessages((prev) => [...olderMessages, ...prev])
				setHasMoreMessages(data.messages.length === 50)

				setTimeout(() => {
					scrollContainer.scrollTop =
						scrollContainer.scrollHeight - oldScrollHeight
				}, 0)
			} else {
				setHasMoreMessages(false)
			}
		} catch (error) {
			toast.error(error.message)
		} finally {
			setIsLoadingOlder(false)
		}
	}, [isLoadingOlder, hasMoreMessages, displayedMessages])

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
			if (error.status === 429) {
				toast.error(
					error.message ||
						"You've reached your daily file upload limit for the free plan.",
					{ id: toastId }
				)
				if (!isPro) {
					setUpgradeModalOpen(true)
				}
			} else {
				toast.error(`Error: ${error.message}`, { id: toastId })
			}
			setSelectedFile(null)
		} finally {
			setIsUploading(false)
		}
	}

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
									// --- CHANGED --- Handle the final 'done' event with parsed data
									if (parsed.done) {
										return {
											...msg,
											content:
												parsed.final_content ||
												msg.content, // Replace content with clean version
											turn_steps:
												parsed.turn_steps ||
												msg.turn_steps // Populate turn_steps
										}
									}
									// For intermediate chunks, just append the token
									return {
										...msg,
										content:
											msg.content + (parsed.token || ""),
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
			} else if (error.status === 429) {
				toast.error(
					error.message ||
						"You've reached a usage limit for today on the free plan."
				)
				if (!isPro) {
					setUpgradeModalOpen(true)
				}
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
						.catch((e) => console.error("Error playing sound:", e))
				}
				// Add a delay to allow ICE connection to stabilize
				setVoiceStatusText("Please wait a moment...")
				setMicrophoneEnabled(false) // Mute mic during stabilization
				setTimeout(() => {
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

					setTimeout(() => {
						if (
							webrtcClientRef.current?.peerConnection
								?.connectionState === "connected"
						) {
							setVoiceStatusText("Listening...")
							setMicrophoneEnabled(true)
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

		setConnectionStatus("connecting")
		setVoiceStatusText("Connecting...")
		try {
			// Step 1: Get the main auth token
			const tokenResponse = await fetch("/api/auth/token")
			if (!tokenResponse.ok) throw new Error("Could not get auth token.")
			const { accessToken } = await tokenResponse.json()

			// Step 2: Use the auth token to get a temporary RTC token
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

			// Step 3: Create and connect WebRTCClient directly
			if (webrtcClientRef.current) {
				webrtcClientRef.current.disconnect()
			}
			const client = new WebRTCClient({
				onConnected: () => handleStatusChange("connected"),
				onDisconnected: () => handleStatusChange("disconnected"),
				onAudioStream: (stream) => {
					if (remoteAudioRef.current) {
						remoteAudioRef.current.srcObject = stream
						remoteAudioRef.current
							.play()
							.catch((e) =>
								console.error("Error playing remote audio:", e)
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
					.catch((e) => console.error("Error playing ringtone:", e))
			}
			await webrtcClientRef.current.connect(
				selectedAudioInputDevice,
				accessToken,
				rtc_token
			)
		} catch (error) {
			if (error.status === 429) {
				toast.error(
					error.message ||
						"You've used all your voice minutes for today on the free plan."
				)
				if (!isPro) {
					setUpgradeModalOpen(true)
				}
			} else {
				toast.error(
					`Failed to connect: ${error.message || "Unknown error"}`
				)
			}
			handleStatusChange("disconnected")
		}
	}

	const initializeVoiceMode = async () => {
		// Check if devices are already loaded to avoid re-prompting
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
			// This is the permission prompt
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
				// Set default device if not already set
				if (!selectedAudioInputDevice) {
					setSelectedAudioInputDevice(audioInputDevices[0].deviceId)
				}
				return true
			} else {
				toast.error("No audio input devices found.")
				return false
			}
		} catch (error) {
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

		// --- ADD POSTHOG EVENT TRACKING & USAGE UPDATE ---
		if (voiceModeStartTimeRef.current) {
			const duration_seconds = Math.round(
				(Date.now() - voiceModeStartTimeRef.current) / 1000
			)
			posthog?.capture("voice_mode_used", { duration_seconds })

			// Send usage update to the server
			fetch("/api/voice/update-usage", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ duration_seconds })
			}).catch((err) =>
				console.error("Failed to update voice usage:", err)
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
		if (!isPro) {
			setUpgradeModalOpen(true)
			return
		}

		if (isVoiceMode) {
			handleStopVoice()
			setIsVoiceMode(false)
		} else {
			// Switching TO voice mode, first get permissions
			const permissionsGranted = await initializeVoiceMode()
			if (permissionsGranted) {
				// --- ADD POSTHOG EVENT TRACKING ---
				posthog?.capture("voice_mode_activated")
				voiceModeStartTimeRef.current = Date.now() // Set start time
				// --- END POSTHOG EVENT TRACKING ---
				setIsVoiceMode(true)
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

	const renderReplyPreview = () => (
		<AnimatePresence>
			{replyingTo && (
				<motion.div
					initial={{ opacity: 0, y: 10 }}
					animate={{ opacity: 1, y: 0 }}
					exit={{ opacity: 0, y: 10 }}
					className="bg-neutral-800/60 p-3 rounded-t-lg border-b border-neutral-700/50 flex justify-between items-center"
				>
					<div>
						<p className="text-xs text-neutral-400 flex items-center gap-1.5">
							<IconArrowBackUp size={14} /> Replying to{" "}
							{replyingTo.role === "user"
								? "yourself"
								: "the assistant"}
						</p>
						<p className="text-sm text-neutral-200 mt-1 truncate">
							{replyingTo.content.replace(/<[^>]+>/g, "").trim()}
						</p>
					</div>
					<button
						onClick={() => setReplyingTo(null)}
						className="p-1.5 rounded-full text-neutral-400 hover:bg-neutral-700 hover:text-white"
					>
						<IconX size={16} />
					</button>
				</motion.div>
			)}
		</AnimatePresence>
	)

	const renderUploadedFilePreview = () => (
		<AnimatePresence>
			{uploadedFilename && (
				<motion.div
					initial={{ opacity: 0, y: 10 }}
					animate={{ opacity: 1, y: 0 }}
					exit={{ opacity: 0, y: 10 }}
					className="bg-neutral-800/60 p-3 rounded-t-lg border-b border-neutral-700/50 flex justify-between items-center"
				>
					<div className="flex items-center gap-2 overflow-hidden">
						<IconFile
							size={16}
							className="text-neutral-400 flex-shrink-0"
						/>
						<p
							className="text-sm text-neutral-200 truncate"
							title={uploadedFilename}
						>
							{uploadedFilename}
						</p>
					</div>
					<button
						onClick={() => {
							setUploadedFilename(null)
							setSelectedFile(null)
						}}
						className="p-1.5 rounded-full text-neutral-400 hover:bg-neutral-700 hover:text-white"
					>
						<IconX size={16} />
					</button>
				</motion.div>
			)}
		</AnimatePresence>
	)

	const renderInputArea = () => (
		<div className="relative bg-neutral-800/60 backdrop-blur-sm border border-neutral-700/50 rounded-2xl">
			<div className="relative p-4 flex items-start gap-4">
				<textarea
					ref={textareaRef}
					value={input}
					onChange={handleInputChange}
					onFocus={() => setIsFocused(true)}
					onBlur={() => setIsFocused(false)}
					onKeyDown={(e) => {
						if (e.key === "Enter" && !e.shiftKey) {
							e.preventDefault()
							sendMessage()
						}
					}}
					placeholder=" "
					className="w-full bg-transparent text-base text-white placeholder-transparent resize-none focus:ring-0 focus:outline-none overflow-y-auto custom-scrollbar z-10"
					rows={1}
					style={{ maxHeight: "200px" }}
				/>
				{!input && !uploadedFilename && (
					<div className="absolute top-1/2 left-4 right-4 -translate-y-1/2 text-neutral-500 pointer-events-none z-0 overflow-hidden">
						<TextLoop className="text-base ml-5 whitespace-normal md:whitespace-nowrap">
							<span>Ask anything...</span>
							<span>Summarize my unread emails from today</span>
							<span>
								Draft a follow-up to the project proposal
							</span>
							<span>Schedule a meeting with the design team</span>
						</TextLoop>
					</div>
				)}
			</div>
			<div className="flex justify-between items-center px-3 pb-3">
				<div className="flex items-center gap-1">
					<input
						type="file"
						ref={fileInputRef}
						onChange={handleFileChange}
						className="hidden"
						accept=".csv,.doc,.docx,.eml,.epub,.gif,.jpg,.jpeg,.json,.html,.htm,.msg,.odt,.pdf,.png,.pptx,.ps,.rtf,.tiff,.tif,.txt,.xlsx,.xls"
					/>
					<button
						onClick={() => fileInputRef.current?.click()}
						disabled={isUploading}
						className="p-2 rounded-full text-neutral-300 hover:bg-neutral-700 hover:text-white transition-colors disabled:opacity-50"
						data-tooltip-id="home-tooltip"
						data-tooltip-content="Attach File (Max 5MB)"
					>
						{isUploading ? (
							<IconLoader size={20} className="animate-spin" />
						) : (
							<IconPaperclip size={20} />
						)}
					</button>
					<button
						ref={toolsButtonRef}
						onClick={() => setIsToolsMenuOpen((prev) => !prev)}
						className="p-2 rounded-full text-neutral-300 hover:bg-neutral-700 hover:text-white transition-colors"
						data-tooltip-id="home-tooltip"
						data-tooltip-content="View Available Tools"
					>
						<IconTool size={20} />
					</button>
					<button
						onClick={() => setIsWelcomeModalOpen(true)}
						className="p-2 rounded-full text-neutral-300 hover:bg-neutral-700 hover:text-white transition-colors"
						data-tooltip-id="home-tooltip"
						data-tooltip-content="About Chat"
					>
						<IconInfoCircle size={20} />
					</button>
				</div>
				<div className="flex items-center gap-2">
					<button
						onClick={toggleVoiceMode}
						className="p-2.5 rounded-full text-white bg-neutral-700 hover:bg-neutral-600 transition-colors"
						data-tooltip-id="home-tooltip"
						data-tooltip-content={
							isPro
								? "Switch to Voice Mode"
								: "Voice Mode (Pro Feature)"
						}
					>
						<IconWaveSine size={18} />
					</button>
					{thinking ? (
						<button
							onClick={handleStopStreaming}
							className="p-2.5 rounded-full text-white bg-red-600 hover:bg-red-500"
							data-tooltip-id="home-tooltip"
							data-tooltip-content="Stop Generation"
						>
							<IconPlayerStopFilled size={18} />
						</button>
					) : (
						<button
							onClick={sendMessage}
							disabled={
								(!input.trim() && !uploadedFilename) ||
								thinking ||
								isUploading
							}
							className="p-2.5 bg-brand-orange rounded-full text-white disabled:opacity-50 hover:bg-brand-orange/90 transition-all shadow-md"
						>
							<IconSend size={18} />
						</button>
					)}
				</div>
			</div>
		</div>
	)

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
							<button
								onClick={() => setIsWelcomeModalOpen(false)}
								className="p-1.5 rounded-full hover:bg-neutral-700"
							>
								<IconX size={18} />
							</button>
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
									<IconTools
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
							<button
								onClick={() => setIsWelcomeModalOpen(false)}
								className="py-2 px-5 rounded-lg bg-neutral-700 hover:bg-neutral-600 text-sm font-medium"
							>
								Got it
							</button>
						</footer>
					</motion.div>
				</motion.div>
			)}
		</AnimatePresence>
	)

	const renderToolsMenu = () => (
		<AnimatePresence>
			{isToolsMenuOpen && (
				<motion.div
					ref={toolsMenuRef}
					initial={{ opacity: 0, y: 10, scale: 0.95 }}
					animate={{ opacity: 1, y: 0, scale: 1 }}
					exit={{ opacity: 0, y: 10, scale: 0.95 }}
					transition={{ duration: 0.2, ease: "easeInOut" }}
					className="absolute bottom-full mb-2 w-full max-w-sm bg-neutral-900/90 backdrop-blur-md border border-neutral-700 rounded-xl shadow-lg p-3 z-50"
				>
					<div className="max-h-72 overflow-y-auto custom-scrollbar pr-2">
						{connectedTools.length > 0 && (
							<div className="mb-3">
								<p className="text-xs text-neutral-400 font-semibold mb-2 px-2">
									Connected Apps
								</p>
								<div className="space-y-1">
									{connectedTools.map((tool) => {
										const Icon =
											toolIcons[tool.name] ||
											toolIcons.default
										return (
											<div
												key={tool.name}
												className="flex items-center gap-3 p-2 rounded-md"
											>
												<Icon className="w-5 h-5 text-neutral-300 flex-shrink-0" />
												<span className="text-sm text-neutral-200 font-medium">
													{tool.display_name}
												</span>
											</div>
										)
									})}
								</div>
							</div>
						)}
						{builtinTools.length > 0 && (
							<div>
								<p className="text-xs text-neutral-400 font-semibold mb-2 px-2">
									Built-in Tools
								</p>
								<div className="space-y-1">
									{builtinTools.map((tool) => {
										const Icon =
											toolIcons[tool.name] ||
											toolIcons.default
										return (
											<div
												key={tool.name}
												className="flex items-center gap-3 p-2 rounded-md"
											>
												<Icon className="w-5 h-5 text-neutral-300 flex-shrink-0" />
												<span className="text-sm text-neutral-200 font-medium">
													{tool.display_name}
												</span>
											</div>
										)
									})}
								</div>
							</div>
						)}
					</div>
				</motion.div>
			)}
		</AnimatePresence>
	)

	const renderOptionsMenu = () => (
		<div className="absolute top-4 right-4 md:top-6 md:right-6 z-30">
			<div className="relative">
				<button
					onClick={() => {
						setIsOptionsOpen(!isOptionsOpen)
						setConfirmClear(false) // Reset confirmation on toggle
					}}
					className="p-2 rounded-full bg-neutral-800/50 hover:bg-neutral-700/80 text-white"
				>
					<IconDotsVertical size={20} />
				</button>
				<AnimatePresence>
					{isOptionsOpen && (
						<motion.div
							initial={{ opacity: 0, y: 10, scale: 0.95 }}
							animate={{ opacity: 1, y: 0, scale: 1 }}
							exit={{ opacity: 0, y: 10, scale: 0.95 }}
							className="absolute top-full right-0 mt-2 w-48 bg-neutral-900/80 backdrop-blur-md border border-neutral-700 rounded-lg shadow-lg p-1"
						>
							<button
								onClick={() => {
									if (confirmClear) {
										handleClearAllMessages()
									} else {
										setConfirmClear(true)
									}
								}}
								className={cn(
									"w-full text-left px-3 py-2 text-sm rounded-md transition-colors",
									confirmClear
										? "bg-red-600/80 text-white hover:bg-red-500"
										: "text-neutral-300 hover:bg-neutral-700/50"
								)}
							>
								{confirmClear ? "Confirm Clear?" : "Clear Chat"}
							</button>
						</motion.div>
					)}
				</AnimatePresence>
			</div>
		</div>
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
			<UpgradeToProModal
				isOpen={isUpgradeModalOpen}
				onClose={() => setUpgradeModalOpen(false)}
			></UpgradeToProModal>
			<AnimatePresence>
				{isDemoModalOpen && (
					<StorylaneDemoModal onClose={handleCloseDemo} />
				)}
			</AnimatePresence>
			{renderWelcomeModal()}
			<audio ref={remoteAudioRef} autoPlay playsInline />
			{displayedMessages.length > 0 &&
				!isVoiceMode &&
				renderOptionsMenu()}
			<div className="flex-1 flex flex-col overflow-hidden relative w-full pt-16 md:pt-0">
				<div className="absolute inset-0 z-[-1] network-grid-background">
					<InteractiveNetworkBackground />
				</div>
				<div className="absolute -top-[250px] left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-brand-orange/10 rounded-full blur-3xl -z-10" />

				<main
					ref={scrollContainerRef}
					className="flex-1 overflow-y-auto px-4 pb-4 md:p-6 flex flex-col custom-scrollbar"
				>
					{isLoading ? (
						<div className="flex-1 flex justify-center items-center">
							<IconLoader className="animate-spin text-neutral-500" />
						</div>
					) : isVoiceMode ? (
						<div className="flex-1 flex flex-col -translate-y-12 relative overflow-hidden">
							{/* The 3D visualization will render here as a background */}
							<SiriSpheres
								status={connectionStatus}
								audioLevel={audioLevel}
							/>

							{/* Overlay for controls and status text */}
							<div className="absolute inset-0 z-20 flex flex-col translate-y-20 items-center justify-end p-6 pb-12">
								{/* Call Control Bar */}
								<div className="flex items-center justify-center gap-4 p-3 bg-neutral-900/50 backdrop-blur-md rounded-full border border-neutral-700/50 shadow-lg mb-6">
									{/* Mic Selector */}
									<select
										value={selectedAudioInputDevice}
										onChange={(e) =>
											setSelectedAudioInputDevice(
												e.target.value
											)
										}
										className="bg-brand-gray backdrop-blur-sm border border-brand-gray text-brand-white text-sm rounded-full px-4 py-4 focus:outline-none focus:border-brand-orange appearance-none max-w-[150px] truncate shadow-lg"
										title="Select Microphone"
										disabled={
											connectionStatus !== "disconnected"
										}
									>
										{audioInputDevices.length === 0 ? (
											<option value="">
												No mics found
											</option>
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

									{/* Mute Button */}
									<AnimatePresence>
										{connectionStatus === "connected" && (
											<motion.button
												initial={{
													opacity: 0,
													scale: 0
												}}
												animate={{
													opacity: 1,
													scale: 1
												}}
												exit={{ opacity: 0, scale: 0 }}
												onClick={handleToggleMute}
												className={cn(
													"flex h-12 w-12 items-center justify-center rounded-full text-white shadow-lg transition-colors duration-200",
													isMuted
														? "bg-white text-black"
														: "bg-neutral-700 hover:bg-neutral-600"
												)}
												title={
													isMuted ? "Unmute" : "Mute"
												}
											>
												{isMuted ? (
													<IconMicrophoneOff
														size={24}
													/>
												) : (
													<IconMicrophone size={24} />
												)}
											</motion.button>
										)}
									</AnimatePresence>

									{/* Main Call/End Button */}
									{connectionStatus === "disconnected" ? (
										<button
											onClick={handleStartVoice}
											className="flex h-12 w-12 items-center justify-center rounded-full bg-brand-green text-white shadow-lg transition-colors duration-200 hover:bg-brand-green/80"
											title="Start Call"
										>
											<IconPhone size={24} />
										</button>
									) : connectionStatus === "connecting" ? (
										<div className="flex h-12 w-12 items-center justify-center rounded-full bg-brand-yellow text-brand-black shadow-lg">
											<IconLoader
												size={24}
												className="animate-spin"
											/>
										</div>
									) : (
										<button
											onClick={handleStopVoice}
											className="flex h-12 w-12 items-center justify-center rounded-full bg-brand-red text-white shadow-lg transition-colors duration-200 hover:bg-brand-red/80"
											title="Hang Up"
										>
											<IconPhoneOff size={24} />
										</button>
									)}

									{/* Switch to Text Mode Button */}
									<button
										onClick={toggleVoiceMode}
										className="flex h-12 w-12 items-center justify-center rounded-full bg-brand-gray hover:bg-neutral-600 text-white shadow-lg"
										title="Switch to Text Mode"
									>
										<IconMessageOff size={24} />
									</button>
								</div>

								{/* Status and Message Display (below controls) */}
								<div className="text-center space-y-2 max-w-2xl">
									<div className="text-lg font-medium text-gray-300 min-h-[24px]">
										<AnimatePresence mode="wait">
											<motion.div
												key={voiceStatusText}
												initial={{ opacity: 0, y: 10 }}
												animate={{ opacity: 1, y: 0 }}
												exit={{ opacity: 0, y: -10 }}
												transition={{ duration: 0.2 }}
											>
												<TextShimmer className="font-mono text-base">
													{voiceStatusText}
												</TextShimmer>
											</motion.div>
										</AnimatePresence>
									</div>
									<div className="text-2xl font-semibold text-white min-h-[64px]">
										<AnimatePresence mode="wait">
											{displayedMessages
												.filter(
													(m) => m.role === "user"
												)
												.slice(-1)
												.map((msg) => (
													<motion.div
														key={msg.id}
														initial={{
															opacity: 0,
															y: 15
														}}
														animate={{
															opacity: 1,
															y: 0
														}}
														exit={{
															opacity: 0,
															y: -15
														}}
														transition={{
															duration: 0.3
														}}
													>
														{msg.content}
													</motion.div>
												))}
										</AnimatePresence>
									</div>
								</div>
							</div>
						</div>
					) : displayedMessages.length === 0 && !thinking ? (
						<div className="flex-1 flex flex-col justify-center items-center p-4 md:p-6">
							<div className="text-center">
								<h1 className="text-4xl sm:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-neutral-100 to-neutral-400 py-4">
									{getGreeting()},{" "}
									{userDetails?.given_name || "User"}
								</h1>
								<p className="mt-2 text-lg text-neutral-400">
									How can I help you today?
								</p>
							</div>
							<div className="w-full max-w-4xl mx-auto mt-12 ">
								<div className="relative">
									{uploadedFilename
										? renderUploadedFilePreview()
										: renderReplyPreview()}
									{renderToolsMenu()}
									{renderInputArea()}
								</div>
								<div className="mt-12"></div>
							</div>
						</div>
					) : (
						<div className="w-full max-w-4xl mx-auto flex flex-col gap-3 md:gap-4 flex-1">
							{isLoadingOlder && (
								<div className="flex justify-center py-4">
									<IconLoader className="animate-spin text-neutral-500" />
								</div>
							)}
							{displayedMessages.map((msg, i) => (
								<div
									key={msg.id || i}
									className={cn(
										"flex w-full",
										msg.role === "user"
											? "justify-end"
											: "justify-start"
									)}
								>
									<ChatBubble
										role={msg.role}
										content={msg.content}
										tools={msg.tools || []}
										turn_steps={msg.turn_steps || []} // --- CHANGED --- Pass turn_steps instead of old props
										onReply={handleReply}
										message={msg}
										allMessages={displayedMessages}
										isStreaming={
											thinking &&
											i === displayedMessages.length - 1
										}
										onDelete={handleDeleteMessage}
									/>
								</div>
							))}
							<div className="flex w-full justify-start">
								<AnimatePresence>
									{thinking && (
										<motion.div
											initial={{ opacity: 0, y: 10 }}
											animate={{ opacity: 1, y: 0 }}
											exit={{ opacity: 0, y: 10 }}
											className="flex items-center gap-2 p-3 bg-neutral-800/50 backdrop-blur-sm rounded-2xl self-start"
										>
											<TextShimmer
												className="font-mono text-sm"
												duration={1.5}
											>
												{statusText || "Thinking..."}
											</TextShimmer>
										</motion.div>
									)}
								</AnimatePresence>
							</div>
							<div ref={chatEndRef} />
						</div>
					)}
				</main>
				{!isLoading && !isVoiceMode && displayedMessages.length > 0 && (
					<div className="flex-shrink-0 px-4 pt-2 pb-4 sm:px-6 sm:pb-6 bg-transparent">
						<div className="relative w-full max-w-4xl mx-auto">
							{uploadedFilename
								? renderUploadedFilePreview()
								: renderReplyPreview()}
							{renderToolsMenu()}
							{renderInputArea()}
						</div>
					</div>
				)}
			</div>
		</div>
	)
}
