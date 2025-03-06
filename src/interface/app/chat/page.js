"use client"

import React from "react"
import { useState, useEffect, useRef } from "react"
import ChatBubble from "@components/ChatBubble" // Component for displaying individual chat messages
import Sidebar from "@components/Sidebar" // Sidebar component for navigation and chat management
import {
	IconSend, // Icon for send action
	IconRefresh, // Icon for refresh/clear chat action
	IconPlayerPlayFilled, // Icon for play/restart server action
	IconLoader, // Icon for loading/thinking state
	IconPlus, // Icon for add/create new chat action
	IconUpload // Icon for file upload action
} from "@tabler/icons-react" // Icon library
import * as d3 from "d3" // D3.js library for creating data visualizations, used for flowchart
import { Tooltip } from "react-tooltip" // Tooltip component for providing hints
import toast from "react-hot-toast" // Library for displaying toast notifications
import AiButton from "@components/CreateChatButton" // Custom button component for creating new chats
import { BackgroundGradientAnimation } from "@components/GradientBg" // Component for animated background gradient
import ModalDialog from "@components/ModalDialog" // Modal dialog component for user interactions
import IconGoogleDocs from "@components/icons/IconGoogleDocs" // Icon for Google Docs command
import IconGoogleSheets from "@components/icons/IconGoogleSheets" // Icon for Google Sheets command
import IconGoogleCalendar from "@components/icons/IconGoogleCalendar" // Icon for Google Calendar command
import IconGoogleSlides from "@components/icons/IconGoogleSlides" // Icon for Google Slides command
import IconGoogleMail from "@components/icons/IconGoogleMail" // Icon for Google Mail command
import IconGoogleDrive from "@components/icons/IconGoogleDrive" // Icon for Google Drive command
import ReminderWidget from "@components/agents/Reminder"

/**
 * Chat component for the main chat interface.
 * Handles message display, user input, server communication, and UI interactions.
 *
 * @returns {React.ReactNode} The Chat component UI.
 */
const Chat = () => {
	// State to manage chat messages. Each message is an object with text and sender info.
	const [messages, setMessages] = useState([]) // messages: { id: string, message: string, isUser: boolean, memoryUsed: boolean, agentsUsed: boolean, internetUsed: boolean }[]
	// State to manage the user input text in the chat textarea.
	const [input, setInput] = useState("") // input: string
	// State to store user details, fetched from the backend.
	const [userDetails, setUserDetails] = useState("") // userDetails: any - Structure depends on backend response
	// State to indicate if the AI is currently processing or thinking.
	const [thinking, setThinking] = useState(false) // thinking: boolean
	// State to track the server connection status.
	const [serverStatus, setServerStatus] = useState(true) // serverStatus: boolean
	// State to control the visibility of the sidebar.
	const [isSidebarVisible, setSidebarVisible] = useState(false) // isSidebarVisible: boolean
	// State to control the visibility of the create chat overlay.
	const [showCreateChatOverlay, setShowCreateChatOverlay] = useState(false) // showCreateChatOverlay: boolean
	// State to store the title of the new chat being created.
	const [newChatTitle, setNewChatTitle] = useState("") // newChatTitle: string
	// State to manage the current chat ID. Defaults to 'home' for the initial state.
	const [chatId, setChatId] = useState("home") // chatId: string
	// State to display status messages from the server (e.g., agent status updates).
	const [statusMessage, setStatusMessage] = useState(null) // statusMessage: string | null
	// State to hold data for the flowchart visualization of agent workflows.
	const [flowchartData, setFlowchartData] = useState([]) // flowchartData: { name: string, difficulty: string }[]
	// State to indicate if a flowchart is currently active and should be displayed.
	const [isFlowActive, setIsFlowActive] = useState(false) // isFlowActive: boolean
	// State to control the visibility of the slash command dropdown.
	const [showCommands, setShowCommands] = useState(false) // showCommands: boolean
	// State to hold the filtered slash commands based on user input.
	const [filteredCommands, setFilteredCommands] = useState([]) // filteredCommands: { text: string, value: string, icon: React.ReactNode }[]
	const [currentModel, setCurrentModel] = useState("")
	const [uploadedFiles, setUploadedFiles] = useState([])
	const [showUploadDropdown, setShowUploadDropdown] = useState(false)

	/**
	 * Array of available slash commands.
	 * Each command includes text for display, value to be inserted in input, and an icon.
	 */
	const SLASH_COMMANDS = [
		{
			text: "Send an email",
			value: "Send an email to [recipient_address] about",
			icon: <IconGoogleMail />
		},
		{
			text: "Search my Gmail inbox",
			value: "Search my Gmail inbox for",
			icon: <IconGoogleMail />
		},
		{
			text: "Create a Gmail draft",
			value: "Create a Gmail draft to [recipient_address] about",
			icon: <IconGoogleMail />
		},
		{
			text: "Forward an email",
			value: "Forward the email which says '[your_query]' to [recipient_address]",
			icon: <IconGoogleMail />
		},
		{
			text: "Reply to an email",
			value: "Reply to an email which says '[your_query]' and say '[your_reply]'",
			icon: <IconGoogleMail />
		},
		{
			text: "Delete an email",
			value: "Delete an email from my Gmail inbox which says '[your_query]'",
			icon: <IconGoogleMail />
		},
		{
			text: "Mark email as read",
			value: "Mark an email in my Gmail inbox which says '[your_query]' as read",
			icon: <IconGoogleMail />
		},
		{
			text: "Mark email as unread",
			value: "Mark an email in my Gmail inbox which says '[your_query]' as unread",
			icon: <IconGoogleMail />
		},
		{
			text: "Delete all spam emails",
			value: "Delete all spam emails from my Gmail inbox",
			icon: <IconGoogleMail />
		},
		{
			text: "Search for a file",
			value: "Search for a file in my Google Drive named [file_name]",
			icon: <IconGoogleDrive />
		},
		{
			text: "Upload a file",
			value: "Upload a file from [file_path] to [folder_name] on my Google Drive",
			icon: <IconGoogleDrive />
		},
		{
			text: "Download a file",
			value: "Find a file named [file_name] in my Google Drive and download it",
			icon: <IconGoogleDrive />
		},
		{
			text: "Schedule an event",
			value: "Schedule an event on [date/day] at [time] in my Google Calendar for",
			icon: <IconGoogleCalendar />
		},
		{
			text: "List upcoming events",
			value: "List upcoming events in my Google Calendar",
			icon: <IconGoogleCalendar />
		},
		{
			text: "Search for events",
			value: "Search my Google Calendar for events about",
			icon: <IconGoogleCalendar />
		},
		{
			text: "Create a document",
			value: "Create a Google Docs document about",
			icon: <IconGoogleDocs />
		},
		{
			text: "Create a spreadsheet",
			value: "Create a Google Sheets spreadsheet about",
			icon: <IconGoogleSheets />
		},
		{
			text: "Create a presentation",
			value: "Create a Google Slides presentation about",
			icon: <IconGoogleSlides />
		}
	]

	// useRef for the textarea element to control its height dynamically.
	const textareaRef = useRef(null) // textareaRef: React.RefObject<HTMLTextAreaElement>
	// useRef to scroll to the end of the chat messages container.
	const chatEndRef = useRef(null) // chatEndRef: React.RefObject<HTMLDivElement>
	// useRef to track if event listeners have been added to avoid duplication.
	const eventListenersAdded = useRef(false) // eventListenersAdded: React.RefObject<boolean>
	// useRef to hold the timeout for status message clearing.
	const statusTimeoutRef = useRef(null) // statusTimeoutRef: React.RefObject<number | null>
	// useRef for the dropdown menu for slash commands.
	const dropdownRef = useRef(null) // dropdownRef: React.RefObject<HTMLDivElement>
	const uploadDropdownRef = useRef(null)


	useEffect(() => {
		const fetchUploadedFiles = async () => {
			const response = await window.electron.invoke("get-uploaded-files")
			if (response.status === 200) {
				setUploadedFiles(response.files)
			}
		}
		fetchUploadedFiles()
	}, [])

	/**
	 * useEffect hook to handle closing the slash command dropdown when clicking outside.
	 */
	useEffect(() => {
		/**
		 * Handles click outside the dropdown to close it.
		 * @param {MouseEvent} event - The click event.
		 */
		const handleClickOutside = (event) => {
			if (
				dropdownRef.current &&
				!dropdownRef.current.contains(event.target)
			) {
				setShowCommands(false)
			}
		}
		document.addEventListener("mousedown", handleClickOutside)
		return () => {
			document.removeEventListener("mousedown", handleClickOutside)
		}
	}, [])

	/**
	 * Handles changes in the input textarea.
	 * Updates the input state, adjusts textarea height, and manages slash commands.
	 *
	 * @param {React.ChangeEvent<HTMLTextAreaElement>} e - The change event.
	 */
	const handleInputChange = (e) => {
		const value = e.target.value
		setInput(value)

		// Dynamically adjust textarea height based on content.
		textareaRef.current.style.height = "auto"
		textareaRef.current.style.height = `${Math.min(
			textareaRef.current.scrollHeight,
			200
		)}px`

		// Handle slash commands visibility and filtering.
		if (value.startsWith("/")) {
			setShowCommands(true)
			const searchText = value.slice(1).toLowerCase()
			setFilteredCommands(
				SLASH_COMMANDS.filter((cmd) =>
					cmd.text.toLowerCase().includes(searchText)
				)
			)
		} else {
			setShowCommands(false)
		}
	}

	/**
	 * Handles the selection of a slash command from the dropdown.
	 * Sets the input value to the selected command and closes the dropdown.
	 *
	 * @param {string} command - The selected command value.
	 */
	const handleCommandSelect = (command) => {
		setInput(command)
		setShowCommands(false)
		textareaRef.current.focus() // Focus back to textarea after command selection
	}

	/**
	 * Fetches the chat history for the current chatId from the backend.
	 * Updates the messages state with the fetched history.
	 */
	const fetchChatHistory = async () => {
		try {
			const response = await window.electron?.invoke(
				"fetch-chat-history",
				{
					chatId
				}
			)
			const { messages } = response
			setMessages(messages)
		} catch (error) {
			toast.error("Error fetching chat history.")
		}
	}

	/**
	 * Fetches user details from the backend.
	 * Updates the userDetails state with the fetched information.
	 */
	const fetchUserDetails = async () => {
		try {
			const response = await window.electron?.invoke("get-profile")
			setUserDetails(response)
		} catch (error) {
			toast.error("Error fetching user details.")
		}
	}

	const fetchCurrentModel = async () => {
		try {
			const response = await window.electron?.invoke("get-db-data")
			if (response.status === 200 && response.data) {
				const selectedModel = response.data.selectedModel
				setCurrentModel(selectedModel || "llama3.2:3b")
			} else {
				setCurrentModel("llama3.2:3b") // Fallback if no data
			}
		} catch (error) {
			toast.error("Error fetching user data.")
			setCurrentModel("llama3.2:3b") // Fallback on error
		}
	}

	/**
	 * Initializes the chat session with the backend for the given chatId.
	 */
	const initializeChat = async () => {
		try {
			const response = await window.electron?.invoke("set-chat", {
				chatId
			})
			if (response.status !== 200) {
				toast.error("Failed to initialize chat.")
			}
		} catch (error) {
			toast.error("Error initializing chat.")
		}
	}

	/**
	 * useEffect hook to handle chatId changes from URL parameters.
	 * Extracts chatId from the URL query parameters on component mount.
	 */
	useEffect(() => {
		const queryParams = new URLSearchParams(window.location.search)
		const queryChatId = queryParams.get("chatId")

		if (queryChatId) {
			setChatId(queryChatId)
		}
	}, [])

	/**
	 * useEffect hook to initialize chat and fetch history when chatId changes.
	 * Calls initializeChat and fetchChatHistory when chatId is updated (and not 'home').
	 */
	useEffect(() => {
		if (chatId && chatId !== "home") {
			initializeChat()
			fetchChatHistory()
		}
	}, [chatId])

	/**
	 * useEffect hook to fetch user details on component mount.
	 */
	useEffect(() => {
		fetchUserDetails()
		fetchCurrentModel()
	}, [])

	/**
	 * useEffect hook to scroll to the bottom of the chat on new messages.
	 * Scrolls the chat container to the latest message whenever messages state updates.
	 */
	useEffect(() => {
		chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
	}, [messages])

	/**
	 * Sets up IPC (Inter-Process Communication) listeners for backend events.
	 * Handles status updates, flow updates, flow end signals, and message streaming.
	 * Ensures listeners are added only once using eventListenersAdded ref.
	 */
	const setupIpcListeners = () => {
		if (!eventListenersAdded.current) {
			/**
			 * Handles status updates from the backend.
			 * Updates statusMessage state and manages thinking state based on status updates.
			 *
			 * @param {Electron.IpcRendererEvent} _event - The IPC event.
			 * @param {string} status - The status message from the backend.
			 */
			const handleStatusUpdate = (_event, status) => {
				setStatusMessage(status)
				setThinking(false) // Stop thinking as status is received

				// Clear any existing timeout to avoid overlapping status clearings.
				if (statusTimeoutRef.current) {
					clearTimeout(statusTimeoutRef.current)
				}

				// Set a timeout to clear the status message after 5 seconds and resume thinking.
				statusTimeoutRef.current = setTimeout(() => {
					setStatusMessage(null)
					setThinking(true) // Resume thinking after status is shown for a while
				}, 5000)
			}

			/**
			 * Handles flow updates from the backend, indicating agent workflow steps.
			 * Updates flowchartData state to visualize the workflow.
			 *
			 * @param {Electron.IpcRendererEvent} _event - The IPC event.
			 * @param {string} flowContent - The name of the flow step.
			 */
			const handleFlowUpdated = (_event, flowContent) => {
				setIsFlowActive(true)
				setFlowchartData((prev) => {
					// Prevent adding duplicate flow steps
					if (prev.some((flow) => flow.name === flowContent)) {
						return prev
					}
					return [
						...prev,
						{ name: flowContent, difficulty: "Medium" }
					]
				})
			}

			/**
			 * Handles flow end signal from the backend, indicating workflow completion.
			 * Clears flowchartData and sets isFlowActive to false.
			 */
			const handleFlowEnd = () => {
				setIsFlowActive(false)
				setFlowchartData([])
			}

			/**
			 * Handles message streaming from the backend.
			 * Appends received tokens to the correct message in the messages state.
			 *
			 * @param {{ messageId: string, token: string }} messageStream - Object containing message ID and token.
			 */
			const handleMessageStream = (messageStream) => {
				const { messageId, token } = messageStream
				setMessages((prevMessages) => {
					const messageIndex = prevMessages.findIndex(
						(msg) => msg.id === messageId
					)

					// If message ID is not found, create a new message entry
					if (messageIndex === -1) {
						return [
							...prevMessages,
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

					// Update existing message by appending the new token
					return prevMessages.map((msg, index) => {
						if (index === messageIndex) {
							if (msg.message.endsWith(token)) {
								return msg // Avoid redundant tokens if same token is received again
							}
							return { ...msg, message: msg.message + token }
						}
						return msg
					})
				})
			}

			window.electron.onStatusUpdated(handleStatusUpdate)
			window.electron.onFlowUpdated(handleFlowUpdated)
			window.electron.onFlowEnded(handleFlowEnd)
			window.electron.onMessageStream(handleMessageStream)

			eventListenersAdded.current = true // Mark listeners as added

			// Cleanup function to be run when component unmounts or dependencies change
			return () => {
				eventListenersAdded.current = false // Reset flag on unmount/rerender
			}
		}
	}

	/**
	 * Creates a tree diagram using D3.js to visualize the flowchart data.
	 * Renders dots and lines representing steps in the flowchart.
	 *
	 * @param {Array<{name: string, difficulty: string}>} data - Array of flowchart step data.
	 */
	const createTreeDiagram = (data) => {
		const svg = d3.select("#flowchart")
		svg.selectAll("*").remove() // Clear previous diagram

		const margin = 20
		const verticalSpacing = 70
		const dotRadius = 8
		const textOffset = 15

		const colors = {
			Default: "#00B2FE" // Default color for flowchart steps
		}

		data.forEach((d, i) => {
			const x = margin + dotRadius
			const y = i * verticalSpacing + margin + dotRadius

			// Draw dot for each step
			svg.append("circle")
				.attr("cx", x)
				.attr("cy", y)
				.attr("r", dotRadius)
				.attr("fill", colors[d.difficulty] || colors.Default)
				.attr("class", "step-dot")

			// Add text label for each step
			svg.append("text")
				.attr("x", x + textOffset)
				.attr("y", y + dotRadius / 2)
				.attr("text-anchor", "start")
				.attr("alignment-baseline", "middle")
				.text(d.name)
				.attr("fill", "#FFFFFF")
				.attr("font-size", "18px")
				.attr("font-family", "Quicksand, sans-serif")

			// Draw line connecting steps
			if (i < data.length - 1) {
				const nextY = (i + 1) * verticalSpacing + margin + dotRadius

				svg.append("line")
					.attr("x1", x)
					.attr("y1", y + dotRadius)
					.attr("x2", x)
					.attr("y2", nextY - dotRadius)
					.attr("stroke", "#FFFFFF")
					.attr("stroke-width", 2)
					.attr("class", "step-line")
			}
		})

		// Set SVG height based on content
		const totalHeight = data.length * verticalSpacing + margin * 2
		svg.attr("height", totalHeight)
		svg.attr("width", 200) // Fixed width for flowchart
	}

	/**
	 * useEffect hook to create the flowchart diagram when flowchartData or isFlowActive changes.
	 * Renders the flowchart only when isFlowActive is true and there is data to display.
	 */
	useEffect(() => {
		if (isFlowActive && flowchartData.length > 0) {
			createTreeDiagram(flowchartData)
		}
	}, [flowchartData, isFlowActive])

	/**
	 * Sends the user's message to the backend for processing.
	 * Updates the local messages state, input state, and thinking state.
	 * Handles timeouts and error scenarios for message sending.
	 */
	const sendMessage = async () => {
		if (input.trim() === "") return // Prevent sending empty messages

		const newMessage = { message: input, isUser: true }
		setMessages((prev) => [...prev, newMessage]) // Add user message to chat
		setInput("") // Clear input field
		setThinking(true) // Indicate AI is thinking

		setupIpcListeners() // Ensure IPC listeners are set up

		// Set up a timeout promise for handling slow responses from backend
		const timeoutPromise = new Promise((_, reject) => {
			setTimeout(() => reject(new Error("Request timed out")), 300000) // 5 minutes timeout
		})

		try {
			// Race between backend response and timeout
			const response = await Promise.race([
				window.electron?.invoke("send-message", {
					chatId,
					input: newMessage.message
				}),
				timeoutPromise
			])

			// Process successful response
			if (response.status === 200) {
				await fetchChatHistory() // Refresh chat history to get AI response

				setThinking(false) // Stop thinking
				setStatusMessage(null) // Clear any status message

				// Clear status message timeout if it exists
				if (statusTimeoutRef.current) {
					clearTimeout(statusTimeoutRef.current)
					statusTimeoutRef.current = null
				}
			} else {
				toast.error("Failed to send message.")
			}
		} catch (error) {
			// Handle errors, especially timeout errors
			toast.error(
				error.message === "Request timed out"
					? "Message sending timed out. Please try again."
					: "Error sending message."
			)
		} finally {
			setThinking(false) // Ensure thinking is off even on error
			if (eventListenersAdded.current) {
				eventListenersAdded.current = false // Reset listener flag after message attempt
			}
		}
	}

	/**
	 * Clears the chat history for the current chatId.
	 * Invokes backend function to clear history and then refetches it.
	 */
	const clearChatHistory = async () => {
		try {
			await window.electron?.invoke("clear-chat-history", { chatId })
			fetchChatHistory() // Refresh chat history after clearing
		} catch (error) {
			toast.error("Error clearing chat history.")
		}
	}

	/**
	 * Reinitiates the FastAPI server.
	 * Sets serverStatus to false during restart and back to true after completion.
	 */
	const reinitiateServer = async () => {
		try {
			setServerStatus(false) // Indicate server is restarting
			await window.electron?.invoke("reinitiate-fastapi-server", {
				chatId
			})
			toast.success("Server reinitiated successfully")
		} catch (error) {
			toast.error("Error restarting the server.")
		} finally {
			setServerStatus(true) // Indicate server is back online, regardless of success or failure
		}
	}

	/**
	 * Handles the creation of a new chat.
	 * Sends the new chat title to the backend and updates the chatId to the new chat.
	 */
	const handleCreateChat = async () => {
		if (!newChatTitle.trim()) return // Prevent creating chat with empty title

		try {
			const response = await window.electron?.invoke("create-chat", {
				title: newChatTitle
			})
			if (response.status === 200) {
				const { chatId } = response
				setShowCreateChatOverlay(false) // Close the create chat overlay
				setNewChatTitle("") // Clear new chat title input
				setChatId(chatId) // Set current chatId to the new chat ID, triggering chat initialization
			}
		} catch (error) {
			toast.error("Error creating chat.")
		}
	}

	const handleFileUpload = async (event) => {
		const files = Array.from(event.target.files)
		const fileData = await Promise.all(
			files.map(async (file) => ({
				name: file.name,
				data: Buffer.from(await file.arrayBuffer())
			}))
		)

		const response = await window.electron.invoke("upload-files", fileData)
		if (response.status === 200) {
			setUploadedFiles((prev) => [
				...new Set([...prev, ...response.files])
			]) // Avoid duplicates
			toast.success("Files uploaded successfully")
			setShowUploadDropdown(true) // Keep dropdown open after upload
		} else {
			toast.error("Error uploading files")
		}
	}

	// Handle file removal
	const handleRemoveFile = async (fileName) => {
		const response = await window.electron.invoke("delete-file", fileName)
		if (response.status === 200) {
			setUploadedFiles((prev) => prev.filter((file) => file !== fileName))
			toast.success("File removed successfully")
		} else {
			toast.error("Error removing file")
		}
	}

	// Handle clicks outside upload dropdown to close it
	useEffect(() => {
		const handleClickOutside = (event) => {
			if (
				uploadDropdownRef.current &&
				!uploadDropdownRef.current.contains(event.target)
			) {
				setShowUploadDropdown(false)
			}
		}
		document.addEventListener("mousedown", handleClickOutside)
		return () =>
			document.removeEventListener("mousedown", handleClickOutside)
	}, [])

	// Render different UI based on chatId. 'home' indicates the initial screen.
	if (chatId === "home") {
		return (
			<div className="h-screen w-full flex relative bg-matteblack">
				<Sidebar
					userDetails={userDetails}
					isSidebarVisible={isSidebarVisible}
					setSidebarVisible={setSidebarVisible}
					chatId={chatId}
					setChatId={setChatId}
					fromChat={true}
				/>
				<BackgroundGradientAnimation
					containerClassName={
						"absolute flex flex-col justify-center items-center w-full h-full"
					}
				/>
				<div className="w-3/5 flex flex-col font-Poppins items-center h-full relative">
					<div className="flex flex-col items-center justify-center flex-grow">
						<h1
							className="text-8xl text-white font-bold mb-8"
							style={{ zIndex: 10 }}
						>
							what's on your mind?
						</h1>
						<AiButton
							onClick={() => setShowCreateChatOverlay(true)}
						/>
					</div>
					{showCreateChatOverlay && (
						<ModalDialog
							title="Create a New Chat"
							inputPlaceholder="Enter chat title"
							inputValue={newChatTitle}
							onInputChange={setNewChatTitle}
							onCancel={() => setShowCreateChatOverlay(false)}
							onConfirm={handleCreateChat}
							confirmButtonText="Create"
							confirmButtonColor="bg-lightblue"
							confirmButtonBorderColor="border-lightblue"
							confirmButtonIcon={IconPlus}
							showInput={true}
						/>
					)}
					<div className="absolute bottom-16 left-0 right-0 flex flex-col items-center justify-center">
						<p className="text-white text-sm mb-5">
							Widgets are coming here soon! 👇
						</p>
						<div className="flex flex-row gap-8">
							<ReminderWidget />
							{/* Add more widgets here in the future, e.g., <AnotherWidget /> */}
						</div>
					</div>
				</div>
			</div>
		)
	}

	// Loading state when chatId is not yet determined.
	if (!chatId) {
		return (
			<div className="h-screen flex justify-center items-center bg-matteblack">
				<p className="text-white text-lg">Loading...</p>
			</div>
		)
	}

	// Main chat UI when chatId is valid.
	return (
		<div className="h-screen bg-matteblack flex relative">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
				chatId={chatId}
				setChatId={setChatId}
				fromChat={true}
			/>
			<div className="w-4/5 flex flex-col justify-center items-start h-full bg-matteblack">
				<div className="flex w-4/5 items-start z-20 justify-between mb-6">
					<button
						onClick={reinitiateServer}
						className="absolute top-10 right-10 p-3 hover-button rounded-full text-white cursor-pointer"
						data-tooltip-id="server-status"
						data-tooltip-content="Facing problems in chat? Try restarting the server"
					>
						{serverStatus ? (
							<IconPlayerPlayFilled className="w-4 h-4 text-white" />
						) : (
							<IconLoader className="w-4 h-4 text-white animate-spin" />
						)}
					</button>
					<Tooltip
						id="server-status"
						place="right"
						type="dark"
						effect="float"
					/>
				</div>

				<div className="w-4/5 ml-10 z-10 h-full overflow-y-scroll no-scrollbar flex flex-col gap-4">
					<div className="grow overflow-y-auto p-4 bg-matteblack rounded-xl no-scrollbar">
						{messages.length === 0 && (
							<div className="font-Poppins h-full flex flex-col justify-center items-center text-gray-500">
								<p className="text-4xl text-white mb-4">
									Send a message to start a conversation
								</p>
								<p className="text-2xl text-gray-500 mb-6">
									or try a sample message:
								</p>
								<div className="flex gap-2">
									<button
										onClick={() =>
											setInput(
												"Help me find a new hobby based on my interests"
											)
										}
										className="px-4 py-2 hover-button rounded-md text-white focus:outline-hidden cursor-pointer"
									>
										Looking for a new hobby? 👀
									</button>
									<button
										onClick={() =>
											setInput(
												"Schedule a lunch meet for tomorrow at 2PM in my Google Calendar"
											)
										}
										className="px-4 py-2 hover-button rounded-md text-white focus:outline-hidden cursor-pointer"
									>
										Schedule a Calendar event 📆
									</button>
									<button
										onClick={() =>
											setInput("Tell me about Steve Jobs")
										}
										className="px-4 py-2 hover-button rounded-md text-white focus:outline-hidden cursor-pointer"
									>
										Search the net 🔍
									</button>
								</div>
							</div>
						)}
						{messages.length > 0 && (
							<div className="flex flex-col gap-3" key={chatId}>
								{messages.map((msg) => {
									return (
										<div
											key={msg.id}
											className={`flex ${
												msg.isUser
													? "justify-end"
													: "justify-start"
											}`}
										>
											<ChatBubble
												message={msg.message}
												isUser={msg.isUser}
												memoryUsed={msg.memoryUsed}
												agentsUsed={msg.agentsUsed}
												internetUsed={msg.internetUsed}
												{...msg}
											/>
										</div>
									)
								})}
								{statusMessage && (
									<p className="inline-flex h-12 animate-shimmer rounded-md bg-[linear-gradient(110deg,#333,45%,#00B2FE,55%,#333)] bg-[length:200%_100%] text-transparent text-xl bg-clip-text px-6 font-normal transition-colors">
										{statusMessage}
									</p>
								)}
							</div>
						)}
						{isFlowActive && flowchartData.length > 0 && (
							<div className="mt-4">
								<svg id="flowchart" className="w-full" />
							</div>
						)}

						{!isFlowActive && thinking && !statusMessage && (
							<div className="flex items-center gap-2 mt-3 animate-pulse">
								<div className="bg-gray-500 rounded-full h-3 w-3"></div>
								<div className="bg-gray-500 rounded-full h-3 w-3"></div>
								<div className="bg-gray-500 rounded-full h-3 w-3"></div>
							</div>
						)}

						<div ref={chatEndRef} />
					</div>
					<p className="text-gray-400 font-Poppins text-sm">
						You can change now your model from the{" "}
						<a href="/settings" className="underline">
							Settings
						</a>{" "}
						page! Current Model:{" "}
						<span className="text-lightblue">{currentModel}</span>
					</p>

					<div className="relative mb-5 flex flex-row gap-4 w-full px-4 py-1 bg-matteblack border-[1px] border-white rounded-lg z-30">
						<div className="flex items-center justify-center">
						<button
							onClick={() =>
								setShowUploadDropdown(!showUploadDropdown)
							}
							className="p-3 hover-button transition-transform transform scale-100 hover:scale-110 cursor-pointer rounded-full text-white"
							data-tooltip-id="upload-tooltip"
							data-tooltip-content="Upload files"
						>
							<IconUpload className="w-4 h-4 text-white" />
							</button>
						</div>
						<Tooltip
							id="upload-tooltip"
							place="top"
							type="dark"
							effect="float"
						/>
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
							placeholder="Start typing or use / for agents"
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
						{showCommands && (
							<div
								ref={dropdownRef}
								className="absolute bottom-16 left-6 bg-smokeblack text-white rounded-xl shadow-lg w-96 p-2 h-[300px] overflow-y-auto"
							>
								{filteredCommands.length === 0 ? (
									<p className="text-gray-400 px-4 py-2">
										No matching commands
									</p>
								) : (
									filteredCommands.map((cmd, index) => (
										<div
											key={index}
											onClick={() =>
												handleCommandSelect(cmd.value)
											}
											className="flex items-center gap-2 px-4 py-2 cursor-pointer hover:bg-gray-700 hover:text-white rounded-sm"
										>
											{cmd.icon}
											<span>{cmd.text}</span>
										</div>
									))
								)}
							</div>
						)}
						{showUploadDropdown && (
							<div
								ref={uploadDropdownRef}
								className="absolute bottom-12 left-0 bg-smokeblack text-white rounded-xl shadow-lg w-64 p-2 max-h-60 overflow-y-auto"
							>
								<input
									type="file"
									multiple
									onChange={handleFileUpload}
									className="mb-2 text-sm text-white file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-lightblue file:text-white hover:file:bg-blue-600"
								/>
								{uploadedFiles.length > 0 ? (
									<ul className="space-y-2">
										{uploadedFiles.map((file) => (
											<li
												key={file}
												className="flex items-center justify-between p-1 hover:bg-gray-700 rounded"
											>
												<span className="truncate flex-1">
													{file}
												</span>
												<button
													onClick={() =>
														handleRemoveFile(file)
													}
													className="ml-2 text-red-500 hover:text-red-700"
												>
													Remove
												</button>
											</li>
										))}
									</ul>
								) : (
									<p className="text-gray-400 text-sm">
										No files uploaded
									</p>
								)}
							</div>
						)}
					</div>
				</div>
			</div>
		</div>
	)
}

export default Chat
