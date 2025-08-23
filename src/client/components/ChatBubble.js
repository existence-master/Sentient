import React from "react"
import { useState } from "react"
import { cn } from "@utils/cn"
import {
	IconClipboard,
	IconCheck,
	IconBrain,
	IconLink,
	IconMail,
	IconTrash,
	IconChevronDown,
	IconChevronUp,
	IconArrowBackUp,
	IconTool,
	IconFileText,
	IconWorldSearch,
	IconMapPin,
	IconShoppingCart,
	IconChartPie,
	IconBrandTrello,
	IconNews,
	IconListCheck,
	IconBrandDiscord,
	IconBrandWhatsapp,
	IconCalendarEvent,
	IconBrandSlack,
	IconBrandNotion,
	IconBrandGithub,
	IconBrandGoogleDrive,
	IconArrowRight
} from "@tabler/icons-react"
import { AnimatePresence, motion } from "framer-motion"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import IconGoogleDocs from "./icons/IconGoogleDocs"
import IconGoogleSheets from "./icons/IconGoogleSheets"
import IconGoogleCalendar from "./icons/IconGoogleCalendar"
import IconGoogleSlides from "./icons/IconGoogleSlides" // This is a custom one, not from tabler
import IconGoogleMail from "./icons/IconGoogleMail"
import IconGoogleDrive from "./icons/IconGoogleMail"
import { Tooltip } from "react-tooltip"
import toast from "react-hot-toast"
import FileCard from "./FileCard"

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
	gshopping: IconShoppingCart,
	quickchart: IconChartPie,
	google_search: IconWorldSearch,
	trello: IconBrandTrello,
	news: IconNews,
	todoist: IconListCheck,
	discord: IconBrandDiscord,
	whatsapp: IconBrandWhatsapp,
	gcalendar_alt: IconCalendarEvent,
	default: IconTool
}
// LinkButton component (no changes needed)
const LinkButton = ({ href, children }) => {
	const toolMapping = {
		"drive.google.com": {
			icon: <IconGoogleDrive size={14} className="mr-1" />,
			name: "Google Drive"
		},
		"mail.google.com": {
			icon: <IconGoogleMail size={14} className="mr-1" />,
			name: "Gmail"
		},
		"gmail.com": {
			icon: <IconGoogleMail size={14} className="mr-1" />,
			name: children
		},
		"docs.google.com/spreadsheets": {
			icon: <IconGoogleSheets />,
			name: "Google Sheets"
		},
		"docs.google.com/presentation": {
			icon: <IconGoogleSlides />,
			name: "Google Slides"
		},
		"calendar.google.com": {
			icon: <IconGoogleCalendar />,
			name: "Google Calendar"
		},
		"docs.google.com": {
			icon: <IconGoogleDocs />,
			name: "Google Docs"
		},
		"external-mail": {
			icon: <IconMail size={14} className="mr-1" />,
			name: children
		},
		default: {
			icon: <IconLink size={14} className="mr-1" />,
			name: "Link"
		}
	}

	const getToolDetails = (url) => {
		for (const domain in toolMapping) {
			if (url.includes(domain)) {
				return toolMapping[domain]
			} else if (url.match(/^[^@]+@[\w.-]+\.[a-z]{2,}$/i)) {
				return toolMapping["external-mail"]
			}
		}
		return toolMapping["default"]
	}

	const { icon, name } = getToolDetails(href)

	return (
		<span
			onClick={() => window.open(href, "_blank", "noopener noreferrer")}
			className="bg-[var(--color-primary-surface)] text-[var(--color-text-primary)] border border-[var(--color-primary-surface-elevated)] hover:border-[var(--color-accent-blue)] py-1 px-2 rounded-md items-center cursor-pointer inline-flex"
			style={{
				display: "inline-flex",
				verticalAlign: "middle"
			}}
		>
			{icon}
			<span>{name}</span>
		</span>
	)
}

// Main ChatBubble component
const ChatBubble = ({
	role,
	content,
	tools = [], // This is for the icons at the bottom, keep it
	turn_steps = [], // --- CHANGED --- Use turn_steps instead of old props
	onReply,
	onDelete,
	message,
	allMessages = [],
	isStreaming = false
}) => {
	const [copied, setCopied] = useState(false)
	const [expandedStates, setExpandedStates] = useState({})
	const [processedContent, setProcessedContent] = useState({
		content: content,
		repliedTo: null
	})
	const isUser = role === "user"

	React.useEffect(() => {
		// This effect handles parsing the reply-to block, which is separate
		// from the main content parsing.
		const replyRegex = /<reply_to id="([^"]+)">[\s\S]*?<\/reply_to>\n*/
		const match = content.match(replyRegex)

		if (match) {
			const repliedToId = match[1]
			const actualContent = content.replace(replyRegex, "")
			const originalMessage = allMessages.find(
				(m) => m.id === repliedToId
			)
			setProcessedContent({
				content: actualContent,
				repliedTo: originalMessage
			})
		} else {
			setProcessedContent({ content: content, repliedTo: null })
		}
	}, [content, allMessages])

	// --- REMOVED --- The large useMemo hook for parsing is no longer needed.

	// --- CHANGED --- Simplified logic based on the new `turn_steps` prop.
	const hasInternalMonologue = turn_steps && turn_steps.length > 0

	// Function to toggle expansion of collapsible sections
	const toggleExpansion = (id) => {
		setExpandedStates((prev) => ({ ...prev, [id]: !prev[id] }))
	}

	// ***************************************************************
	// *** UPDATED LOGIC: Function to render message content       ***
	// ***************************************************************
	const transformLinkUri = (uri) => {
		return uri.startsWith("file:") ? uri : uri // Let ReactMarkdown handle its default security
	}
	const renderedContent = React.useMemo(() => {
		const markdownComponents = {
			a: ({ href, children }) => {
				if (href && href.startsWith("file:")) {
					const filename = href.substring(5)
					return <FileCard filename={filename} />
				}
				return <LinkButton href={href} children={children} />
			}
		}

		// User message rendering is simple and unchanged
		if (isUser) {
			return (
				<ReactMarkdown
					className="prose prose-invert max-w-none" // Added max-w-none to allow full width
					remarkPlugins={[remarkGfm]}
					children={processedContent.content || ""}
					urlTransform={transformLinkUri}
					components={markdownComponents}
				/>
			)
		}

		// --- CHANGED --- Assistant message rendering now uses the structured `turn_steps` prop.
		return (
			<>
				{hasInternalMonologue && (
					<div className="mb-4 border-l-2 border-yellow-500 pl-3">
						<button
							onClick={() =>
								toggleExpansion("agent_thought_process")
							}
							className="flex w-full items-center justify-start gap-2 text-yellow-400 hover:text-yellow-300 text-sm font-semibold"
						>
							{expandedStates["agent_thought_process"] ? (
								<IconChevronUp size={16} />
							) : (
								<IconChevronDown size={16} />
							)}
							Agent's Thought Process
						</button>
						<AnimatePresence>
							{expandedStates["agent_thought_process"] && (
								<motion.div
									initial={{ height: 0, opacity: 0 }}
									animate={{ height: "auto", opacity: 1 }}
									exit={{ height: 0, opacity: 0 }}
									transition={{
										duration: 0.3,
										ease: "easeInOut"
									}}
									className="overflow-hidden"
								>
									<div className="mt-2 p-3 bg-neutral-800/50 rounded-md space-y-3">
										{turn_steps.map((step, index) => {
											switch (step.type) {
												case "thought":
													return (
														<div
															key={index}
															className="flex items-start gap-2"
														>
															<IconBrain
																size={16}
																className="text-yellow-400/80 flex-shrink-0 mt-0.5"
															/>
															<ReactMarkdown className="prose prose-sm prose-invert text-neutral-300 whitespace-pre-wrap">
																{step.content}
															</ReactMarkdown>
														</div>
													)
												case "tool_call":
													let formattedArgs =
														step.arguments
													try {
														const parsed =
															JSON.parse(
																step.arguments
															)
														formattedArgs =
															JSON.stringify(
																parsed,
																null,
																2
															)
													} catch (e) {
														/* not json, leave as is */
													}
													return (
														<div
															key={index}
															className="space-y-1"
														>
															<p className="text-xs font-semibold text-green-400 flex items-center gap-1.5">
																<IconTool
																	size={14}
																/>{" "}
																Tool Call:{" "}
																{step.tool_name}
															</p>
															<pre className="text-xs text-neutral-300 whitespace-pre-wrap font-mono bg-black/30 p-2 rounded-md">
																<code>
																	{
																		formattedArgs
																	}
																</code>
															</pre>
														</div>
													)
												case "tool_result":
													let formattedResult =
														step.result
													try {
														const parsed =
															JSON.parse(
																step.result
															)
														formattedResult =
															JSON.stringify(
																parsed,
																null,
																2
															)
													} catch (e) {
														/* not json, leave as is */
													}
													return (
														<div
															key={index}
															className="space-y-1"
														>
															<p className="text-xs font-semibold text-purple-400 flex items-center gap-1.5">
																<IconArrowRight
																	size={14}
																/>{" "}
																Tool Result:{" "}
																{step.tool_name}
															</p>
															<pre className="text-xs text-neutral-400 whitespace-pre-wrap font-mono bg-black/30 p-2 rounded-md">
																<code>
																	{
																		formattedResult
																	}
																</code>
															</pre>
														</div>
													)
												default:
													return null
											}
										})}
									</div>
								</motion.div>
							)}
						</AnimatePresence>
					</div>
				)}

				{/* Render the final, clean content (which is now just the `content` prop) */}
				{processedContent.content && (
					<div
						className={cn(
							hasInternalMonologue &&
								"mt-4 pt-4 border-t border-neutral-700/50",
							isStreaming && "opacity-70" // Dim the text while streaming to indicate it's not final
						)}
					>
						<ReactMarkdown
							className="prose prose-invert"
							remarkPlugins={[remarkGfm]}
							children={processedContent.content}
							urlTransform={transformLinkUri}
							components={markdownComponents}
						/>
					</div>
				)}
			</>
		)
	}, [
		processedContent.content,
		expandedStates,
		isUser,
		turn_steps,
		isStreaming
	])

	// Function to copy message content to clipboard
	const handleCopyToClipboard = () => {
		const textToCopy = processedContent.content
		if (!textToCopy) {
			toast.error("Nothing to copy.")
			return
		}
		navigator.clipboard
			.writeText(textToCopy)
			.then(() => {
				setCopied(true)
				setTimeout(() => setCopied(false), 2000)
			})
			.catch((err) => toast.error(`Failed to copy text: ${err}`))
	}

	return (
		<div
			className={cn(
				"px-4 py-3 rounded-2xl relative group text-white backdrop-blur-sm",
				"max-w-full md:max-w-[80%]",
				isUser
					? "bg-blue-600/30 border border-blue-500/50 rounded-br-none"
					: "bg-neutral-800/30 border border-neutral-700/50 rounded-bl-none"
			)}
			style={{ wordBreak: "break-word" }}
		>
			{processedContent.repliedTo && (
				<div className="mb-3 p-2 border-l-2 border-neutral-500 bg-black/20 rounded-md">
					<p className="text-xs text-neutral-400 font-semibold">
						Replying to{" "}
						{processedContent.repliedTo.role === "user"
							? "you"
							: "assistant"}
					</p>
					<p className="text-sm text-neutral-300 mt-1 truncate">
						{processedContent.repliedTo.content}
					</p>
				</div>
			)}
			{renderedContent}
			{tools && tools.length > 0 && (
				<div className="flex items-center gap-2 mt-3 text-xs text-neutral-400">
					<IconTool size={14} />
					<div className="flex flex-wrap gap-1.5">
						{tools.map((toolName) => {
							const Icon =
								toolIcons[toolName] || toolIcons.default
							return (
								<div
									key={toolName}
									className="flex items-center gap-1 bg-neutral-800/60 px-2 py-0.5 rounded-full"
								>
									<Icon size={12} />
									<span>{toolName}</span>
								</div>
							)
						})}
					</div>
				</div>
			)}
			<div
				className={cn(
					"flex items-center gap-2 mt-4 transition-opacity",
					"opacity-100 md:opacity-0 group-hover:md:opacity-100",
					isUser ? "justify-end" : "justify-start"
				)}
			>
				<Tooltip
					place={isUser ? "left-start" : "right-start"}
					id="chat-bubble-tooltip"
					style={{ zIndex: 9999 }}
				/>

				{/* Assistant-only buttons */}
				{!isUser && (
					<>
						<button
							onClick={handleCopyToClipboard}
							className="flex items-center p-1.5 rounded-full text-neutral-400 hover:bg-neutral-700 hover:text-white"
							data-tooltip-id="chat-bubble-tooltip"
							data-tooltip-content={
								copied ? "Copied!" : "Copy response"
							}
						>
							{copied ? (
								<IconCheck size={16} />
							) : (
								<IconClipboard size={16} />
							)}
						</button>
						<button
							onClick={() => onReply(message)}
							className="p-1.5 rounded-full text-neutral-400 hover:bg-neutral-700 hover:text-white"
							data-tooltip-id="chat-bubble-tooltip"
							data-tooltip-content="Reply"
						>
							<IconArrowBackUp size={16} />
						</button>
					</>
				)}

				{/* Delete button for both user and assistant */}
				{onDelete && (
					<button
						onClick={() => onDelete(message.id)}
						className="p-1.5 rounded-full text-neutral-400 hover:bg-neutral-700 hover:text-red-400"
						data-tooltip-id="chat-bubble-tooltip"
						data-tooltip-content="Delete"
					>
						<IconTrash size={16} />
					</button>
				)}
			</div>
		</div>
	)
}

export default ChatBubble
