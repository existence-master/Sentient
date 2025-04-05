import React from "react"
import { useState } from "react" // Importing useState hook from React for managing component state
import {
	IconClipboard, // Icon for clipboard (copy) action
	IconCheck, // Icon for checkmark (confirmation) action
	IconBrain, // Icon for brain (memory) feature
	IconSettings, // Icon for settings (agents) feature
	IconGlobe, // Icon for globe (internet) feature
	IconLink, // Icon for external link
	IconMail // Icon for mail
} from "@tabler/icons-react" // Importing icons from tabler-icons-react library
import { Tooltip } from "react-tooltip" // Importing Tooltip component for displaying tooltips
import ReactMarkdown from "react-markdown" // Importing ReactMarkdown component for rendering Markdown content
import remarkGfm from "remark-gfm" // Importing remarkGfm plugin for ReactMarkdown to support GitHub Flavored Markdown
import IconGoogleDocs from "./icons/IconGoogleDocs" // Importing custom icon component for Google Docs
import IconGoogleSheets from "./icons/IconGoogleSheets" // Importing custom icon component for Google Sheets
import IconGoogleCalendar from "./icons/IconGoogleCalendar" // Importing custom icon component for Google Calendar
import IconGoogleSlides from "./icons/IconGoogleSlides" // Importing custom icon component for Google Slides
import IconGoogleDrive from "./icons/IconGoogleDrive" // Importing custom icon component for Google Drive
import IconGoogleMail from "./icons/IconGoogleMail" // Importing custom icon component for Google Mail
import GmailSearchResults from "./agents/GmailSearchResults" // Importing GmailSearchResults component
import toast from "react-hot-toast" // Importing toast for displaying toast notifications

/**
 * LinkButton Component - Renders a styled button that opens a link in a new tab.
 *
 * This component is used within chat messages to display URLs in a button format.
 * It automatically detects the type of link (e.g., Google Docs, Gmail, generic link)
 * and displays an appropriate icon and name. Clicking the button opens the link in a new tab.
 *
 * @param {object} props - Component props.
 * @param {string} props.href - The URL to be opened when the button is clicked.
 * @param {React.ReactNode} props.children - The display text for the link, used as fallback name if tool name not found.
 * @returns {React.ReactNode} - The LinkButton component UI.
 */
const LinkButton = ({ href, children }) => {
	// Mapping of domain names to their respective icons and names for tool identification
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
			name: children // Fallback name if specific tool name is not found
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
			name: children // Uses children as name for external mail links
		},
		default: {
			icon: <IconLink size={14} className="mr-1" />,
			name: "Link" // Default name for generic links
		}
	}

	/**
	 * Determines the tool details (icon and name) based on the URL.
	 *
	 * Iterates through the `toolMapping` to find a matching domain in the given URL.
	 * If a match is found, returns the corresponding icon and name.
	 * If no specific domain is matched, defaults to a generic link icon and name.
	 *
	 * @function getToolDetails
	 * @param {string} url - The URL to analyze.
	 * @returns {{ icon: React.ReactNode, name: string }} - An object containing the icon and name for the tool.
	 */
	const getToolDetails = (url) => {
		for (const domain in toolMapping) {
			if (url.includes(domain)) {
				return toolMapping[domain] // Return tool details if domain is found in URL
			} else if (url.match(/^[^@]+@[\w.-]+\.[a-z]{2,}$/i)) {
				return toolMapping["external-mail"] // Return external mail tool details if URL matches email format
			}
		}
		return toolMapping["default"] // Return default tool details for unmatched URLs
	}

	const { icon, name } = getToolDetails(href) // Get tool icon and name based on href

	return (
		<span
			onClick={() => window.open(href, "_blank", "noopener noreferrer")} // Open URL in new tab on click
			className="bg-white text-black border-2 border-black hover:border-lightblue py-1 px-2 rounded-md items-center cursor-pointer inline-flex"
			// Styling for the link button: background, text color, border, padding, rounded corners, cursor, inline-flex display
			style={{
				display: "inline-flex", // Ensure inline-flex for proper alignment
				verticalAlign: "middle", // Vertical alignment to middle
				margin: "0 4px" // Margin for spacing between link buttons
			}}
		>
			{icon} {/* Render the icon determined by getToolDetails */}
			<span>{name}</span>{" "}
			{/* Render the name determined by getToolDetails */}
		</span>
	)
}

/**
 * ChatBubble Component - Displays a single chat message bubble.
 *
 * This component renders a chat message, distinguishing between user and AI messages with different styles.
 * It supports rendering Markdown content, detects and renders URLs as LinkButtons, and provides functionality
 * to copy message text to the clipboard. For AI messages, it also conditionally displays icons indicating
 * if memory, agents, or internet were used in generating the response, along with tooltips for these icons.
 *
 * @param {object} props - Component props.
 * @param {string} props.message - The text content of the chat message, can be Markdown or JSON.
 * @param {boolean} props.isUser - Boolean indicating if the message is from the user or AI.
 * @param {boolean} props.memoryUsed - Boolean indicating if memory was used to generate the response (AI messages only).
 * @param {boolean} props.agentsUsed - Boolean indicating if agents were used (AI messages only).
 * @param {boolean} props.internetUsed - Boolean indicating if internet was used (AI messages only).
 * @returns {React.ReactNode} - The ChatBubble component UI.
 */
const ChatBubble = ({
	message, // Text content of the message, can be Markdown or JSON - message: string
	isUser, // Boolean, true if message is from user, false if from AI - isUser: boolean
	memoryUsed, // Boolean, true if memory was used in response generation (AI only) - memoryUsed: boolean
	agentsUsed, // Boolean, true if agents were used in response generation (AI only) - agentsUsed: boolean
	internetUsed // Boolean, true if internet was used in response generation (AI only) - internetUsed: boolean
}) => {
	// State to manage the 'copied' status for the copy button, indicating if the message text has been copied.
	const [copied, setCopied] = useState(false) // copied: boolean - true if message text is copied, false otherwise

	/**
	 * Handles copying the message text to the clipboard.
	 *
	 * When the copy button is clicked, this function attempts to write the message text to the clipboard.
	 * On success, it sets the 'copied' state to true to update the button icon to a checkmark,
	 * and then resets it back to false after a short delay (2 seconds) to revert the icon.
	 * If copying fails, it displays an error toast notification to inform the user.
	 *
	 * @function handleCopyToClipboard
	 * @returns {void}
	 */
	const handleCopyToClipboard = () => {
		navigator.clipboard
			.writeText(message) // Attempts to write the message text to the clipboard
			.then(() => {
				setCopied(true) // Set 'copied' state to true on successful copy
				setTimeout(() => setCopied(false), 2000) // Reset 'copied' state to false after 2 seconds
			})
			.catch((err) => toast.error(`Failed to copy text: ${err}`)) // Display error toast if copy operation fails
	}

	/**
	 * Renders the content of the chat bubble.
	 *
	 * Attempts to parse the message as JSON to handle structured messages like tool results.
	 * If the message is a tool result for Gmail inbox search, it renders the GmailSearchResults component.
	 * For other message types or if JSON parsing fails, it renders the message as Markdown with
	 * support for LinkButtons for URLs, and conditionally displays icons for memory, agents, and internet usage for AI messages.
	 *
	 * @function renderContent
	 * @returns {React.ReactNode} - The content to be rendered inside the chat bubble.
	 */
	const renderContent = () => {
		try {
			// Attempt to parse the message as JSON to handle structured responses
			let parsedMessage = JSON.parse(message)
			// If parsed message is a string, attempt to parse it again (for double-stringified JSON)
			if (typeof parsedMessage === "string") {
				parsedMessage = JSON.parse(parsedMessage)
			}
			// Check if the parsed message is a tool result for Gmail inbox search
			if (
				parsedMessage.type === "toolResult" &&
				parsedMessage.tool_name === "search_inbox"
			) {
				// Render GmailSearchResults component for inbox search results
				return (
					<GmailSearchResults
						emails={parsedMessage.emails} // Pass emails array to GmailSearchResults
						gmailSearchUrl={parsedMessage.gmail_search_url} // Pass Gmail search URL to GmailSearchResults
					/>
				)
			}
		} catch (e) {
			// JSON parsing failed or message is not a structured tool result, render as Markdown
		}

		// Default rendering for text messages (Markdown)
		return (
			<div
				className={`p-4 rounded-lg ${
					isUser
						? "bg-white text-black text-lg font-semibold self-end max-w-xs md:max-w-md lg:max-w-lg mt-5 mb-5"
						: "bg-transparent text-lg text-white self-start w-full border-b border-t border-lightblue"
					// Conditional styling based on whether the message is from the user or AI
					// User messages: white background, black text, bold font, right-aligned, max-width, margins
					// AI messages: transparent background, white text, left-aligned, full width, top/bottom border, border-lightblue
				} mb-2 relative font-Inter`} // Common styles: margin-bottom, relative positioning, font-Inter
				style={{ wordBreak: "break-word" }} // Style to break words for long text to prevent overflow
			>
				{/* ReactMarkdown component to render message content as Markdown */}
				<ReactMarkdown
					className="prose prose-invert" // Apply prose styling for better readability in inverted color scheme
					remarkPlugins={[remarkGfm]} // Enable GitHub Flavored Markdown plugin
					children={message} // Message content to be rendered as Markdown
					components={{
						a: ({ href, children }) => (
							<LinkButton href={href} children={children} /> // Render URLs as LinkButton components
						)
					}}
				/>

				{/* Conditionally render icons and copy button for AI messages (non-user messages) */}
				{!isUser && (
					<div className="flex justify-start items-center space-x-4 mt-6">
						{/* Memory Used Icon and Tooltip */}
						{memoryUsed && (
							<>
								<span
									data-tooltip-id="memory-used" // Tooltip ID for "Memory Used"
									data-tooltip-content="Memory was used to generate this response" // Tooltip text
									className="flex items-center text-gray-400" // Styling for the icon container
								>
									<IconBrain size={18} />{" "}
									{/* Brain icon to indicate memory usage */}
								</span>
								<Tooltip
									id="memory-used" // Tooltip ID to associate with the span
									place="right" // Position of the tooltip
									type="dark" // Tooltip theme
									effect="float" // Tooltip animation effect
								/>
							</>
						)}
						{/* Agents Used Icon and Tooltip */}
						{agentsUsed && (
							<>
								<span
									data-tooltip-id="agents-used" // Tooltip ID for "Agents Used"
									data-tooltip-content="Agents were used to process this response" // Tooltip text
									className="flex items-center text-gray-400" // Styling for the icon container
								>
									<IconSettings size={18} />{" "}
									{/* Settings icon to indicate agents usage */}
								</span>
								<Tooltip
									id="agents-used" // Tooltip ID to associate with the span
									place="right" // Position of the tooltip
									type="dark" // Tooltip theme
									effect="float" // Tooltip animation effect
								/>
							</>
						)}
						{/* Internet Used Icon and Tooltip */}
						{internetUsed && (
							<>
								<span
									data-tooltip-id="internet-used" // Tooltip ID for "Internet Used"
									data-tooltip-content="Internet was used to gather information for this response" // Tooltip text
									className="flex items-center text-gray-400" // Styling for the icon container
								>
									<IconGlobe size={18} />{" "}
									{/* Globe icon to indicate internet usage */}
								</span>
								<Tooltip
									id="internet-used" // Tooltip ID to associate with the span
									place="right" // Tooltip position
									type="dark" // Tooltip theme
									effect="float" // Tooltip animation type
								/>
							</>
						)}
						{/* Copy to Clipboard Button */}
						<button
							onClick={handleCopyToClipboard} // Call handleCopyToClipboard function on button click
							className="flex items-center text-gray-400 hover:text-green-500 transition-colors"
							// Styling for the copy button: text color, hover text color, transition for color change
						>
							{/* Conditional rendering of icon based on 'copied' state */}
							{copied ? (
								<IconCheck size={18} /> // Show check icon if copied is true
							) : (
								<IconClipboard size={18} /> // Show clipboard icon if copied is false
							)}
						</button>
					</div>
				)}
			</div>
		)
	}

	return (
		<div className={`...`}>
			{/* Render the content of the chat bubble by calling renderContent function */}
			{renderContent()}
		</div>
	)
}

export default ChatBubble
