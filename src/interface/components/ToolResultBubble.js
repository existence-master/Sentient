// ToolResultBubble.js
import React, { useState } from "react"
import {
	IconClipboard,
	IconCheck,
	IconBrain,
	IconSettings,
	IconGlobe,
	IconLink,
	IconMail
} from "@tabler/icons-react"
import { Tooltip } from "react-tooltip"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import IconGoogleDocs from "./icons/IconGoogleDocs"
import IconGoogleSheets from "./icons/IconGoogleSheets"
import IconGoogleCalendar from "./icons/IconGoogleCalendar"
import IconGoogleSlides from "./icons/IconGoogleSlides"
import IconGoogleDrive from "./icons/IconGoogleDrive"
import IconGoogleMail from "./icons/IconGoogleMail"
import toast from "react-hot-toast"

/**
 * LinkButton Component - Renders a styled button that opens a link in a new tab.
 * (Copied from ChatBubble for ToolResultBubble functionality)
 */
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
			className="bg-white text-black border-2 border-black hover:border-lightblue py-1 px-2 rounded-md items-center cursor-pointer inline-flex"
			style={{
				display: "inline-flex",
				verticalAlign: "middle",
				margin: "0 4px"
			}}
		>
			{icon}
			<span>{name}</span>
		</span>
	)
}

const ToolResultBubble = ({
	task,
	result,
	memoryUsed,
	agentsUsed,
	internetUsed
}) => {
	const [copied, setCopied] = useState(false)

	const handleCopyToClipboard = () => {
		navigator.clipboard
			.writeText(result)
			.then(() => {
				setCopied(true)
				setTimeout(() => setCopied(false), 2000)
			})
			.catch((err) => toast.error(`Failed to copy text: ${err}`))
	}

	return (
		<div className="w-fit max-w-[80%] bg-gray-800 text-white p-4 rounded-lg border border-lightblue font-Inter relative">
			{task && (
				<h4 className="font-semibold mb-2 text-lightblue">
					Update on '{task}'
				</h4>
			)}
			<p className="text-sm font-semibold">Result:</p>
			<div style={{ wordBreak: "break-word" }}>
				<ReactMarkdown
					className="prose prose-invert text-sm"
					remarkPlugins={[remarkGfm]}
					children={result}
					components={{
						a: ({ href, children }) => (
							<LinkButton href={href} children={children} />
						)
					}}
				/>
			</div>

			<div className="flex justify-start items-center space-x-4 mt-4">
				{memoryUsed && (
					<>
						<span
							data-tooltip-id="memory-used-tool-result"
							data-tooltip-content="Memory was used to generate this response"
							className="flex items-center text-gray-400"
						>
							<IconBrain size={18} />
						</span>
						<Tooltip
							id="memory-used-tool-result"
							place="right"
							type="dark"
							effect="float"
						/>
					</>
				)}
				{agentsUsed && (
					<>
						<span
							data-tooltip-id="agents-used-tool-result"
							data-tooltip-content="Agents were used to process this response"
							className="flex items-center text-gray-400"
						>
							<IconSettings size={18} />
						</span>
						<Tooltip
							id="agents-used-tool-result"
							place="right"
							type="dark"
							effect="float"
						/>
					</>
				)}
				{internetUsed && (
					<>
						<span
							data-tooltip-id="internet-used-tool-result"
							data-tooltip-content="Internet was used to gather information for this response"
							className="flex items-center text-gray-400"
						>
							<IconGlobe size={18} />
						</span>
						<Tooltip
							id="internet-used-tool-result"
							place="right"
							type="dark"
							effect="float"
						/>
					</>
				)}
				<button
					onClick={handleCopyToClipboard}
					className="flex items-center text-gray-400 hover:text-green-500 transition-colors"
				>
					{copied ? (
						<IconCheck size={18} />
					) : (
						<IconClipboard size={18} />
					)}
				</button>
			</div>
		</div>
	)
}

export default ToolResultBubble