"use client"

import {
	IconSend,
	IconPlayerStopFilled,
	IconTool,
	IconInfoCircle,
	IconPaperclip,
	IconLoader,
	IconHeadphonesFilled,
	IconArrowBackUp,
	IconX,
	IconFile
} from "@tabler/icons-react"
import { motion, AnimatePresence } from "framer-motion"
import { TextLoop } from "@components/ui/TextLoop"
import { Button } from "@components/ui/button"
import useClickOutside from "@hooks/useClickOutside"
import { useState, useMemo, useRef } from "react"

const ChatInputArea = ({
	input,
	handleInputChange,
	sendMessage,
	textareaRef,
	uploadedFilename,
	isUploading,
	fileInputRef,
	handleFileChange,
	integrations,
	toolIcons,
	setIsWelcomeModalOpen,
	toggleVoiceMode,
	thinking,
	handleStopStreaming,
	replyingTo,
	setReplyingTo,
	setUploadedFilename,
	setSelectedFile
}) => {
	const [isToolsMenuOpen, setIsToolsMenuOpen] = useState(false)
	const toolsMenuRef = useRef(null)
	const toolsButtonRef = useRef(null)

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

	const renderReplyPreview = () => (
		<AnimatePresence>
			{replyingTo && (
				<motion.div
					initial={{ opacity: 0, y: 10 }}
					animate={{ opacity: 1, y: 0 }}
					exit={{ opacity: 0, y: 10 }}
					className="bg-neutral-800/60 p-3 rounded-t-lg border-b border-neutral-700/50 flex justify-between items-center gap-4"
				>
					<div className="min-w-0 flex-1">
						<p className="text-xs text-neutral-400 flex items-center gap-1.5">
							<IconArrowBackUp size={14} /> Replying to{" "}
							{replyingTo.role === "user"
								? "yourself"
								: "the assistant"}
						</p>
						<p
							className="text-sm text-neutral-200 mt-1 truncate max-w-[320px] sm:max-w-[480px] md:max-w-[600px] overflow-hidden"
							title={replyingTo.content
								.replace(/<[^>]+>/g, "")
								.trim()}
						>
							{(() => {
								const clean = replyingTo.content
									.replace(/<[^>]+>/g, "")
									.trim()
								return clean.length > 120
									? clean.slice(0, 120) + "…"
									: clean
							})()}
						</p>
					</div>
					<button
						onClick={() => setReplyingTo(null)}
						className="p-1.5 rounded-full text-neutral-400 hover:bg-neutral-700 hover:text-white flex-shrink-0"
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

	return (
		<div className="relative">
			{uploadedFilename
				? renderUploadedFilePreview()
				: renderReplyPreview()}
			{renderToolsMenu()}
			<div
				data-tour-id="chat-input-area"
				className="relative bg-neutral-800/60 backdrop-blur-sm border border-neutral-700/50 rounded-2xl"
			>
				<div className="relative p-3 md:p-4 flex items-start gap-3 md:gap-4">
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
						placeholder=" "
						className="w-full bg-transparent text-base text-white placeholder-transparent resize-none focus:ring-0 focus:outline-none overflow-y-auto custom-scrollbar z-10"
						rows={1}
						style={{ maxHeight: "200px" }}
					/>
					{!input && !uploadedFilename && (
						<div className="absolute top-1/2 left-3 right-3 md:left-4 md:right-4 -translate-y-1/2 text-neutral-500 pointer-events-none z-0 overflow-hidden">
							<TextLoop className="text-base ml-4 md:ml-5 whitespace-nowrap">
								<span>Ask anything...</span>
								<span>
									Summarize my unread emails from today
								</span>
								<span>
									Draft a follow-up to the project proposal
								</span>
								<span>
									Schedule a meeting with the design team
								</span>
							</TextLoop>
						</div>
					)}
				</div>
				<div className="flex justify-between items-center px-2 pb-2 md:px-3 md:pb-3">
					<div className="flex items-center gap-1">
						<input
							type="file"
							ref={fileInputRef}
							onChange={handleFileChange}
							className="hidden"
							accept=".csv,.doc,.docx,.eml,.epub,.gif,.jpg,.jpeg,.json,.html,.htm,.msg,.odt,.pdf,.png,.pptx,.ps,.rtf,.tiff,.tif,.txt,.xlsx,.xls"
						/>
						<Button
							onClick={() => fileInputRef.current?.click()}
							disabled={isUploading}
							variant="ghost"
							size="icon"
							className="rounded-full p-2 h-auto disabled:opacity-50"
							data-tooltip-id="home-tooltip"
							data-tooltip-content="Attach File (Max 5MB)"
						>
							{isUploading ? (
								<IconLoader
									size={20}
									className="animate-spin"
								/>
							) : (
								<IconPaperclip size={20} />
							)}
						</Button>
						<Button
							ref={toolsButtonRef}
							onClick={() => setIsToolsMenuOpen((prev) => !prev)}
							variant="ghost"
							size="icon"
							className="rounded-full p-2 h-auto"
							data-tooltip-id="home-tooltip"
							data-tooltip-content="View Available Tools"
						>
							<IconTool size={20} />
						</Button>
						<Button
							onClick={() => setIsWelcomeModalOpen(true)}
							variant="ghost"
							size="icon"
							className="rounded-full p-2 h-auto"
							data-tooltip-id="home-tooltip"
							data-tooltip-content="About Chat"
						>
							<IconInfoCircle size={20} />
						</Button>
					</div>
					<div className="flex items-center gap-2">
						<Button
							onClick={toggleVoiceMode}
							variant="secondary"
							size="icon"
							className="rounded-full p-2.5 h-auto bg-neutral-700 hover:bg-neutral-600"
							data-tooltip-id="home-tooltip"
							data-tooltip-content="Switch to Voice Mode"
						>
							<IconHeadphonesFilled size={18} />
						</Button>
						{thinking ? (
							<Button
								onClick={handleStopStreaming}
								variant="destructive"
								size="icon"
								className="rounded-full p-2.5 h-auto"
								data-tooltip-id="home-tooltip"
								data-tooltip-content="Stop Generation"
							>
								<IconPlayerStopFilled size={18} />
							</Button>
						) : (
							<Button
								onClick={sendMessage}
								disabled={
									(!input.trim() && !uploadedFilename) ||
									thinking ||
									isUploading
								}
								size="icon"
								className="rounded-full p-2.5 h-auto bg-brand-orange hover:bg-brand-orange/90"
							>
								<IconSend size={18} />
							</Button>
						)}
					</div>
				</div>
			</div>
		</div>
	)
}

export default ChatInputArea
