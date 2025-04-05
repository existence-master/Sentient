"use client"
import React from "react"
// MODIFIED: Removed call-related icons
import { IconMessage, IconMicrophone } from "@tabler/icons-react"
import { cn } from "@utils/cn" // Assuming cn utility is available

const TopControlBar = ({
	chatMode,
	// REMOVED: connectionStatus prop (no longer needed here)
	onToggleMode
	// REMOVED: onStartVoice, onStopVoice props
}) => {
	// REMOVED: getCallButtonIcon helper function
	// REMOVED: handleCallButtonClick handler
	// REMOVED: isCallButtonDisabled variable

	return (
		<div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-30">
			<div className="flex items-center space-x-1 bg-neutral-800/80 backdrop-blur-sm rounded-full p-1 shadow-lg">
				{/* Chat Mode Button (remains the same) */}
				<button
					onClick={() => onToggleMode("text")}
					className={cn(
						"px-4 py-2 rounded-full text-sm font-medium transition-colors duration-200 ease-in-out flex items-center gap-2",
						chatMode === "text"
							? "bg-lightblue/80 text-white"
							: "text-gray-400 hover:text-white hover:bg-neutral-700/60"
					)}
					title="Switch to Text Chat"
				>
					<IconMessage className="w-4 h-4" />
					Chat
				</button>

				{/* Voice Mode Button (remains the same) */}
				<button
					onClick={() => onToggleMode("voice")}
					className={cn(
						"px-4 py-2 rounded-full text-sm font-medium transition-colors duration-200 ease-in-out flex items-center gap-2",
						chatMode === "voice"
							? "bg-lightblue/80 text-white"
							: "text-gray-400 hover:text-white hover:bg-neutral-700/60"
					)}
					title="Switch to Voice Mode"
				>
					<IconMicrophone className="w-4 h-4" />
					Voice
				</button>

				{/* REMOVED: Call Control Button section */}
				{/* {chatMode === "voice" && ( ... removed ... )} */}
			</div>
		</div>
	)
}

export default TopControlBar
