"use client" // Ensure client-side rendering
import React from "react"
import {
	IconLoader,
	IconPlugConnected,
	IconPlugOff,
	IconX
} from "@tabler/icons-react" // Added icons
import ProIcon from "@components/ProIcon" // Keep ProIcon import
import { cn } from "@utils/cn" // Assuming cn utility is available

// MODIFIED: Component signature and props
const AppCard = ({
	logo,
	name,
	description,
	onClick,
	action = "connect", // 'connect', 'disconnect', or 'pro'
	loading = false,
	disabled = false,
	isConnected = false, // ADDED: Explicitly pass connection status
	requiresUrl = false, // ADDED: Does this action need a URL input?
	profileUrl = "", // ADDED: State for URL input
	setProfileUrl, // ADDED: Function to update URL state
	icon: IconComponent // ADDED: Specific icon for the app
}) => {
	// Determine button text and style based on action and loading state
	let buttonText = "Connect"
	let buttonStyle = "bg-lightblue hover:bg-blue-700" // Default connect style
	let showProIcon = false

	if (action === "disconnect") {
		buttonText = loading ? "Disconnecting..." : "Disconnect"
		buttonStyle = "bg-red-600 hover:bg-red-500" // Disconnect style
	} else if (action === "connect") {
		buttonText = loading ? "Connecting..." : "Connect"
		buttonStyle = "bg-lightblue hover:bg-blue-700"
	} else if (action === "pro") {
		buttonText = "Connect" // Show connect text but disable and show Pro icon
		buttonStyle = "bg-neutral-600" // Disabled style for Pro requirement
		showProIcon = true
		disabled = true // Ensure button is disabled if action is 'pro'
	}

	return (
		// MODIFIED: Card Styling - removed gradient border, using theme colors
		<div className="flex flex-col bg-neutral-800 rounded-xl p-5 shadow-md border border-neutral-700/50 h-full transition-shadow hover:shadow-lg">
			{/* Header */}
			<div className="flex items-center gap-4 mb-4">
				{/* Use passed icon or fallback */}
				{IconComponent ? (
					<IconComponent className="w-10 h-10 text-lightblue flex-shrink-0" />
				) : (
					<img
						src={logo}
						alt={`${name} Logo`}
						className="w-10 h-10 flex-shrink-0"
					/> // Fallback to image
				)}
				<h2 className="text-white text-xl font-semibold">{name}</h2>
				{/* Connection Status Indicator */}
				{isConnected ? (
					<span
						title="Connected"
						className="ml-auto p-1.5 bg-green-500/20 text-green-400 rounded-full"
					>
						<IconPlugConnected size={16} />
					</span>
				) : (
					<span
						title="Disconnected"
						className="ml-auto p-1.5 bg-neutral-700 text-neutral-400 rounded-full"
					>
						<IconPlugOff size={16} />
					</span>
				)}
			</div>
			{/* Description */}
			<p className="text-gray-400 text-sm mb-4 flex-grow">
				{description}
			</p>

			{/* Conditionally render URL input */}
			{requiresUrl && action !== "disconnect" && !isConnected && (
				<input
					type="text"
					placeholder={`Enter ${name} Profile URL`}
					value={profileUrl}
					onChange={(e) => setProfileUrl(e.target.value)}
					className="border border-neutral-600 p-2 rounded-md mb-4 w-full text-sm bg-neutral-700 text-white focus:outline-none focus:border-lightblue"
					disabled={loading || disabled || showProIcon} // Disable if loading or pro needed
				/>
			)}

			{/* Action Button */}
			<button
				// MODIFIED: Button Styling - using theme colors, adjusted padding/text size
				className={cn(
					"w-full rounded-md text-white font-medium py-2.5 px-4 text-sm transition-colors duration-150 flex items-center justify-center gap-2",
					disabled || loading
						? "opacity-60 cursor-not-allowed" // Apply disabled styles
						: "cursor-pointer", // Apply cursor pointer only if enabled
					buttonStyle // Apply dynamic background color
				)}
				onClick={onClick}
				disabled={disabled || loading} // Disable based on props
			>
				{loading && <IconLoader size={18} className="animate-spin" />}{" "}
				{/* Show loader when loading */}
				{buttonText}
				{showProIcon && <ProIcon />}{" "}
				{/* Show Pro icon if action is 'pro' */}
			</button>
		</div>
	)
}

export default AppCard
