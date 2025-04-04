import React from "react"
import { IconAlertTriangle, IconX } from "@tabler/icons-react" // Added icons

const Disclaimer = ({
	appName,
	profileUrl,
	setProfileUrl,
	onAccept,
	onDecline,
	action,
	showInput = false // Default showInput to false if not provided
}) => {
	return (
		// MODIFIED: Overlay styling
		<div className="fixed inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm z-50 p-4">
			{/* MODIFIED: Dialog styling */}
			<div className="bg-neutral-800 rounded-lg p-6 shadow-xl w-full max-w-md text-center border border-neutral-700">
				<div className="flex justify-center mb-4">
					<IconAlertTriangle className="w-10 h-10 text-yellow-500" />
				</div>
				<h2 className="text-white text-xl font-semibold mb-3">
					{action === "connect"
						? `Connect ${appName}`
						: `Disconnect ${appName}`}
				</h2>
				<p className="text-gray-300 text-sm mb-5">
					{action === "connect"
						? `By connecting your ${appName} account, you agree to let Sentient access your public profile information solely to enhance your personalized experience. Your data remains stored locally on your device.`
						: `Disconnecting your ${appName} account will remove the associated data from Sentient's local storage and knowledge graph.`}
				</p>
				{/* MODIFIED: Input styling and conditional rendering */}
				{action === "connect" &&
					showInput && ( // Ensure showInput prop controls rendering
						<input
							type="text"
							placeholder={`Enter ${appName} Profile URL`}
							value={profileUrl}
							onChange={(e) => setProfileUrl(e.target.value)}
							className="border border-neutral-600 p-2.5 rounded-md mb-6 w-full text-sm bg-neutral-700 text-white focus:outline-none focus:border-lightblue" // Adjusted style
						/>
					)}
				{/* MODIFIED: Button styling */}
				<div className="flex justify-center gap-4">
					<button
						className="py-2 px-6 rounded-md bg-neutral-600 hover:bg-neutral-500 text-white text-sm font-medium transition-colors"
						onClick={onDecline}
					>
						Cancel
					</button>
					<button
						className={cn(
							"py-2 px-6 rounded-md text-white text-sm font-medium transition-colors",
							action === "connect"
								? "bg-green-600 hover:bg-green-500"
								: "bg-red-600 hover:bg-red-500"
						)}
						onClick={onAccept}
					>
						{action === "connect"
							? "Accept & Connect"
							: "Confirm Disconnect"}
					</button>
				</div>
			</div>
		</div>
	)
}

export default Disclaimer
