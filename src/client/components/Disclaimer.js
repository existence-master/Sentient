import React from "react" // Importing React library

/**
 * Disclaimer Component - Displays a disclaimer modal for data usage consent.
 *
 * This component is a modal that presents a disclaimer to the user regarding the use of their data
 * when connecting or disconnecting social media accounts (like LinkedIn, Reddit, Twitter).
 * It dynamically adjusts its message based on the action (connect or disconnect) and the app name.
 * For connection actions, it also includes an input field for the user to enter their profile URL.
 *
 * @param {object} props - Component props.
 * @param {string} props.appName - The name of the application (e.g., LinkedIn) for context in the disclaimer.
 * @param {string} props.profileUrl - The URL of the social media profile, used for input field value.
 * @param {function} props.setProfileUrl - Function to update the profile URL state.
 * @param {function} props.onAccept - Handler function called when the user accepts the disclaimer.
 * @param {function} props.onDecline - Handler function called when the user declines the disclaimer.
 * @param {string} props.action - Action type, either "connect" or "disconnect", to customize the disclaimer message.
 * @returns {React.ReactNode} - The Disclaimer component UI.
 */
const Disclaimer = ({
	appName, // Name of the application for disclaimer context - appName: string
	profileUrl, // URL of the social media profile, for input value - profileUrl: string
	setProfileUrl, // Function to update profile URL state - setProfileUrl: (url: string) => void
	onAccept, // Handler function for accept action - onAccept: () => void
	onDecline, // Handler function for decline action - onDecline: () => void
	action // Action type ("connect" or "disconnect") to customize message - action: "connect" | "disconnect"
}) => {
	return (
		<div className="absolute inset-0 flex items-center justify-center bg-[#22211D]/75">
			{/* Overlay container: absolute positioning to cover entire viewport, flex layout to center content, background color with opacity */}
			<div className="bg-white p-6 rounded-lg shadow-lg max-w-md text-center">
				{/* Disclaimer modal container: white background, padding, rounded corners, shadow, max width, text-center alignment */}
				<h2 className="text-black text-xl font-bold mb-4">
					Data Usage Disclaimer
				</h2>
				{/* Disclaimer title: black text color, text size xl, bold font, margin bottom */}
				<p className="text-black mb-4">
					{/* Disclaimer message paragraph: black text color, margin bottom */}
					{
						action === "connect"
							? `By connecting your ${appName} account, you agree to let us use your public profile information to enhance your experience. We value your privacy and your data stays local.`
							: // Message for "connect" action: explains data usage for enhancing experience, emphasizes privacy and local data storage
								`By disconnecting your ${appName} account, you understand that your app data will be removed from our services.`
						// Message for "disconnect" action: informs user about data removal upon disconnection
					}
				</p>
				{/* Conditional rendering of profile URL input field, only for "connect" action */}
				{action === "connect" && (
					<input
						type="text"
						placeholder={`Enter ${appName} Profile URL`}
						value={profileUrl} // Input value from profileUrl prop
						onChange={(e) => setProfileUrl(e.target.value)} // Update profileUrl state on input change
						className="border p-2 rounded-sm mb-4 w-full text-black"
						// Input field styling: border, padding, rounded corners, margin bottom, full width, black text color
					/>
				)}
				{/* Accept button */}
				<button
					className="bg-green-500 text-white font-bold py-2 px-4 rounded-full mr-2 cursor-pointer"
					// Accept button styling: green background, white text color, bold font, padding, rounded full corners, margin right
					onClick={onAccept} // Calls the onAccept handler passed as prop
				>
					Accept
				</button>
				{/* Decline button */}
				<button
					className="bg-red-500 text-white font-bold py-2 px-4 rounded-full cursor-pointer"
					// Decline button styling: red background, white text color, bold font, padding, rounded full corners
					onClick={onDecline} // Calls the onDecline handler passed as prop
				>
					Decline
				</button>
			</div>
		</div>
	)
}

export default Disclaimer