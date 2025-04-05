// agents/EmailDetails.jsx
import { useState } from "react" // Importing useState hook from React for managing component state
import { IconMail, IconCopy, IconCheck } from "@tabler/icons-react" // Importing icons from tabler-icons-react library
import toast from "react-hot-toast" // Importing toast for displaying toast notifications
import React from "react"

/**
 * EmailDetails Component - Displays detailed information of an email and allows copying the email body.
 *
 * This component is used to present email details such as subject, sender, snippet, and body.
 * It also provides a button to copy the email body to the clipboard, enhancing user interaction
 * when dealing with email content within the application.
 *
 * @param {object} props - Component props.
 * @param {object} props.email - An object containing the details of the email to be displayed.
 * @param {string} props.email.subject - The subject of the email.
 * @param {string} props.email.from - The sender's email address or name.
 * @param {string} props.email.snippet - A brief preview of the email content.
 * @param {string} props.email.body - The full body content of the email.
 *
 * @returns {React.ReactNode} - The EmailDetails component UI.
 */
const EmailDetails = ({ email }) => {
	// State to manage the 'copied' status for the copy button, indicating if the email body has been copied.
	const [copied, setCopied] = useState(false) // copied: boolean - true if email body is copied, false otherwise

	/**
	 * Handles the copy action of the email body to the clipboard.
	 *
	 * When the copy button is clicked, this function attempts to write the email body text to the clipboard.
	 * On successful copy, it sets the 'copied' state to true to update the button icon to a checkmark,
	 * and then resets it back to false after a short delay (2 seconds) to revert the icon.
	 * If copying fails, it displays an error toast notification to inform the user.
	 *
	 * @function handleCopy
	 * @returns {void}
	 */
	const handleCopy = () => {
		navigator.clipboard
			.writeText(email.body) // Attempts to write the email body text to the clipboard
			.then(() => {
				setCopied(true) // Set 'copied' state to true on successful copy
				setTimeout(() => setCopied(false), 2000) // Reset 'copied' state to false after 2 seconds
			})
			.catch((err) => toast.error("Copy failed:", err)) // Display error toast if copy operation fails
	}

	return (
		<div className="w-full bg-smokeblack rounded-lg p-4 mb-4 border border-lightblue">
			{/* Header section of the Email Details card */}
			<div className="flex items-center gap-2 mb-4">
				<IconMail className="w-6 h-6 text-lightblue" />
				<h3 className="text-xl font-Poppins font-semibold text-white">
					Email Details
				</h3>
			</div>
			{/* Container for displaying email content */}
			<div className="p-4 bg-matteblack rounded-md border border-gray-700">
				{/* Display email Subject */}
				<div className="mb-2">
					<span className="font-Quicksand text-sm text-gray-400">
						Subject:{" "}
					</span>
					<span className="font-Quicksand text-white">
						{email.subject}
					</span>
				</div>
				{/* Display email Sender */}
				<div className="mb-2">
					<span className="font-Quicksand text-sm text-gray-400">
						From:{" "}
					</span>
					<span className="font-Quicksand text-white">
						{email.from}
					</span>
				</div>
				{/* Display email Snippet */}
				<div className="mb-2">
					<span className="font-Quicksand text-sm text-gray-400">
						Snippet:{" "}
					</span>
					<span className="font-Quicksand text-white">
						{email.snippet}
					</span>
				</div>
				{/* Display email Body */}
				<div className="mb-2">
					<span className="font-Quicksand text-sm text-gray-400">
						Body:{" "}
					</span>
					{/* Email body content, whitespace-pre-wrap to maintain formatting */}
					<p className="font-Quicksand text-white whitespace-pre-wrap">
						{email.body}
					</p>
				</div>
				{/* Copy Email Body Button */}
				<div className="mt-2">
					<button
						onClick={handleCopy} // Call handleCopy function on button click
						className="flex items-center text-lightblue hover:text-white transition-colors"
					>
						{/* Conditional rendering of icon based on 'copied' state */}
						{copied ? (
							<IconCheck className="w-5 h-5 mr-1" /> // Show check icon if copied is true
						) : (
							<IconCopy className="w-5 h-5 mr-1" /> // Show copy icon if copied is false
						)}
						<span className="font-Quicksand text-sm">
							Copy Email Body
						</span>
					</button>
				</div>
			</div>
		</div>
	)
}

export default EmailDetails
