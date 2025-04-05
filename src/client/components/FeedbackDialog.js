import { useState } from "react" // Importing useState hook from React for managing component state
import React from "react"

/**
 * FeedbackDialog Component - A modal dialog for users to provide feedback.
 *
 * This component renders a modal dialog that prompts users to provide their preferred answer or feedback
 * in a textarea. It includes a submit button to send the feedback and a cancel button to close the dialog.
 * The dialog is conditionally rendered based on the `isOpen` prop.
 *
 * @param {object} props - Component props.
 * @param {boolean} props.isOpen - Boolean to control the visibility of the dialog.
 * @param {function} props.onClose - Handler function to close the dialog.
 * @param {function} props.onSubmit - Handler function to submit the feedback, takes the feedback text as argument.
 * @returns {React.ReactNode} - The FeedbackDialog component UI. Returns null if `isOpen` is false.
 */
const FeedbackDialog = ({ isOpen, onClose, onSubmit }) => {
	// State to store the preferred answer text input by the user - preferredAnswer: string
	const [preferredAnswer, setPreferredAnswer] = useState("")

	// If isOpen is false, Dialog is not rendered - Conditional rendering based on isOpen prop
	if (!isOpen) return null

	/**
	 * Handles the submission of the feedback.
	 *
	 * This function is called when the Submit button is clicked. It calls the `onSubmit` handler
	 * passed as prop, sending the current `preferredAnswer` text as feedback. After submission,
	 * it clears the `preferredAnswer` state and closes the dialog by calling `onClose`.
	 *
	 * @function handleSubmit
	 * @returns {void}
	 */
	const handleSubmit = () => {
		onSubmit(preferredAnswer) // Call onSubmit handler with the preferredAnswer
		setPreferredAnswer("") // Clear the preferredAnswer state
		onClose() // Close the FeedbackDialog
	}

	return (
		<div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
			{/* Fixed overlay container: covers the entire viewport, semi-transparent black background, flex layout to center dialog, high z-index */}
			<div className="bg-gray-700 rounded-lg p-6 w-1/3">
				{/* Dialog container: gray background, rounded corners, padding, one-third of viewport width */}
				<h2 className="text-lg font-semibold mb-4">
					Provide your preferred answer
				</h2>
				{/* Dialog title: text size lg, semi-bold font weight, margin bottom */}
				<textarea
					value={preferredAnswer} // Input value from preferredAnswer state
					onChange={(e) => setPreferredAnswer(e.target.value)} // Update preferredAnswer state on input change
					className="w-full p-2 border border-gray-400 bg-gray-700 rounded-sm"
					// Textarea styling: full width, padding, border, gray-400 border color, gray background, rounded corners
					placeholder="Enter your answer here" // Placeholder text for textarea
					rows="4" // Sets the initial height to 4 rows of text
				/>
				{/* Button container: flex layout to align buttons to the end, margin top */}
				<div className="flex justify-end mt-4">
					{/* Cancel button */}
					<button
						className="px-4 py-2 mr-2 bg-gray-600 border border-gray-400 rounded-sm"
						// Cancel button styling: padding, margin right, gray background, border, rounded corners
						onClick={onClose} // Calls the onClose handler to close the dialog
					>
						Cancel
					</button>
					{/* Submit button */}
					<button
						className="px-4 py-2 gradient-bg text-white rounded-sm"
						// Submit button styling: padding, gradient background (defined elsewhere), white text color, rounded corners
						onClick={handleSubmit} // Calls the handleSubmit function on button click
					>
						Submit
					</button>
				</div>
			</div>
		</div>
	)
}

export default FeedbackDialog
