// REMOVED: ShiningButton import
import React from "react"
import { IconX, IconLoader } from "@tabler/icons-react" // Added icons
import { cn } from "@utils/cn" // Import cn utility

const ModalDialog = ({
	title,
	description = "",
	inputPlaceholder = "",
	inputValue = "",
	onInputChange,
	onCancel,
	onConfirm,
	confirmButtonText = "Confirm",
	// REMOVED: Color props for ShiningButton
	// confirmButtonColor = "bg-lightblue",
	// confirmButtonBorderColor = "border-lightblue",
	// MODIFIED: Added button type prop and loading state
	confirmButtonType = "primary", // 'primary', 'danger', 'success'
	confirmButtonLoading = false,
	confirmButtonIcon: ConfirmIcon, // Rename for clarity
	showInput = false,
	extraContent = null,
	cancelButton = true
}) => {
	// ADDED: Determine button style based on type
	const getConfirmButtonStyle = () => {
		switch (confirmButtonType) {
			case "danger":
				return "bg-red-600 hover:bg-red-500"
			case "success":
				return "bg-green-600 hover:bg-green-500"
			case "primary":
			default:
				return "bg-lightblue hover:bg-blue-700"
		}
	}

	return (
		// MODIFIED: Overlay and Dialog styling
		<div className="fixed inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm z-50 p-4">
			<div className="bg-neutral-800 rounded-lg p-6 shadow-xl space-y-5 w-full max-w-md border border-neutral-700">
				{" "}
				{/* Adjusted width/padding */}
				<div className="flex justify-between items-center">
					<h3 className="text-lg font-semibold text-white">
						{title}
					</h3>{" "}
					{/* Adjusted size */}
					{cancelButton && ( // Show close button only if cancel is possible
						<button
							onClick={onCancel}
							className="text-gray-400 hover:text-white"
						>
							{" "}
							<IconX size={22} />{" "}
						</button>
					)}
				</div>
				{description && (
					<p className="text-gray-300 text-sm">{description}</p>
				)}
				{showInput && (
					<input
						type="text"
						placeholder={inputPlaceholder}
						value={inputValue}
						onChange={(e) => onInputChange(e.target.value)}
						// MODIFIED: Input styling
						className="w-full border border-neutral-600 text-white rounded-md p-2.5 bg-neutral-700 focus:outline-none focus:border-lightblue text-sm"
					/>
				)}
				{extraContent && <div className="mt-4">{extraContent}</div>}
				{/* MODIFIED: Button container and button styling */}
				<div
					className={cn(
						"flex items-center gap-3",
						cancelButton ? "justify-end" : "justify-center"
					)}
				>
					{cancelButton && (
						<button
							onClick={onCancel}
							// MODIFIED: Button styling
							className="py-2 px-4 rounded-md bg-neutral-600 hover:bg-neutral-500 text-white text-sm font-medium transition-colors"
						>
							Cancel
						</button>
					)}
					{/* MODIFIED: Replaced ShiningButton with standard button */}
					<button
						onClick={onConfirm}
						disabled={confirmButtonLoading}
						// MODIFIED: Button styling using helper and theme colors
						className={cn(
							"py-2 px-5 rounded-md text-white text-sm font-medium transition-colors flex items-center justify-center gap-2 disabled:opacity-60",
							getConfirmButtonStyle() // Apply dynamic style
						)}
					>
						{confirmButtonLoading && (
							<IconLoader size={16} className="animate-spin" />
						)}
						{ConfirmIcon && !confirmButtonLoading && (
							<ConfirmIcon size={16} />
						)}
						<span>
							{confirmButtonLoading
								? "Processing..."
								: confirmButtonText}
						</span>
					</button>
				</div>
			</div>
		</div>
	)
}

export default ModalDialog
