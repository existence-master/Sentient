"use client"
import { useCallback, useRef } from "react" // Importing React hooks: useCallback, useRef
import { useMousePosition } from "@hooks/useMousePosition" // Importing custom hook useMousePosition
import clsx from "clsx" // Importing clsx for conditionally joining class names
import { twMerge } from "tailwind-merge" // Importing twMerge for merging Tailwind CSS class names
import { IconArrowLeft } from "@tabler/icons-react" // Importing IconArrowLeft from tabler-icons-react library
import React from "react"

/**
 * Utility function to conditionally merge and apply class names.
 * Combines class names using clsx and then merges Tailwind CSS class names using twMerge.
 *
 * @param {...any} inputs - Class names to be merged and applied.
 * @returns {string} - Merged and optimized class names string.
 */
const cn = (...inputs) => {
	return twMerge(clsx(inputs)) // Merges class names using twMerge after conditional joining with clsx
}

/**
 * ShinyCard Component - A card component with a shiny overlay effect that follows the mouse.
 *
 * This component renders a card with a dynamic shiny overlay that moves based on the mouse position.
 * It's designed for interactive elements like question cards in personality tests, providing visual feedback
 * and a modern UI experience. The component includes a back button, question index, question text,
 * and a scale for option selection.
 *
 * @param {object} props - Component props.
 * @param {string} props.className - Optional CSS class names to apply to the ShinyCard container.
 * @param {string} props.questionText - The text content of the question displayed on the card.
 * @param {number} props.currentQuestionIndex - The index of the current question, for display purposes.
 * @param {number} props.totalQuestions - The total number of questions, for display purposes.
 * @param {function} props.onOptionClick - Handler function to be called when an option button is clicked, expects a value argument.
 * @param {number | null} props.selectedOption - The currently selected option value, used to highlight the selected option.
 * @param {function} props.onBackClick - Handler function to be called when the back button is clicked.
 * @returns {React.ReactNode} - The ShinyCard component UI.
 */
export default function ShinyCard({
	className, // Optional class names for ShinyCard container - className: string
	questionText, // Text content for the question - questionText: string
	currentQuestionIndex, // Index of the current question - currentQuestionIndex: number
	totalQuestions, // Total number of questions - totalQuestions: number
	onOptionClick, // Handler for option button click - onOptionClick: (value: number) => void
	selectedOption, // Currently selected option value - selectedOption: number | null
	onBackClick // Handler for back button click - onBackClick: () => void
}) {
	const containerRef = useRef(null) // useRef to get the container element of the card - containerRef: React.RefObject<HTMLDivElement>
	const overlayRef = useRef(null) // useRef to get the overlay element for shiny effect - overlayRef: React.RefObject<HTMLDivElement>

	/**
	 * useCallback hook to update the shiny overlay position based on mouse coordinates.
	 *
	 * This function is memoized with useCallback to prevent unnecessary re-renders. It calculates the offset
	 * needed to center the shiny overlay under the mouse cursor within the card and sets CSS variables
	 * `--x` and `--y` on the overlay element to dynamically position the gradient.
	 *
	 * @function update
	 * @param {object} position - Object containing mouse x and y coordinates.
	 * @param {number} position.x - Mouse X coordinate relative to the viewport.
	 * @param {number} position.y - Mouse Y coordinate relative to the viewport.
	 * @returns {void}
	 */
	const update = useCallback(({ x, y }) => {
		if (!overlayRef.current) {
			return // If overlayRef is not yet attached to a DOM element, return early
		}

		const { width, height } =
			overlayRef.current?.getBoundingClientRect() ?? {} // Get overlay dimensions
		const xOffset = x - width / 2 // Calculate X offset to center overlay under mouse
		const yOffset = y - height / 2 // Calculate Y offset to center overlay under mouse

		overlayRef.current?.style.setProperty("--x", `${xOffset}px`) // Set CSS variable --x for overlay position
		overlayRef.current?.style.setProperty("--y", `${yOffset}px`) // Set CSS variable --y for overlay position
	}, []) // Empty dependency array as it does not depend on any state or prop

	// useMousePosition hook to track mouse position relative to containerRef and call update function
	useMousePosition(containerRef, update)

	return (
		<div
			ref={containerRef} // Attach containerRef to this div
			className={cn(
				"group relative w-full max-w-4xl overflow-hidden rounded-md border border-border bg-matteblack p-6 text-zinc-200 shadow-lg",
				// Base card styling: group for hover effects, relative positioning, full width, max width, overflow hidden, rounded corners, border, background color, padding, text color, shadow, dynamic className
				className // Dynamic classNames for ShinyCard container
			)}
		>
			{/* Shiny overlay div */}
			<div
				ref={overlayRef} // Attach overlayRef to this div
				className="absolute h-64 w-64 rounded-full bg-white opacity-0 bg-blend-soft-light blur-3xl transition-opacity group-hover:opacity-20"
				// Shiny overlay styling: absolute positioning, fixed height and width, rounded full for circle shape, white background, initial opacity 0, blend mode, blur effect, opacity transition on group hover
				style={{
					zIndex: "1", // z-index to position overlay behind content
					transform: "translate(var(--x), var(--y))" // Dynamic translation based on CSS variables --x and --y
				}}
			/>

			{/* Back button */}
			<div className="absolute top-4 left-4" style={{ zIndex: "5" }}>
				{/* Container for back button, absolute positioning top-left, higher z-index */}
				<button
					onClick={onBackClick} // Calls onBackClick handler on button click
					className="flex items-center text-gray-300 hover:text-white transition-colors"
					// Back button styling: flex layout, items-center for vertical alignment, gray-300 text color, hover text color white, transition for color change
					style={{ zIndex: "5" }} // Inline style to ensure higher z-index
				>
					<IconArrowLeft className="w-6 h-6 mr-2" />{" "}
					{/* Arrow left icon, size and margin right */}
					<span className="text-sm">Back</span>{" "}
					{/* Back button text, small text size */}
				</button>
			</div>

			{/* Question index display */}
			<div className="mb-6 mt-6" style={{ zIndex: "5" }}>
				{/* Container for question index, margin bottom and top, higher z-index */}
				<span className="text-gray-400 text-sm">{`Question ${
					currentQuestionIndex + 1 // Display current question index (1-based)
				} of ${totalQuestions}`}</span>{" "}
				{/* Display question index and total questions count */}
			</div>

			{/* Question text */}
			<h2 className="text-white text-4xl font-bold mb-8 text-center">
				{questionText} {/* Render question text */}
			</h2>

			{/* Options scale container */}
			<div
				className="flex items-center justify-between mb-6"
				style={{ zIndex: "5" }}
			>
				{/* Container for options scale, flex layout, items-center for vertical alignment, justify-between for even spacing, margin bottom, higher z-index */}
				<span className="text-gray-300 text-base">Disagree</span>
				{/* "Disagree" label for the left side of the scale */}
				<div className="flex gap-6" style={{ zIndex: "5" }}>
					{/* Options buttons container, flex layout, horizontal gap between buttons, higher z-index */}
					{Array.from({ length: 7 }, (_, i) => (
						// Map over an array of length 7 to create 7 option buttons
						<button
							key={i} // Key prop for React list rendering
							onClick={() => {
								onOptionClick(i - 3) // Call onOptionClick handler with value based on index
							}}
							className={`${
								i === 0 || i === 6
									? "w-16 h-16" // Larger size for крайние buttons (index 0 and 6)
									: i === 3
										? "w-10 h-10" // Smallest size for middle button (index 3)
										: i === 1 || i === 5
											? "w-14 h-14" // Medium size for buttons next to крайние (index 1 and 5)
											: "w-12 h-12" // Default size for other buttons (index 2 and 4)
							} rounded-full border-2 ${
								selectedOption === i - 3
									? "bg-blue-500 border-blue-300" // Styling for selected option: blue background and border
									: "bg-gray-700 border-gray-600 hover:border-lightblue" // Styling for non-selected option: gray background and border, hover border lightblue
							} shadow-md hover:scale-105 transition-transform`}
							// Option button styling: dynamic width and height based on index, rounded full corners, border, dynamic background and border color based on selectedOption, shadow, hover scale effect, transition for transform
							style={{ zIndex: "5" }} // Inline style to ensure higher z-index
						></button>
					))}
				</div>
				<span className="text-gray-300 text-base">Agree</span>
				{/* "Agree" label for the right side of the scale */}
			</div>
		</div>
	)
}
