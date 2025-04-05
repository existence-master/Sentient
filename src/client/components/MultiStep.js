"use client"
import { AnimatePresence, motion } from "framer-motion" // Importing AnimatePresence and motion from framer-motion library
import { useState, useEffect } from "react" // Importing React hooks: useState, useEffect
import { clsx } from "clsx" // Importing clsx for conditionally joining class names
import { twMerge } from "tailwind-merge" // Importing twMerge for merging Tailwind CSS class names
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
 * CheckIcon Component - Renders a simple checkmark icon as an SVG.
 *
 * This component is a basic checkmark icon, part of the MultiStepLoader component,
 * used to indicate a completed step in the loading process. It's a visual cue for users
 * to track progress through different loading states.
 *
 * @param {object} props - Component props.
 * @param {string} props.className - Optional CSS class names to apply to the icon SVG.
 * @returns {React.ReactNode} - The CheckIcon component UI.
 */
const CheckIcon = ({ className }) => {
	return (
		<svg
			xmlns="http://www.w3.org/2000/svg" // Specifies the namespace for SVG
			fill="none" // Sets fill attribute to 'none'
			viewBox="0 0 24 24" // Defines the view box for the SVG
			strokeWidth={1.5} // Sets the stroke width
			stroke="currentColor" // Sets the stroke color to the current color
			className={cn("w-6 h-6 ", className)} // Base classes for width and height, plus dynamic classNames
		>
			{/* Path element for the checkmark */}
			<path d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
		</svg>
	)
}

/**
 * CheckFilled Component - Renders a filled checkmark icon as an SVG.
 *
 * This component is a filled version of the checkmark icon, used to visually distinguish
 * the currently active or completed step in the MultiStepLoader component. It provides a more prominent
 * visual indication compared to the outlined CheckIcon.
 *
 * @param {object} props - Component props.
 * @param {string} props.className - Optional CSS class names to apply to the icon SVG.
 * @returns {React.ReactNode} - The CheckFilled component UI.
 */
const CheckFilled = ({ className }) => {
	return (
		<svg
			xmlns="http://www.w3.org/2000/svg" // Specifies the namespace for SVG
			viewBox="0 0 24 24" // Defines the view box for the SVG
			fill="currentColor" // Sets fill to the current color, making it filled
			className={cn("w-6 h-6 ", className)} // Base classes for width and height, plus dynamic classNames
		>
			{/* Path element for the filled checkmark */}
			<path
				fillRule="evenodd" // Sets how interior of the shape is determined
				d="M2.25 12c0-5.385 4.365-9.75 9.75-9.75s9.75 4.365 9.75 9.75-4.365 9.75-9.75 9.75S2.25 17.385 2.25 12Zm13.36-1.814a.75.75 0 1 0-1.22-.872l-3.236 4.53L9.53 12.22a.75.75 0 0 0-1.06 1.06l2.25 2.25a.75.75 0 0 0 1.14-.094l3.75-5.25Z"
				clipRule="evenodd" // Sets how parts of the shape are clipped
			/>
		</svg>
	)
}

/**
 * LoaderCore Component - Core loading animation logic and rendering for MultiStepLoader.
 *
 * This component is responsible for rendering the list of loading states and their corresponding
 * check icons. It animates the opacity and vertical position of each loading state to create a smooth,
 * step-by-step loading indicator. It's used internally by MultiStepLoader to display the loading progress.
 *
 * @param {object} props - Component props.
 * @param {Array<{ text: string }>} props.loadingStates - An array of loading state objects, each with a text description.
 * @param {number} props.value - Current loading state index, determines which steps are marked as completed.
 * @returns {React.ReactNode} - The LoaderCore component UI.
 */
const LoaderCore = ({ loadingStates, value = 0 }) => {
	return (
		<div className="flex relative justify-start w-full mx-auto flex-col mt-40">
			{/* Container for loading states: flex layout, relative positioning, justify start, full width, horizontal auto margins, margin top */}
			{loadingStates.map((loadingState, index) => {
				const distance = Math.abs(index - value) // Calculate distance from current state to determine opacity - distance: number
				const opacity = Math.max(1 - distance * 0.2, 0) // Calculate opacity based on distance, ensuring value between 0 and 1 - opacity: number

				return (
					<motion.div
						key={index} // Key prop for React list rendering
						className={cn("text-left flex gap-2 mb-4")} // Styling for each loading state item: text alignment left, flex layout, gap, margin bottom
						initial={{ opacity: 0, y: -(value * 40) }} // Initial animation state: opacity 0, vertical position offset based on value
						animate={{ opacity: opacity, y: -(value * 40) }} // Animation target state: dynamic opacity, vertical position offset based on value
						transition={{ duration: 0.5 }} // Animation transition duration: 0.5 seconds
					>
						<div>
							{/* Conditional rendering of CheckIcon or CheckFilled based on index and value */}
							{index > value && (
								<CheckIcon className="text-white" /> // CheckIcon for states not yet reached (outlined checkmark)
							)}
							{index <= value && (
								<CheckFilled
									className={cn(
										"text-white", // Base text color white
										value === index &&
											"text-lightblue opacity-100" // Highlight current state with light blue color and full opacity
									)}
								/> // CheckFilled for states reached or current (filled checkmark)
							)}
						</div>
						<span
							className={cn(
								"text-white", // Base text color white
								value === index && "text-lightblue opacity-100" // Highlight current state text with light blue color and full opacity
							)}
						>
							{loadingState.text}{" "}
							{/* Text for the loading state */}
						</span>
					</motion.div>
				)
			})}
		</div>
	)
}

/**
 * MultiStepLoader Component - Displays a multi-step loading animation with checkmarks and text.
 *
 * This component renders a full-screen loading overlay with a step-by-step loader.
 * It animates through a list of loading states, marking each as completed with a checkmark icon
 * and highlighting the current state. It uses LoaderCore for the step-by-step rendering and animation,
 * and AnimatePresence for smooth mounting and unmounting animations of the loader overlay.
 *
 * @param {object} props - Component props.
 * @param {Array<{ text: string }>} props.loadingStates - An array of loading state objects, each with a text description.
 * @param {boolean} props.loading - Boolean to control the visibility and animation of the loader.
 * @param {number} props.duration - Duration for each loading state step in milliseconds, defaults to 2000ms.
 * @param {boolean} props.loop - Boolean to control whether the loader should loop back to the first state after completing all, defaults to true.
 * @returns {React.ReactNode} - The MultiStepLoader component UI.
 */
export const MultiStepLoader = ({
	loadingStates, // Array of loading state objects - loadingStates: { text: string }[]
	loading, // Boolean to control loading animation - loading: boolean
	duration = 2000, // Duration for each loading state, default 2000ms - duration: number
	loop = true // Boolean for loader loop, default true - loop: boolean
}) => {
	// State to manage current loading state index, starts at 0 - currentState: number
	const [currentState, setCurrentState] = useState(0)

	/**
	 * useEffect hook to advance the loading state based on a timer.
	 *
	 * This hook is responsible for advancing the loader to the next state after a определенный duration.
	 * It uses `setTimeout` to increment `currentState` after each duration, creating a step-by-step animation.
	 * If `loop` is true, it loops back to the first state after the last one; otherwise, it stops at the last state.
	 * The effect depends on `loading`, `loop`, `loadingStates.length`, and `duration` to control the animation lifecycle.
	 */
	useEffect(() => {
		if (!loading) {
			setCurrentState(0) // Reset currentState to 0 if loading is false
			return // Exit effect if not loading
		}
		// Set timeout to advance to the next loading state
		const timeout = setTimeout(() => {
			setCurrentState(
				(prevState) =>
					loop
						? prevState === loadingStates.length - 1
							? 0 // Loop back to the first state if loop is true and current state is the last one
							: prevState + 1 // Increment to the next state
						: Math.min(prevState + 1, loadingStates.length - 1) // Increment state but do not exceed the last state if loop is false
			)
		}, duration) // Timeout duration from props

		// Cleanup function to clear timeout on component unmount or re-render
		return () => clearTimeout(timeout)
	}, [currentState, loading, loop, loadingStates.length, duration]) // Effect dependencies: currentState, loading, loop, loadingStates.length, duration

	return (
		<AnimatePresence mode="wait">
			{/* AnimatePresence to handle mounting and unmounting animations */}
			{loading && (
				<motion.div
					initial={{ opacity: 0 }} // Initial animation state: opacity 0
					animate={{ opacity: 1 }} // Animation target state: opacity 1
					exit={{ opacity: 0 }} // Exit animation state: opacity 0
					className="fixed z-50 flex items-center justify-center backdrop-blur-2xl"
					// Container for loader overlay: fixed positioning to cover viewport, flex layout to center content, backdrop blur effect, high z-index
				>
					<div className="absolute inset-0 bg-black/20 z-40" />
					{/* Semi-transparent black overlay behind the loader content */}
					<div className="relative z-50 h-96 flex w-72 flex-col items-center justify-center">
						{/* Loader content container: relative positioning, z-index above overlay, fixed height and width, flex layout to center content */}
						<LoaderCore
							value={currentState}
							loadingStates={loadingStates}
						/>
						{/* LoaderCore component for step-by-step loading animation, passing currentState and loadingStates props */}
					</div>
				</motion.div>
			)}
		</AnimatePresence>
	)
}
