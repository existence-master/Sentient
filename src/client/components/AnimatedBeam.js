"use client"

import { useEffect, useRef, useState } from "react" // Importing React hooks: useEffect, useRef, useState
import { cn } from "@utils/cn" // Importing the 'cn' utility function for conditional class names
import React from "react"

/**
 * Beam Component - Renders a single animated beam for the background effect.
 *
 * This component creates a visual beam effect that is part of the animated background.
 * It uses CSS animations to simulate a meteor-like beam moving across the screen.
 * The animation duration varies based on the index prop to create a staggered effect.
 *
 * @param {object} props - Component props.
 * @param {number} props.index - Index of the beam, used to determine animation duration.
 * @returns {React.ReactNode} - The Beam component UI.
 */
function Beam({ index }) {
	// Determine if the beam should have a longer duration based on its index
	const flag = index % 8 === 0 // flag: boolean - true if index is a multiple of 8, false otherwise

	return (
		<div
			className={cn("h-full animate-meteor", {
				// Base class 'animate-meteor' and dynamic duration classes
				"[--duration:7s]": flag, // Longer duration for beams where flag is true (every 8th beam)
				"[--duration:11s]": !flag // Shorter duration for other beams
			})}
			style={{
				width: "6px", // Width of the beam
				transform: "translateY(-20%)", // Initial vertical position, slightly above the container
				"--delay": `${index * 0.5}s` // Animation delay based on index for staggered effect
			}}
		>
			<div
				style={{
					clipPath: "polygon(54% 0, 54% 0, 60% 100%, 40% 100%)" // Clip path to shape the beam like a meteor tail
				}}
				className={cn("w-full", {
					// Dynamic height classes based on flag
					"h-8": flag, // Shorter height for beams where flag is true
					"h-12": !flag // Taller height for other beams
				})}
			>
				<div className="h-full w-full bg-linear-to-b from-neutral-50/50 via-neutral-100 via-75% to-neutral-50" />
				{/* Gradient background for the beam, creating a fading effect */}
			</div>
		</div>
	)
}

/**
 * useGridCount Hook - Dynamically calculates the number of grid columns based on container width.
 *
 * This hook uses a useRef to get the current DOM element and calculates how many beams can fit
 * based on the container width and a predefined cell size. It updates the count on resize events
 * to maintain responsiveness.
 *
 * @returns {object} - An object containing:
 *                   - count: The number of grid columns (beams) that fit in the container.
 *                   - containerRef: React ref to attach to the container element.
 */
function useGridCount() {
	const containerRef = useRef(null) // useRef to get the container element
	const [count, setCount] = useState(0) // State to store the number of grid columns - count: number

	useEffect(() => {
		/**
		 * Updates the grid column count based on the container width.
		 *
		 * Calculates the width of the container and determines how many cells of a fixed size can fit.
		 * Sets the count state with this calculated number.
		 * @function updateCount
		 * @returns {void}
		 */
		const updateCount = () => {
			const rect = containerRef.current?.getBoundingClientRect() // Get container dimensions
			if (!rect) {
				return // If containerRef is not yet attached to a DOM element, return early
			}
			const width = rect.width // Container width
			const cellSize = 40 // Fixed cell size for grid calculation
			setCount(Math.ceil(width / cellSize)) // Calculate and set the number of grid columns
		}

		updateCount() // Initial count update on mount

		window.addEventListener("resize", updateCount) // Update count on window resize
		return () => window.removeEventListener("resize", updateCount) // Cleanup listener on unmount
	}, []) // Empty dependency array ensures this effect runs only once on mount and unmount

	return {
		count, // Number of grid columns
		containerRef // Ref to the container element
	}
}

/**
 * Background Component - Renders the animated background with beams.
 *
 * This component sets up the background for the animated beam effect. It uses the useGridCount hook
 * to dynamically determine how many beams to render based on the width of the container.
 * It creates a radial gradient background and renders multiple Beam components to create the animated effect.
 *
 * @returns {React.ReactNode} - The Background component UI.
 */
function Background() {
	const { count, containerRef } = useGridCount() // Get grid count and containerRef from useGridCount hook

	return (
		<div
			ref={containerRef} // Attach containerRef to this div
			className="absolute inset-0 flex h-full w-full flex-row justify-between bg-black z-0"
			// Absolute positioning to cover the entire parent, flex layout for beams, black background, z-index 0
		>
			<div
				style={{
					background:
						"radial-gradient(50% 50% at 50% 50%,#072a39 0%,rgb(7,42,57) 50%,rgba(7,42,57,0) 100%)"
					// Radial gradient for a subtle background shading effect
				}}
				className="absolute inset-0 top-1/2 h-full w-full rounded-full opacity-40"
				// Absolute positioning, rounded full to create a circular gradient, opacity for subtle effect
			/>
			{/* Render beams dynamically based on the count */}
			{Array.from({ length: count }, (_, i) => (
				<div
					key={i}
					className="relative h-full w-px rotate-12 bg-gray-100/10"
				>
					{/* Container for each beam, rotated for diagonal alignment, styled to be subtle */}
					{(1 + i) % 4 === 0 && <Beam index={i + 1} />}
					{/* Render Beam component every 4th iteration to create a spaced out effect */}
				</div>
			))}
		</div>
	)
}

/**
 * AnimatedBeam Component - A container component for the animated beam background effect.
 *
 * This component serves as a wrapper for content that should have the animated beam background.
 * It positions the Background component and ensures that the children are rendered above the background,
 * creating a visually engaging container with an animated backdrop.
 *
 * @param {object} props - Component props.
 * @param {React.ReactNode} props.children - The content to be rendered above the animated background.
 * @param {string} props.className - Optional CSS class names to apply to the container.
 * @returns {React.ReactNode} - The AnimatedBeam component UI, wrapping children with the animated background.
 */
export default function AnimatedBeam({ children, className }) {
	return (
		<div className={cn("relative w-full overflow-hidden z-0", className)}>
			{/* Relative positioning for container, overflow hidden to clip beams, z-index 0, and any additional classNames */}
			<Background />
			{/* Render the Background component to create the beam animation */}
			<div className="relative h-full w-full">{children}</div>
			{/* Container for children, positioned above the background with a relative z-index */}
		</div>
	)
}
