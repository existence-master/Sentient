"use client"

import React from "react"
import { useMotionValue, useMotionTemplate, motion } from "framer-motion" // Importing motion-related hooks and components from framer-motion library
import { useState } from "react" // Importing useState hook from React for managing component state

/**
 * EvervaultButton Component - A button with a dynamic radial gradient mask effect on hover.
 *
 * This component creates a button that, on mouse hover, applies a radial gradient mask that dynamically
 * follows the mouse cursor, creating a spotlight effect. It utilizes framer-motion hooks for animation
 * and state management for hover interactions.
 *
 * @param {object} props - Component props.
 * @param {string} props.text - The text to be displayed on the button.
 * @param {function} props.onClick - Handler function to be called when the button is clicked.
 * @param {string} props.className - Optional CSS class names to apply to the button container.
 * @returns {React.ReactNode} - The EvervaultButton component UI.
 */
const EvervaultButton = ({ text, onClick, className }) => {
	// useMotionValue to track mouse X position for dynamic styling - mouseX: MotionValue<number>
	let mouseX = useMotionValue(0)
	// useMotionValue to track mouse Y position for dynamic styling - mouseY: MotionValue<number>
	let mouseY = useMotionValue(0)
	// State to track hover status, controlling mask effect - isHovered: boolean
	const [isHovered, setIsHovered] = useState(false) // Initially set to false

	/**
	 * Handles mouse move events to update mouseX and mouseY motion values.
	 *
	 * This function is called on mousemove events within the button. It calculates the mouse position
	 * relative to the button and updates the `mouseX` and `mouseY` motion values, which in turn
	 * drive the dynamic radial gradient effect.
	 *
	 * @function onMouseMove
	 * @param {MouseEvent} event - The mousemove event object.
	 * @param {HTMLElement} event.currentTarget - The button element that the event listener is attached to.
	 * @param {number} event.clientX - The horizontal coordinate of the mouse pointer, relative to the entire document.
	 * @param {number} event.clientY - The vertical coordinate of the mouse pointer, relative to the entire document.
	 * @returns {void}
	 */
	function onMouseMove({ currentTarget, clientX, clientY }) {
		const { left, top } = currentTarget.getBoundingClientRect() // Get button's bounding rectangle
		mouseX.set(clientX - left) // Set mouseX motion value to mouse position relative to button's left edge
		mouseY.set(clientY - top) // Set mouseY motion value to mouse position relative to button's top edge
	}

	/**
	 * Handles mouse leave event to reset hover state and mouse motion values.
	 *
	 * This function is called on mouseleave events from the button. It sets `isHovered` to false
	 * and resets both `mouseX` and `mouseY` motion values to 0, which effectively removes the
	 * radial gradient mask effect.
	 *
	 * @function onMouseLeave
	 * @returns {void}
	 */
	function onMouseLeave() {
		setIsHovered(false) // Set isHovered state to false, disabling hover effect
		mouseX.set(0) // Reset mouseX motion value to 0
		mouseY.set(0) // Reset mouseY motion value to 0
	}

	/**
	 * Handles mouse enter event to set hover state.
	 *
	 * This function is called on mouseenter events on the button. It sets the `isHovered` state to true,
	 * which enables the radial gradient mask effect on the button.
	 *
	 * @function onMouseEnter
	 * @returns {void}
	 */
	function onMouseEnter() {
		setIsHovered(true) // Set isHovered state to true, enabling hover effect
	}

	// useMotionTemplate to create dynamic radial gradient mask based on mouseX and mouseY motion values - maskImage: MotionValue<string>
	const maskImage = useMotionTemplate`radial-gradient(250px at ${mouseX}px ${mouseY}px, white, transparent)`
	// Style object to apply dynamic maskImage - style: object
	const style = {
		maskImage, // Apply maskImage for standard CSS 'mask-image' property
		WebkitMaskImage: maskImage // Apply maskImage for Webkit browsers using '-webkit-mask-image' property
	}

	return (
		<div className={`relative ${className}`}>
			{/* Container div with relative positioning and dynamic classNames */}
			<motion.button
				onMouseEnter={onMouseEnter} // Call onMouseEnter handler on mouse enter
				onMouseMove={onMouseMove} // Call onMouseMove handler on mouse move
				onMouseLeave={onMouseLeave} // Call onMouseLeave handler on mouse leave
				onClick={onClick} // Call onClick handler on button click
				className={`relative flex items-center justify-center py-2 px-6 rounded-lg font-bold text-white border-2 text-lg transition-transform hover:bg-linear-to-r hover:from-lightblue hover:to-darkblue`}
				// Button styling: relative positioning, flex layout, center alignment, padding, rounded corners, bold font, white text color, border, text size, transition, hover gradient background
				style={isHovered ? style : ""} // Apply dynamic style (mask) only when isHovered is true
			>
				<span className="z-10">{text}</span>
				{/* Button text with z-index to ensure it's above the blur effect */}
				<div
					className={`absolute inset-0 bg-white/[0.2] rounded-lg blur-xl group-hover:opacity-50`}
					// Blur effect div: absolute positioning to cover button, semi-transparent white background, rounded corners, blur effect, opacity transition on group hover
				/>
			</motion.button>
		</div>
	)
}

export default EvervaultButton
