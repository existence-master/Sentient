"use client"
import { useCallback, useRef, useState } from "react" // Importing React hooks: useCallback, useRef, useState
import React from "react"

/**
 * RippleButton Component - A button component with a ripple effect on click and hover.
 *
 * This component renders a button with a material design-like ripple effect that expands from the point of click or hover.
 * The ripple effect is implemented using CSS animations and is triggered by mouse events.
 *
 * @param {object} props - Component props.
 * @param {React.ReactNode} props.children - Content to be rendered inside the RippleButton (e.g., button text).
 * @param {object} props.otherProps - Any other props to be passed to the button element.
 * @returns {React.ReactNode} - The RippleButton component UI.
 */
export default function RippleButton({ children, ...props }) {
	const buttonRef = useRef(null) // useRef to get the button element - buttonRef: React.RefObject<HTMLButtonElement>
	const rippleRef = useRef(null) // useRef to get the ripple span element - rippleRef: React.RefObject<HTMLSpanElement>
	// State to track hover status, mainly used to prevent ripple effect on initial hover - isHovered: boolean
	const [isHovered, setIsHovered] = useState(false) // Initially set to false

	/**
	 * Creates and animates the ripple effect on button click or mouse enter.
	 *
	 * This useCallback function is responsible for generating the ripple effect. It calculates the size and position
	 * of the ripple based on the button's dimensions and the event's client coordinates. It then applies CSS classes
	 * to the ripple span to trigger the enter animation.
	 *
	 * @function createRipple
	 * @param {MouseEvent} event - The mouse event that triggered the ripple (e.g., click, mouseenter).
	 * @returns {void}
	 */
	const createRipple = useCallback(
		(event) => {
			if (isHovered || !buttonRef.current || !rippleRef.current) return // Do not create ripple if already hovered or refs are missing
			setIsHovered(true) // Set isHovered to true to indicate ripple effect is active

			const button = buttonRef.current // Get button element from ref
			const ripple = rippleRef.current // Get ripple element from ref
			const rect = button.getBoundingClientRect() // Get button's bounding rectangle
			const size = Math.max(rect.width, rect.height) * 2 // Calculate ripple size to cover button
			const x = event.clientX - rect.left - size / 2 // Calculate ripple starting X position
			const y = event.clientY - rect.top - size / 2 // Calculate ripple starting Y position

			ripple.style.width = `${size}px` // Set ripple width
			ripple.style.height = `${size}px` // Set ripple height
			ripple.style.left = `${x}px` // Set ripple left position
			ripple.style.top = `${y}px` // Set ripple top position

			ripple.classList.remove("ripple-leave") // Remove leave animation class
			ripple.classList.add("ripple-enter") // Add enter animation class to start ripple animation
		},
		[isHovered] // Dependency array: effect depends on isHovered state
	)

	/**
	 * Removes the ripple effect and triggers the leave animation.
	 *
	 * This useCallback function handles the removal of the ripple effect, typically on mouse leave.
	 * It triggers the ripple-leave CSS animation and sets up an event listener to clean up after the animation ends,
	 * removing the ripple-leave class and the animationend event listener itself.
	 *
	 * @function removeRipple
	 * @param {MouseEvent} event - The mouse event that triggered ripple removal (e.g., mouseleave).
	 * @returns {void}
	 */
	const removeRipple = useCallback(
		(event) => {
			if (event.target !== event.currentTarget) return // Only proceed if the event target is the button itself
			if (!buttonRef.current || !rippleRef.current) return // Do not remove ripple if refs are missing
			setIsHovered(false) // Set isHovered to false as mouse has left button area

			const ripple = rippleRef.current // Get ripple element from ref
			const rect = buttonRef.current.getBoundingClientRect() // Get button's bounding rectangle
			const size = Math.max(rect.width, rect.height) * 2 // Calculate ripple size
			const x = event.clientX - rect.left - size / 2 // Calculate ripple starting X position
			const y = event.clientY - rect.top - size / 2 // Calculate ripple starting Y position

			ripple.style.left = `${x}px` // Set ripple left position (no movement, just for consistent animation start)
			ripple.style.top = `${y}px` // Set ripple top position (no movement, just for consistent animation start)

			ripple.classList.remove("ripple-enter") // Remove enter animation class
			ripple.classList.add("ripple-leave") // Add leave animation class to start ripple fade out animation

			/**
			 * Handles the animation end event for ripple-leave animation.
			 *
			 * This function is called when the ripple-leave animation ends. It removes the ripple-leave class
			 * and removes itself as an event listener to prevent memory leaks and ensure proper event handling cleanup.
			 * @function handleAnimationEnd
			 * @returns {void}
			 */
			const handleAnimationEnd = () => {
				if (ripple) {
					ripple.classList.remove("ripple-leave") // Remove ripple-leave class after animation ends
					ripple.removeEventListener(
						"animationend",
						handleAnimationEnd
					) // Remove event listener to cleanup
				}
			}

			ripple.addEventListener("animationend", handleAnimationEnd) // Add event listener for animation end to handle cleanup
		},
		[] // Empty dependency array as it does not depend on any state or prop
	)

	/**
	 * Handles mouse move events to reposition the ripple effect during hover.
	 *
	 * This useCallback function is called on mousemove events within the button when hovered.
	 * It updates the ripple's position dynamically based on the current mouse coordinates,
	 * making the ripple effect follow the mouse cursor as it moves over the button.
	 *
	 * @function handleMouseMove
	 * @param {MouseEvent} event - The mousemove event object.
	 * @returns {void}
	 */
	const handleMouseMove = useCallback(
		(event) => {
			if (!buttonRef.current || !rippleRef.current || !isHovered) return // Do not handle mouse move if not hovered or refs missing

			const button = buttonRef.current // Get button element from ref
			const ripple = rippleRef.current // Get ripple element from ref
			const rect = button.getBoundingClientRect() // Get button's bounding rectangle
			const size = Math.max(rect.width, rect.height) * 2 // Calculate ripple size
			const x = event.clientX - rect.left - size / 2 // Calculate ripple starting X position
			const y = event.clientY - rect.top - size / 2 // Calculate ripple starting Y position

			ripple.style.left = `${x}px` // Update ripple left position to follow mouse
			ripple.style.top = `${y}px` // Update ripple top position to follow mouse
		},
		[isHovered] // Dependency array: effect depends on isHovered state
	)

	return (
		<button
			ref={buttonRef} // Attach buttonRef to the button element
			className="font-jost duration-600ms relative flex items-center justify-center overflow-hidden rounded-full bg-[#cbfe7e] p-[1.3rem] text-[1.2rem] font-medium text-[#0e352e] transition hover:text-white"
			// Button styling: font, transition duration, relative positioning, flex layout, overflow hidden, rounded corners, background color, padding, text size, font weight, text color, transition for hover, hover text color
			onMouseEnter={(e) => {
				if (e.target === e.currentTarget) {
					// Check if the event target is the button itself
					createRipple(e) // Call createRipple function on mouse enter
				}
			}}
			onMouseLeave={(e) => {
				if (e.target === e.currentTarget) {
					// Check if the event target is the button itself
					removeRipple(e) // Call removeRipple function on mouse leave
				}
			}}
			onMouseMove={handleMouseMove} // Call handleMouseMove function on mouse move
			{...props} // Spread any additional props to the button element
		>
			<span className="relative z-2">{children}</span>
			{/* Span for button text, relative positioning and z-index to be above ripple */}
			<span ref={rippleRef} className="ripple" />
			{/* Span for ripple effect, ref attached, class 'ripple' for base styling, animations defined in <style> block below */}
			<style>{`
        .ripple {
          position: absolute; /* Absolute positioning relative to button */
          border-radius: 50%; /* Make ripple shape circular */
          pointer-events: none; /* Ripple should not interfere with mouse events */
          background-color: rgba(0, 0, 0, 1); /* Ripple color - black with opacity adjust as needed */
          z-index: -1; /* Position ripple behind button content */
          opacity: 0; /* Initially hidden */
          transition: transform 50ms linear; /* Smooth transition for transform property */
        }
        .ripple-enter {
          animation: ripple-enter 600ms ease-out forwards; /* Apply ripple-enter animation on class addition */
        }
        .ripple-leave {
          animation: ripple-leave 600ms ease-out forwards; /* Apply ripple-leave animation on class addition */
        }
        @keyframes ripple-enter {
          from {
            transform: scale(0); /* Start with scale 0 (invisible) */
            opacity: 1; /* Start with opacity 1 (fully visible) */
          }
          to {
            transform: scale(1); /* End with scale 1 (fully expanded) */
            opacity: 1; /* End with opacity 1 (fully visible) */
          }
        }
        @keyframes ripple-leave {
          from {
            transform: scale(1); /* Start with scale 1 (fully expanded) */
            opacity: 1; /* Start with opacity 1 (fully visible) */
          }
          to {
            transform: scale(0); /* End with scale 0 (invisible) */
            opacity: 1; /* End with opacity 1 (fully visible) - opacity can be adjusted for fade out effect */
          }
        }
      `}</style>
		</button>
	)
}
