"use client"

import { Rocket } from "lucide-react" // Importing Rocket icon from lucide-react library
import { cn } from "@utils/cn" // Importing cn utility for class name merging
import React from "react"

/**
 * SwipeButton Component - A button component with a swipeable background and icon reveal animation on hover.
 *
 * This component renders a button with a visually engaging swipe animation on hover.
 * It features a background overlay that swipes from left to right on hover to reveal an icon,
 * while simultaneously sliding the button text out of view. It is customizable via props for text, colors, and styling.
 *
 * @param {object} props - Component props.
 * @param {string} props.text - Text to be displayed on the button, defaults to "Open".
 * @param {string} props.textColor - Text color of the button text, defaults to "#bf49ff" (light purple).
 * @param {string} props.buttonOverlayColor - Background color of the sliding overlay, defaults to "#bf49ff" (light purple).
 * @param {string} props.borderColor - Border color of the button, defaults to "#c284f9" (purple).
 * @param {string} props.iconColor - Color of the icon revealed on hover, defaults to "white".
 * @param {string} props.className - Optional CSS class names to apply to the button.
 * @param {object} props.otherProps - Any other props to be passed to the button element.
 * @returns {React.ReactNode} - The SwipeButton component UI.
 */
export default function SwipeButton({
	text = "Open", // Button text, default "Open" - text: string
	textColor = "#bf49ff", // Text color, default light purple - textColor: string
	buttonOverlayColor = "#bf49ff", // Overlay background color, default light purple - buttonOverlayColor: string
	borderColor = "#c284f9", // Border color, default purple - borderColor: string
	iconColor = "white", // Icon color, default white - iconColor: string
	className, // Optional class names for button styling - className: string
	...props // Spread operator to capture any other props - props: object
}) {
	return (
		<button
			style={{ borderColor: borderColor }} // Dynamic border color from borderColor prop, applied as inline style
			{...props} // Spreading any other props to the button element
			className={cn(
				"group relative inline-flex items-center justify-center overflow-hidden rounded-full border-2 border-purple-400 bg-background px-6 py-3 font-medium shadow-md transition duration-300 ease-out",
				// Base button styling: group class for hover effects, relative and inline-flex display, items and content centering, overflow hidden for animation, rounded full corners, border width, border color, background color, padding, font, shadow, transition for duration and easing, dynamic className
				className // Dynamic classNames for SwipeButton component
			)}
		>
			{/* Sliding overlay span element */}
			<span
				style={{ background: buttonOverlayColor }} // Dynamic background color from buttonOverlayColor prop, applied as inline style
				className={cn(
					"ease absolute inset-0 flex h-full w-full -translate-x-full items-center justify-center bg-purple-400 text-white duration-300 group-hover:translate-x-0"
					// Sliding overlay styling: ease-in-out animation, absolute positioning to cover button, flex layout to center content, full height and width, initial translation to hide, background color, white text color, transition duration, hover translation to reveal
				)}
			>
				<Rocket style={{ color: iconColor }} />
				{/* Rocket icon, color from iconColor prop, revealed on hover */}
			</span>
			{/* Button text span element */}
			<span
				style={{ color: textColor }} // Dynamic text color from textColor prop, applied as inline style
				className={cn(
					"absolute flex h-full w-full transform items-center justify-center font-bold transition-all duration-300 ease-in-out group-hover:translate-x-full"
					// Button text styling: absolute positioning, flex layout to center content, full height and width, transform for animation, items and content centering, font weight bold, transition for all properties, duration and easing, hover translation to move text out of view
				)}
			>
				{text} {/* Button text from text prop */}
			</span>
			{/* Invisible span for layout purposes, ensuring button dimensions are maintained during animation */}
			<span className="invisible relative">Button</span>
		</button>
	)
}
