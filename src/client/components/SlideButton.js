import { ArrowRight } from "lucide-react" // Importing ArrowRight icon from lucide-react library
import React from "react"

/**
 * SlideButton Component - A button component with a sliding background and icon animation on hover.
 *
 * This component renders a button with a visually engaging slide animation on hover.
 * It features a background that slides from left to right on hover, revealing an icon and changing text color.
 * It's customizable in terms of text, colors, and icon, providing a modern and interactive button style.
 *
 * @param {object} props - Component props.
 * @param {string} props.text - Text to be displayed on the button, defaults to "Get Started".
 * @param {string} props.primaryColor - Primary color for the sliding background, defaults to "#6f3cff" (a shade of purple).
 * @param {string} props.className - Optional CSS class names to apply to the button.
 * @param {React.ReactNode} props.icon - Icon to display inside the button, defaults to ArrowRight icon from lucide-react.
 * @param {object} props.otherProps - Any other props to be passed to the button element.
 * @returns {React.ReactNode} - The SlideButton component UI.
 */
export default function SlideButton({
	text = "Get Started", // Button text, default "Get Started" - text: string
	primaryColor = "#6f3cff", // Primary color for slide background, default light purple - primaryColor: string
	className = "", // Optional class names for button styling - className: string
	icon, // Icon to display, optional - icon: React.ReactNode
	...props // Spread operator to capture any other props - props: object
}) {
	return (
		<button
			className={`group relative rounded-full border border-white bg-white p-2 text-xl font-semibold ${className}`}
			// Base button styling: group class for hover effects, relative positioning, rounded full corners, border, white background, padding, text size, font weight, dynamic className
			{...props} // Spreading any other props to the button element
		>
			{/* Sliding background element */}
			<div
				className="absolute left-0 top-0 flex h-full w-11 items-center justify-end rounded-full transition-all duration-200 ease-in-out group-hover:w-full"
				// Sliding background styling: absolute positioning, left and top 0, flex layout to center content, fixed width initially, rounded full corners, transition for all properties, duration, easing, hover expands width to full
				style={{ backgroundColor: primaryColor }} // Dynamic background color from primaryColor prop
			>
				<span className="mr-3 text-white transition-all duration-200 ease-in-out">
					{/* Icon container span styling: margin right for spacing, white text color, transition for all properties, duration and easing */}
					{icon ? icon : <ArrowRight size={20} />}
					{/* Render icon prop if provided, else default to ArrowRight icon, size 20 */}
				</span>
			</div>
			{/* Button text span */}
			<span className="relative left-4 z-10 whitespace-nowrap px-8 font-semibold text-black transition-all duration-200 ease-in-out group-hover:-left-3 group-hover:text-white">
				{text} {/* Render button text from text prop */}
			</span>
		</button>
	)
}
