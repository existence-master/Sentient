"use client"

import React, { useState } from "react" // Importing React library and useState hook
import { Plus } from "lucide-react" // Importing Plus icon from lucide-react library

/**
 * Tooltip Component - A reusable tooltip component that displays text on hover.
 *
 * This component renders a tooltip that appears when the user hovers over its children.
 * It's highly customizable, allowing for different placements and directions.
 *
 * @param {object} props - Component props.
 * @param {string} props.text - Text content to be displayed inside the tooltip.
 * @param {React.ReactNode} props.children - The children components that the tooltip will trigger on hover.
 * @param {string} props.direction - Direction in which the tooltip expands, can be "up", "down", "left", or "right".
 * @param {string} props.tooltipPosition - Position of the tooltip relative to the child, e.g., "right", "left".
 * @returns {React.ReactNode} - The Tooltip component UI.
 */
const Tooltip = ({ text, children, direction, tooltipPosition }) => {
	// State to manage tooltip visibility - visible: boolean
	const [visible, setVisible] = useState(false) // Initially set to false

	/**
	 * Shows the tooltip by setting the visible state to true.
	 * @function showTooltip
	 * @returns {void}
	 */
	const showTooltip = () => setVisible(true) // Function to set tooltip visibility to true

	/**
	 * Hides the tooltip by setting the visible state to false.
	 * @function hideTooltip
	 * @returns {void}
	 */
	const hideTooltip = () => setVisible(false) // Function to set tooltip visibility to false

	return (
		<div
			onMouseEnter={showTooltip}
			onMouseLeave={hideTooltip}
			className="relative inline-block"
		>
			{/* Container div for tooltip and children, relative positioning for tooltip absolute positioning, inline-block to wrap content */}
			{children}{" "}
			{/* Render children components that trigger the tooltip */}
			{/* Conditional rendering of tooltip based on visibility state */}
			{visible && (
				<div
					className={` ${
						direction === "up" || direction === "down"
							? tooltipPosition === "right"
								? "absolute left-full top-1/2 z-10 ml-2 -translate-y-1/2 transform rounded-sm min-w-fit bg-gray-800 px-2 py-1 text-sm text-white"
								: "absolute right-full top-1/2 z-10 ml-2 -translate-y-1/2 transform rounded-sm min-w-fit bg-gray-800 px-2 py-1 text-sm text-white"
							: // Conditional classNames for tooltip positioning based on direction and tooltipPosition props
							  // For 'up' or 'down' direction: positions tooltip to the right or left based on tooltipPosition
							  "absolute bottom-full left-1/2 z-10 mb-2 -translate-x-1/2 transform rounded-sm min-w-fit bg-gray-800 px-2 py-1 text-sm text-white"
						// Default positioning for other directions: positions tooltip above the child, centered horizontally
					} `}
				>
					{text} {/* Tooltip text content */}
				</div>
			)}
		</div>
	)
}

/**
 * Speeddial Component - A speed dial component with expandable action buttons on hover.
 *
 * This component renders a speed dial button that expands to reveal a list of action buttons when hovered.
 * It's customizable in terms of icon, direction of expansion, and the action buttons it contains.
 * It uses the Tooltip component to provide labels for each action button on hover.
 *
 * @param {object} props - Component props.
 * @param {React.ReactNode} props.icon - Icon to display on the main speed dial button.
 * @param {string} props.direction - Direction in which the speed dial expands, can be "up", "down", "left", or "right".
 * @param {Array<object>} props.actionButtons - Array of action button objects, each with label, action, and icon.
 * @param {string} props.tooltipPosition - Position of the tooltip relative to the action button, e.g., "right", "left".
 * @returns {React.ReactNode} - The Speeddial component UI.
 */
export default function Speeddial({
	icon,
	direction,
	actionButtons,
	tooltipPosition
}) {
	// State to manage hover status, controlling visibility of action buttons - isHovered: boolean
	const [isHovered, setIsHovered] = useState(false) // Initially set to false

	/**
	 * Determines the animation origin and flex direction classes based on the direction prop.
	 * @function getAnimation
	 * @returns {string} - CSS class names for animation, based on direction prop.
	 */
	const getAnimation = () => {
		switch (direction) {
			case "up":
				return "origin-bottom flex-col order-0" // Expand upwards, flex column layout, order 0 in flex container
			case "down":
				return "origin-top flex-col order-2" // Expand downwards, flex column layout, order 2 in flex container
			case "left":
				return "origin-right order-0" // Expand to the left, default flex row layout, order 0 in flex container
			case "right":
				return "origin-left order-2" // Expand to the right, default flex row layout, order 2 in flex container
			default:
				return "" // Default case: no extra animation classes
		}
	}

	// Event handler for mouse enter to set hover state to true - handleMouseEnter: () => void
	const handleMouseEnter = () => setIsHovered(true)
	// Event handler for mouse leave to set hover state to false - handleMouseLeave: () => void
	const handleMouseLeave = () => setIsHovered(false)

	/**
	 * Returns CSS class names for glassy button effect, reused for main button and action buttons.
	 * @function getGlassyClasses
	 * @returns {string} - CSS class names for glassy button style.
	 */
	const getGlassyClasses = () => {
		return "backdrop-filter backdrop-blur-xl hover:border-lightblue bg-white border-2 border-white rounded-xl shadow-lg transition-all duration-300"
		// Class names for glassy button: backdrop blur, hover border, white background, border, rounded corners, shadow, transition
	}

	return (
		<div
			onMouseLeave={handleMouseLeave} // Call handleMouseLeave on mouse leave of the speed dial container
			className={`relative mb-3 flex w-fit items-center gap-3 ${
				direction === "up" || direction === "down"
					? "flex-col"
					: "flex-row"
			}`}
			// Container div for speed dial: relative positioning, margin bottom, flex layout, dynamic flex direction based on direction prop, width fit to content, items-center for vertical alignment, gap for spacing
		>
			{/* Main speed dial button */}
			<button
				onMouseEnter={handleMouseEnter} // Call handleMouseEnter on mouse enter of the main button
				className={`${getGlassyClasses()} order-1 flex items-center p-2 text-gray-800 transition-all cursor-pointer duration-300 hover:bg-slate-100`}
				// Main button styling: glassy classes from getGlassyClasses, order classes for flex layout, flex layout for icon centering, padding, text color, transition, hover background
			>
				{icon ? icon : <Plus size={20} />}
				{/* Icon for the main button, use icon prop if provided, else default to Plus icon, size 20 */}
			</button>

			{/* Action buttons container, expands on hover */}
			<div
				className={`${
					isHovered ? "scale-100 opacity-100" : "scale-0 opacity-0" // Dynamic scaling and opacity based on hover state for expand/collapse animation
				} flex items-center gap-3 transition-all duration-500 ease-in-out ${getAnimation()}`}
				// Action buttons container styling: dynamic scale and opacity based on hover, flex layout, items-center for vertical alignment, gap for spacing, transition for scale and opacity, animation classes from getAnimation
			>
				{/* Map over actionButtons array to render each action button */}
				{actionButtons.map((action, index) => (
					<Tooltip
						text={action.label}
						key={index}
						direction={direction}
						tooltipPosition={tooltipPosition}
					>
						{/* Tooltip for each action button, tooltip text from action.label, dynamic direction and tooltipPosition */}
						<button
							key={index} // Key prop for React list rendering
							onClick={action.action} // Call action.action function on button click
							className={`${getGlassyClasses()} flex cursor-pointer items-center p-3 text-gray-800 transition-all duration-300 hover:bg-slate-100`}
							// Action button styling: glassy classes from getGlassyClasses, flex layout for icon centering, padding, text color, transition, hover background
						>
							{action.icon}{" "}
							{/* Icon for the action button from action.icon */}
						</button>
					</Tooltip>
				))}
			</div>
		</div>
	)
}