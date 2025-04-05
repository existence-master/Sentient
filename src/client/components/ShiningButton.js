"use client"
import { Tooltip } from "@node_modules/react-tooltip/dist/react-tooltip" // Importing Tooltip component from react-tooltip library
import { cn } from "@utils/cn" // Importing cn utility for class name merging
import React from "react"

/**
 * ShiningButton Component - A button component with a shining animation effect and optional icon and tooltip.
 *
 * This component renders a button with a visually engaging "shining" animation effect on hover.
 * It's highly customizable, allowing for different background and border colors, optional icons,
 * tooltips, and various styling options through props.
 *
 * @param {object} props - Component props.
 * @param {React.ReactNode} props.children - Content to be rendered inside the ShiningButton (e.g., button text).
 * @param {function} props.onClick - Handler function to be called when the button is clicked.
 * @param {boolean} props.disabled - Boolean to disable the button, defaults to false.
 * @param {string} props.className - Optional CSS class names to apply to the button's text/content area.
 * @param {string} props.borderClassName - Optional CSS class names to apply to the button's border.
 * @param {React.ReactNode} props.icon - Icon component to be displayed inside the button, typically from tabler-icons-react or similar.
 * @param {string} props.bgColor - Background color class for the button, defaults to "bg-lightblue".
 * @param {string} props.borderColor - Border color class for the button, defaults to "border-lightblue".
 * @param {string} props.type - Type attribute for the button (e.g., "button", "submit", "reset"), defaults to "button".
 * @param {string} props.dataTooltipId - ID for the tooltip, if a tooltip is needed.
 * @param {string} props.dataTooltipContent - Content of the tooltip, displayed on hover.
 * @returns {React.ReactNode} - The ShiningButton component UI.
 */
export default function ShiningButton({
	children, // Content of the button, e.g., text - children: React.ReactNode
	onClick, // Click handler function for the button - onClick: () => void
	disabled, // Boolean to disable the button, default false - disabled: boolean
	className, // Optional class names for button content area - className: string
	borderClassName, // Optional class names for button border - borderClassName: string
	icon: Icon, // Icon component to display in button - Icon: React.ComponentType<any>
	bgColor = "bg-lightblue", // Background color class, default 'bg-lightblue' - bgColor: string
	borderColor = "border-lightblue", // Border color class, default 'border-lightblue' - borderColor: string
	type = "button", // Button type, default 'button' - type: "button" | "submit" | "reset"
	dataTooltipId = "", // Tooltip ID for react-tooltip - dataTooltipId: string
	dataTooltipContent = "" // Tooltip content for react-tooltip - dataTooltipContent: string
}) {
	return (
		<>
			{/* React Fragment to wrap button and tooltip */}
			<button
				onClick={onClick} // Click event handler from props
				type={type} // Button type from props
				disabled={disabled} // Disabled state from props
				className={`group ${
					disabled
						? "opacity-50 cursor-not-allowed"
						: "cursor-pointer" // Dynamic cursor and opacity classes based on disabled prop
				} rounded-xl border-0 ${borderColor} ${borderClassName} bg-transparent p-1 transition-all duration-500 hover:border-4 hover:${borderColor} ${className}`}
				// Base button styling: group for hover effects, dynamic cursor and opacity based on disabled, rounded corners, border width, dynamic border color and classNames, transparent background, padding, transition for all properties, hover border opacity, dynamic classNames
				data-tooltip-id={dataTooltipId} // Tooltip ID for react-tooltip
				data-tooltip-content={dataTooltipContent} // Tooltip content for react-tooltip
			>
				<div
					className={`relative flex items-center justify-center gap-4 overflow-hidden rounded-lg ${bgColor} px-6 py-4 font-bold text-white ${className}`}
					// Inner button container styling: relative positioning, flex layout to center content, gap between items, overflow hidden for animation, rounded corners, dynamic background color and classNames, padding, bold font, white text color
				>
					{children}{" "}
					{/* Button content (text) - children: React.ReactNode */}
					{/* Conditional rendering of icon if Icon component is provided */}
					{Icon && (
						<Icon
							className={cn(
								"transition-all group-hover:translate-x-2 group-hover:scale-125",
								// Icon styling: transition for all properties, hover translation on X-axis, hover scale increase, dynamic opacity based on disabled state
								{ "opacity-100": !disabled }
							)}
						/>
					)}
					{/* Shining effect element - white strip with animation on hover */}
					<div
						className={cn(
							"absolute -left-16 top-0 h-full w-12 rotate-[30deg] scale-y-150 bg-white/10 transition-all duration-700 group-hover:left-[calc(100%+1rem)]"
							// Shining strip styling: absolute positioning, offset to left, full height, fixed width, rotation, vertical scale, semi-transparent white background, transition for all properties, hover translation to move strip out of view
						)}
					/>
				</div>
			</button>
			{/* Tooltip component from react-tooltip, conditionally rendered if dataTooltipId is provided */}
			<Tooltip
				id={dataTooltipId}
				place="bottom"
				type="dark"
				effect="float"
			/>
		</>
	)
}
