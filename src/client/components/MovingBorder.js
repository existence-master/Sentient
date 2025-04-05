"use client"
import React from "react" // Importing React library
import {
	motion, // Importing motion component for animations
	useAnimationFrame, // Importing hook for animation frame updates
	useMotionTemplate, // Importing hook for creating motion templates
	useMotionValue, // Importing hook for creating motion values
	useTransform // Importing hook for transforming motion values
} from "framer-motion"
import { useRef } from "react" // Importing useRef hook from React
import { cn } from "@utils/cn" // Importing cn utility for class name merging

/**
 * Button Component - A customizable button with a moving border animation.
 *
 * This component creates a button with a visually engaging animated border.
 * It uses a MovingBorder component for the animated border effect and provides
 * flexibility in terms of styling, size, and content. It's built using framer-motion
 * for smooth animations and transitions.
 *
 * @param {object} props - Component props.
 * @param {string} props.borderRadius - Border radius of the button, defaults to "1.75rem".
 * @param {React.ReactNode} props.children - Content to be rendered inside the button.
 * @param {elementType} props.as - Type of element to render as the base for the button, defaults to "button".
 * @param {string} props.containerClassName - Optional CSS class names for the button container.
 * @param {string} props.borderClassName - Optional CSS class names for the moving border element.
 * @param {number} props.duration - Duration of the border animation in milliseconds.
 * @param {string} props.className - Optional CSS class names for the button text/content area.
 * @param {object} props.otherProps - Any other props to be passed to the button element.
 * @returns {React.ReactNode} - The Button component UI.
 */
export function Button({
	borderRadius = "1.75rem", // Border radius for the button, defaults to '1.75rem' - borderRadius: string
	children, // Content of the button - children: React.ReactNode
	as: Component = "button", // Element type for the button, defaults to 'button' - Component: elementType
	containerClassName, // Optional class names for the button container - containerClassName: string
	borderClassName, // Optional class names for the moving border - borderClassName: string
	duration, // Duration for the moving border animation - duration: number
	className, // Optional class names for the button's text/content area - className: string
	...otherProps // Spread operator to capture any other props - otherProps: object
}) {
	return (
		<Component
			className={cn(
				"bg-transparent relative text-xl  h-16 w-40 p-[1px] overflow-hidden ", // Base button styling: transparent background, relative positioning, text size, fixed height and width, padding, overflow hidden
				containerClassName // Dynamic classNames for container
			)}
			style={{
				borderRadius: borderRadius // Border radius from props, applied as inline style
			}}
			{...otherProps} // Spreading any other props to the button component
		>
			<div
				className="absolute inset-0" // Absolute positioning to cover the button
				style={{ borderRadius: `calc(${borderRadius} * 0.96)` }} // Slightly smaller border radius for inner element
			>
				{/* MovingBorder component for animated border effect */}
				<MovingBorder duration={duration} rx="30%" ry="30%">
					<div
						className={cn(
							"h-20 w-20 opacity-[0.8] bg-[radial-gradient(var(--sky-500)_40%,transparent_60%)]", // Styling for the moving border element: fixed height and width, opacity, radial gradient background
							borderClassName // Dynamic classNames for border element
						)}
					/>
				</MovingBorder>
			</div>
			<div
				className={cn(
					"relative bg-slate-900/[0.8] border border-slate-800 backdrop-blur-xl text-white flex items-center justify-center w-full h-full text-sm antialiased",
					// Inner content container styling: relative positioning, semi-transparent dark background, border, backdrop blur, white text color, flex layout to center content, text size, font smoothing
					className // Dynamic classNames for inner content container
				)}
				style={{
					borderRadius: `calc(${borderRadius} * 0.96)` // Slightly smaller border radius for inner content container
				}}
			>
				{children} {/* Render children (button text/content) */}
			</div>
		</Component>
	)
}

/**
 * MovingBorder Component - Animates a border moving around a given path.
 *
 * This component creates an animated border effect by moving a div around a rectangular path,
 * defined by a <rect> element within an <svg>. It uses framer-motion's animation hooks
 * to smoothly animate the movement of the border.
 *
 * @param {object} props - Component props.
 * @param {React.ReactNode} props.children - The element to be animated along the border path.
 * @param {number} props.duration - Duration of one full animation cycle in milliseconds, defaults to 2000ms.
 * @param {string} props.rx - x-axis radius for rounded corners of the rectangle path.
 * @param {string} props.ry - y-axis radius for rounded corners of the rectangle path.
 * @param {object} props.otherProps - Any other props to be passed to the SVG element.
 * @returns {React.ReactNode} - The MovingBorder component UI.
 */
export const MovingBorder = ({
	children, // Element to be animated along the border path - children: React.ReactNode
	duration = 2000, // Duration of animation cycle, defaults to 2000ms - duration: number
	rx, // x-axis radius for rounded corners of path - rx: string
	ry, // y-axis radius for rounded corners of path - ry: string
	...otherProps // Spread operator to capture any other props - otherProps: object
}) => {
	const pathRef = useRef() // useRef to get the SVG path element - pathRef: React.RefObject<SVGRectElement>
	// useMotionValue to track animation progress from 0 to path length - progress: MotionValue<number>
	const progress = useMotionValue(0)

	/**
	 * useAnimationFrame hook to update animation progress based on time.
	 *
	 * This hook runs on every animation frame and updates the `progress` motion value based on elapsed time.
	 * It calculates the progress by dividing the current time by the duration of the animation cycle,
	 * ensuring a smooth, continuous animation loop.
	 */
	useAnimationFrame((time) => {
		const length = pathRef.current?.getTotalLength() // Get total length of the SVG path
		if (length) {
			const pxPerMillisecond = length / duration // Calculate pixels to move per millisecond
			progress.set((time * pxPerMillisecond) % length) // Update progress motion value based on time and path length
		}
	})

	// useTransform hook to derive x position from animation progress - x: MotionValue<number>
	const x = useTransform(
		progress,
		(val) => pathRef.current?.getPointAtLength(val).x
	)
	// useTransform hook to derive y position from animation progress - y: MotionValue<number>
	const y = useTransform(
		progress,
		(val) => pathRef.current?.getPointAtLength(val).y
	)

	// useMotionTemplate to create dynamic transform style based on x and y motion values - transform: MotionValue<string>
	const transform = useMotionTemplate`translateX(${x}px) translateY(${y}px) translateX(-50%) translateY(-50%)`

	return (
		<>
			<svg
				xmlns="http://www.w3.org/2000/svg" // Specifies the namespace for SVG
				preserveAspectRatio="none" // Disables aspect ratio preservation
				className="absolute h-full w-full" // Absolute positioning to cover parent, full height and width
				width="100%" // SVG width 100% of container
				height="100%" // SVG height 100% of container
				{...otherProps} // Spreads any additional props to the SVG element
			>
				{/* Rectangle path for the moving border */}
				<rect
					fill="none"
					width="100%"
					height="100%"
					rx={rx}
					ry={ry}
					ref={pathRef}
				/>
				{/* Rectangle styling: no fill, 100% width and height, rounded corners from props, ref attached */}
			</svg>
			{/* Animated div element */}
			<motion.div
				style={{
					position: "absolute", // Absolute positioning for animation within SVG
					top: 0, // Position at the top
					left: 0, // Position at the left
					display: "inline-block", // Inline-block display for proper transform application
					transform // Apply dynamic transform from useMotionTemplate
				}}
			>
				{children} {/* Render children (animated border element) */}
			</motion.div>
		</>
	)
}