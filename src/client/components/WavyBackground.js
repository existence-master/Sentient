"use client"
import { cn } from "@utils/cn" // Importing cn utility for class name merging
import React, { useEffect, useRef, useState } from "react" // Importing React hooks and React
import { createNoise3D } from "simplex-noise" // Importing createNoise3D function from simplex-noise library

/**
 * WavyBackground Component - Renders an animated wavy background using canvas and simplex noise.
 *
 * This component creates a full-screen animated background with wavy lines, using HTML5 canvas
 * and simplex noise for smooth, organic wave patterns. It allows customization of wave colors,
 * width, background fill, blur, speed, and opacity through props, providing a dynamic and visually
 * appealing background effect.
 *
 * @param {object} props - Component props.
 * @param {React.ReactNode} props.children - Content to be rendered on top of the wavy background.
 * @param {string} props.className - Optional CSS class names to apply to the content container.
 * @param {string} props.containerClassName - Optional CSS class names to apply to the background container.
 * @param {string[]} props.colors - Array of colors for the wavy lines, defaults to a blueish color palette.
 * @param {number} props.waveWidth - Width of the wavy lines, defaults to 50 pixels.
 * @param {string} props.backgroundFill - Fill color of the background, defaults to "black".
 * @param {number} props.blur - Blur radius applied to the canvas, defaults to 10 pixels.
 * @param {string} props.speed - Animation speed of the waves, can be "slow" or "fast", defaults to "fast".
 * @param {number} props.waveOpacity - Opacity of the wavy lines, defaults to 0.5.
 * @param {object} props.otherProps - Any other props to be passed to the container div.
 * @returns {React.ReactNode} - The WavyBackground component UI.
 */
export const WavyBackground = ({
	children, // Content to be rendered above the wavy background - children: React.ReactNode
	className, // Optional class names for content container - className: string
	containerClassName, // Optional class names for background container - containerClassName: string
	colors, // Array of colors for waves - colors: string[]
	waveWidth, // Width of wave lines - waveWidth: number
	backgroundFill, // Background fill color - backgroundFill: string
	blur = 10, // Blur value for canvas filter, default 10px - blur: number
	speed = "fast", // Animation speed, 'slow' or 'fast', default 'fast' - speed: "slow" | "fast"
	waveOpacity = 0.5, // Opacity of wave lines, default 0.5 - waveOpacity: number
	...props // Spread operator to capture any other props - props: object
}) => {
	const noise = createNoise3D() // Initialize simplex-noise 3D noise function - noise: NoiseFunction3D
	let w, // Canvas width - w: number
		h, // Canvas height - h: number
		nt, // Noise time or animation ticker - nt: number
		i, // Loop counter variable - i: number
		x, // X-coordinate variable - x: number
		ctx, // Canvas 2D rendering context - ctx: CanvasRenderingContext2D
		canvas // Canvas element - canvas: HTMLCanvasElement
	const canvasRef = useRef(null) // useRef for the canvas element - canvasRef: React.RefObject<HTMLCanvasElement>
	/**
	 * Determines the animation speed multiplier based on the speed prop.
	 * @function getSpeed
	 * @returns {number} - Speed multiplier for animation, based on speed prop.
	 */
	const getSpeed = () => {
		switch (speed) {
			case "slow":
				return 0.001 // Slower speed multiplier
			case "fast":
				return 0.002 // Faster speed multiplier
			default:
				return 0.001 // Default speed multiplier
		}
	}

	/**
	 * Initializes canvas and animation context, sets up resize listener, and starts render loop.
	 * @function init
	 * @returns {void}
	 */
	const init = () => {
		canvas = canvasRef.current // Get canvas element from ref
		ctx = canvas.getContext("2d") // Get 2D rendering context
		w = ctx.canvas.width = window.innerWidth // Set canvas width to window width
		h = ctx.canvas.height = window.innerHeight // Set canvas height to window height
		ctx.filter = `blur(${blur}px)` // Apply blur filter to canvas context
		nt = 0 // Initialize noise time/ticker

		// Window resize event listener to handle canvas resize on window resize
		window.onresize = function () {
			w = ctx.canvas.width = window.innerWidth // Update canvas width on resize
			h = ctx.canvas.height = window.innerHeight // Update canvas height on resize
			ctx.filter = `blur(${blur}px)` // Re-apply blur filter after resize
		}
		render() // Start animation render loop
	}

	// Default wave colors array if colors prop is not provided
	const waveColors = colors || [
		"#00B2FE",
		"#005CFE",
		"#1c8cd6",
		"#137cbd",
		"#0d5b85"
	]

	/**
	 * Draws wavy lines on the canvas.
	 * @function drawWave
	 * @param {number} n - Number of wave layers to draw.
	 * @returns {void}
	 */
	const drawWave = (n) => {
		nt += getSpeed() // Increment noise time/ticker based on selected speed
		// Loop to draw multiple wave layers
		for (i = 0; i < n; i++) {
			ctx.beginPath() // Start new path for each wave
			ctx.lineWidth = waveWidth || 50 // Set line width for waves, default 50px
			ctx.strokeStyle = waveColors[i % waveColors.length] // Set stroke color for waves, cycling through waveColors array
			// Loop to generate points for each wave line
			for (x = 0; x < w; x += 5) {
				var y = noise(x / 800, 0.3 * i, nt) * 100 // Calculate Y position using noise function for wavy effect
				ctx.lineTo(x, y + h * 0.5) // Draw line to calculated point, offset vertically to center
			}
			ctx.stroke() // Stroke the path to render the wave line
			ctx.closePath() // Close path
		}
	}

	let animationId // Variable to hold animation frame ID for cancellation - animationId: number
	/**
	 * Animation render loop function.
	 *
	 * Fills background, sets global alpha, draws waves, and requests next animation frame.
	 * @function render
	 * @returns {void}
	 */
	const render = () => {
		ctx.fillStyle = backgroundFill || "black" // Set fill style for background, default black
		ctx.globalAlpha = waveOpacity || 0.5 // Set global alpha for wave opacity, default 0.5
		ctx.fillRect(0, 0, w, h) // Fill canvas with background color
		drawWave(5) // Draw 5 layers of waves
		animationId = requestAnimationFrame(render) // Request next animation frame, recursive call for animation loop
	}

	/**
	 * useEffect hook to initialize animation and handle resize cleanup.
	 *
	 * Initializes the animation by calling `init` on component mount.
	 * Sets up and clears animation frame on component unmount to optimize performance and prevent memory leaks.
	 */
	useEffect(() => {
		init() // Initialize canvas and animation on component mount

		// Cleanup function to cancel animation frame on component unmount
		return () => {
			cancelAnimationFrame(animationId) // Cancel animation frame to stop animation loop and prevent memory leaks
		}
	}, []) // Empty dependency array ensures this effect runs only once on mount and unmount

	// State to detect if the browser is Safari - isSafari: boolean
	const [isSafari, setIsSafari] = useState(false)
	/**
	 * useEffect hook to detect if the browser is Safari.
	 *
	 * Sets the `isSafari` state based on userAgent string to apply browser-specific styles or workarounds, if needed.
	 */
	useEffect(() => {
		setIsSafari(
			typeof window !== "undefined" &&
				navigator.userAgent.includes("Safari") &&
				!navigator.userAgent.includes("Chrome")
		)
		// Detect Safari browser using userAgent string (excluding Chrome on Safari, e.g., iOS Chrome)
	}, []) // Empty dependency array ensures this effect runs only once on mount

	return (
		<div
			className={cn(
				"h-screen flex flex-col items-center justify-center",
				containerClassName
			)}
			// Container div for wavy background and content: full screen height, flex layout to center content, dynamic containerClassName
		>
			<canvas
				className="absolute inset-0 z-0" // Canvas styling: absolute positioning to cover parent, z-index 0 to stay behind content
				ref={canvasRef} // Attach canvasRef to this canvas element
				id="canvas" // ID attribute for canvas element (optional)
				style={{
					...(isSafari ? { filter: `blur(${blur}px)` } : {})
					// Apply blur filter style, conditionally applied only for Safari browser for performance reasons
				}}
			></canvas>
			{/* Div to hold content, positioned above the wavy background, dynamic className */}
			<div className={cn("relative z-10", className)} {...props}>
				{children}{" "}
				{/* Render children components, positioned above wavy background */}
			</div>
		</div>
	)
}