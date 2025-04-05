"use client"

import { cn } from "@utils/cn" // Importing cn utility for class name merging
import { useEffect, useRef } from "react" // Importing React hooks: useEffect, useRef
import { createNoise3D } from "simplex-noise" // Importing createNoise3D function from simplex-noise library
import { motion } from "framer-motion" // Importing motion component from framer-motion library
import React from "react"

/**
 * Vortex Component - Renders an animated vortex background using canvas and simplex noise.
 *
 * This component creates a mesmerizing vortex animation using HTML5 canvas and simplex noise for organic movement.
 * It customizes particle behavior, colors, and animation dynamics through props, offering a visually rich
 * and customizable background effect. The animation is performant and adapts to window resizing.
 *
 * @param {object} props - Component props.
 * @param {number} props.particleCount - Number of particles in the vortex, defaults to 500.
 * @param {number} props.rangeY - Vertical range of particle distribution, defaults to 50.
 * @param {number} props.baseTTL - Base time-to-live for particles, defaults to 50 frames.
 * @param {number} props.rangeTTL - Range of random time-to-live variation for particles, defaults to 100 frames.
 * @param {number} props.baseSpeed - Base movement speed of particles, defaults to 0.0.
 * @param {number} props.rangeSpeed - Range of random speed variation for particles, defaults to 1.0.
 * @param {number} props.baseRadius - Base radius of particles, defaults to 1 pixel.
 * @param {number} props.rangeRadius - Range of random radius variation for particles, defaults to 1 pixel.
 * @param {string} props.backgroundColor - Background color of the canvas, defaults to "#000000" (black).
 * @param {string} props.className - Optional CSS class names to apply to the content container.
 * @param {string} props.containerClassName - Optional CSS class names to apply to the vortex container.
 * @param {React.ReactNode} props.children - Content to be rendered above the vortex animation.
 * @returns {React.ReactNode} - The Vortex component UI.
 */
export const Vortex = (props) => {
	const canvasRef = useRef(null) // useRef for the canvas element - canvasRef: React.RefObject<HTMLCanvasElement>
	const containerRef = useRef(null) // useRef for the container div - containerRef: React.RefObject<HTMLDivElement>
	const particleCount = props.particleCount || 500 // Number of particles, default 500 - particleCount: number
	const particlePropCount = 9 // Number of properties per particle (x, y, vx, vy, life, ttl, speed, radius, hue) - particlePropCount: number
	const particlePropsLength = particleCount * particlePropCount // Total length of particle properties array - particlePropsLength: number
	const rangeY = props.rangeY || 50 // Vertical range for particle distribution, default 50 - rangeY: number
	const baseTTL = 50 // Base time-to-live for particles, default 50 frames - baseTTL: number
	const rangeTTL = 100 // Range of TTL variation, default 100 frames - rangeTTL: number
	const baseSpeed = props.baseSpeed || 0.0 // Base particle speed, default 0.0 - baseSpeed: number
	const rangeSpeed = props.rangeSpeed || 1.0 // Range of speed variation, default 1.0 - rangeSpeed: number
	const baseRadius = props.baseRadius || 1 // Base particle radius, default 1 pixel - baseRadius: number
	const rangeRadius = props.rangeRadius || 1 // Range of radius variation, default 1 pixel - rangeRadius: number
	const baseHue = 200 // Base hue value for particle color, default 200 (cyan/blue range) - baseHue: number
	const rangeHue = 40 // Range of hue variation, default 40 - rangeHue: number
	const noiseSteps = 2 // Noise factor for particle movement - noiseSteps: number
	const xOff = 0.001 // X-axis offset for noise function - xOff: number
	const yOff = 0.001 // Y-axis offset for noise function - yOff: number
	const zOff = 0.0005 // Z-axis offset for noise function (time evolution) - zOff: number
	const backgroundColor = props.backgroundColor || "#000000" // Background color, default black - backgroundColor: string
	let tick = 0 // Animation frame counter - tick: number
	const noise3D = createNoise3D() // Initialize simplex-noise 3D noise function - noise3D: NoiseFunction3D
	let particleProps = new Float32Array(particlePropsLength) // Particle properties array, Float32Array for performance - particleProps: Float32Array
	let center = [0, 0] // Center of canvas, updated on resize - center: number[]

	const TAU = 2 * Math.PI // Constant for TAU (2PI) radians - TAU: number
	const rand = (n) => n * Math.random() // Function to generate random number up to n - rand: (n: number) => number
	const randRange = (n) => n - rand(2 * n) // Function to generate random number in range -n to n - randRange: (n: number) => number
	const fadeInOut = (t, m) => {
		// Function for fade-in-out effect based on time t and max time m - fadeInOut: (t: number, m: number) => number
		let hm = 0.5 * m // Half of max time - hm: number
		return Math.abs(((t + hm) % m) - hm) / hm // Calculate fade-in-out value
	}
	const lerp = (n1, n2, speed) => (1 - speed) * n1 + speed * n2 // Linear interpolation function - lerp: (n1: number, n2: number, speed: number) => number

	/**
	 * Initializes canvas and animation setup.
	 *
	 * Gets canvas context, resizes canvas, initializes particles, and starts the draw loop.
	 * @function setup
	 * @returns {void}
	 */
	const setup = () => {
		const canvas = canvasRef.current // Get canvas element from ref
		const container = containerRef.current // Get container element from ref
		if (canvas && container) {
			const ctx = canvas.getContext("2d") // Get 2D rendering context

			if (ctx) {
				resize(canvas, ctx) // Resize canvas to window dimensions
				initParticles() // Initialize particles array
				draw(canvas, ctx) // Start animation draw loop
			}
		}
	}

	/**
	 * Initializes particle properties for all particles.
	 *
	 * Resets tick counter and populates particleProps array with initial values for each particle.
	 * @function initParticles
	 * @returns {void}
	 */
	const initParticles = () => {
		tick = 0 // Reset animation frame counter
		particleProps = new Float32Array(particlePropsLength) // Re-initialize particle properties array

		// Initialize properties for each particle
		for (let i = 0; i < particlePropsLength; i += particlePropCount) {
			initParticle(i) // Call initParticle for each particle
		}
	}

	/**
	 * Initializes properties for a single particle.
	 *
	 * Sets random values for particle position, velocity, life, time-to-live, speed, radius, and hue.
	 * @function initParticle
	 * @param {number} i - Index of the particle in the particleProps array.
	 * @returns {void}
	 */
	const initParticle = (i) => {
		const canvas = canvasRef.current // Get canvas element from ref
		if (!canvas) return

		let x, y, vx, vy, life, ttl, speed, radius, hue // Declare particle properties

		x = rand(canvas.width) // Random X position within canvas width
		y = center[1] + randRange(rangeY) // Random Y position centered and within vertical range
		vx = 0 // Initial velocity X
		vy = 0 // Initial velocity Y
		life = 0 // Particle life counter
		ttl = baseTTL + rand(rangeTTL) // Random time-to-live
		speed = baseSpeed + rand(rangeSpeed) // Random speed
		radius = baseRadius + rand(rangeRadius) // Random radius
		hue = baseHue + rand(rangeHue) // Random hue value

		particleProps.set([x, y, vx, vy, life, ttl, speed, radius, hue], i) // Set particle properties in array
	}

	/**
	 * Main animation draw loop.
	 *
	 * Clears canvas, fills background, draws particles, renders glow, and composites to screen.
	 * Uses requestAnimationFrame for smooth animation.
	 * @function draw
	 * @param {HTMLCanvasElement} canvas - Canvas element to draw on.
	 * @param {CanvasRenderingContext2D} ctx - 2D rendering context.
	 * @returns {void}
	 */
	const draw = (canvas, ctx) => {
		tick++ // Increment animation frame counter

		ctx.clearRect(0, 0, canvas.width, canvas.height) // Clear canvas on each frame

		ctx.fillStyle = backgroundColor // Set background fill style
		ctx.fillRect(0, 0, canvas.width, canvas.height) // Fill canvas with background color

		drawParticles(ctx) // Draw particles on canvas
		renderGlow(canvas, ctx) // Apply glow effect
		renderToScreen(canvas, ctx) // Composite rendered image to screen

		window.requestAnimationFrame(() => draw(canvas, ctx)) // Request next animation frame, recursive call for animation loop
	}

	/**
	 * Draws all particles by iterating through particleProps array.
	 * @function drawParticles
	 * @param {CanvasRenderingContext2D} ctx - 2D rendering context.
	 * @returns {void}
	 */
	const drawParticles = (ctx) => {
		// Iterate through particleProps array, incrementing by particlePropCount to access each particle's properties
		for (let i = 0; i < particlePropsLength; i += particlePropCount) {
			updateParticle(i, ctx) // Update and draw each particle
		}
	}

	/**
	 * Updates and draws a single particle.
	 *
	 * Updates particle position based on simplex noise, applies velocity, handles particle bounds,
	 * and initializes particle if out of bounds or TTL expired.
	 * @function updateParticle
	 * @param {number} i - Index of the particle in the particleProps array.
	 * @param {CanvasRenderingContext2D} ctx - 2D rendering context.
	 * @returns {void}
	 */
	const updateParticle = (i, ctx) => {
		const canvas = canvasRef.current // Get canvas element from ref
		if (!canvas) return

		// Indices for accessing particle properties in the Float32Array
		let i2 = 1 + i, // Index for Y position
			i3 = 2 + i, // Index for velocity X
			i4 = 3 + i, // Index for velocity Y
			i5 = 4 + i, // Index for life
			i6 = 5 + i, // Index for TTL (time-to-live)
			i7 = 6 + i, // Index for speed
			i8 = 7 + i, // Index for radius
			i9 = 8 + i // Index for hue
		let n, x, y, vx, vy, life, ttl, speed, x2, y2, radius, hue // Declare variables for particle properties

		x = particleProps[i] // X position
		y = particleProps[i2] // Y position
		n = noise3D(x * xOff, y * yOff, tick * zOff) * noiseSteps * TAU // Calculate noise value for particle direction
		vx = lerp(particleProps[i3], Math.cos(n), 0.5) // Interpolate velocity X with noise direction
		vy = lerp(particleProps[i4], Math.sin(n), 0.5) // Interpolate velocity Y with noise direction
		life = particleProps[i5] // Particle life
		ttl = particleProps[i6] // Particle time-to-live
		speed = particleProps[i7] // Particle speed
		x2 = x + vx * speed // Calculate next X position
		y2 = y + vy * speed // Calculate next Y position
		radius = particleProps[i8] // Particle radius
		hue = particleProps[i9] // Particle hue

		drawParticle(x, y, x2, y2, life, ttl, radius, hue, ctx) // Draw particle

		life++ // Increment particle life

		particleProps[i] = x2 // Update particle X position
		particleProps[i2] = y2 // Update particle Y position
		particleProps[i3] = vx // Update particle velocity X
		particleProps[i4] = vy // Update particle velocity Y
		particleProps[i5] = life // Update particle life
		;(checkBounds(x, y, canvas) || life > ttl) && initParticle(i) // Re-initialize particle if out of bounds or TTL expired
	}

	/**
	 * Draws a single particle as a line.
	 *
	 * Sets line style, color, and draws a line from current position to next position.
	 * @function drawParticle
	 * @param {number} x - Current X position.
	 * @param {number} y - Current Y position.
	 * @param {number} x2 - Next X position.
	 * @param {number} y2 - Next Y position.
	 * @param {number} life - Particle life.
	 * @param {number} ttl - Particle time-to-live.
	 * @param {number} radius - Particle radius (line width).
	 * @param {number} hue - Particle hue (color).
	 * @param {CanvasRenderingContext2D} ctx - 2D rendering context.
	 * @returns {void}
	 */
	const drawParticle = (x, y, x2, y2, life, ttl, radius, hue, ctx) => {
		ctx.save() // Save current canvas state
		ctx.lineCap = "round" // Set line cap to round for smoother lines
		ctx.lineWidth = radius // Set line width to particle radius
		ctx.strokeStyle = `hsla(${hue},100%,60%,${fadeInOut(life, ttl)})` // Set stroke style with hue, saturation, lightness, and fade-in-out opacity
		ctx.beginPath() // Begin path for drawing
		ctx.moveTo(x, y) // Move path starting point to current particle position
		ctx.lineTo(x2, y2) // Draw line to next particle position
		ctx.stroke() // Stroke the path (draw the line)
		ctx.closePath() // Close path
		ctx.restore() // Restore canvas state
	}

	/**
	 * Checks if particle is out of canvas bounds.
	 * @function checkBounds
	 * @param {number} x - Particle X position.
	 * @param {number} y - Particle Y position.
	 * @param {HTMLCanvasElement} canvas - Canvas element.
	 * @returns {boolean} - True if particle is out of bounds, false otherwise.
	 */
	const checkBounds = (x, y, canvas) => {
		return x > canvas.width || x < 0 || y > canvas.height || y < 0 // Check if x or y is outside canvas bounds
	}

	/**
	 * Resizes canvas to window dimensions and updates center point.
	 * @function resize
	 * @param {HTMLCanvasElement} canvas - Canvas element to resize.
	 * @param {CanvasRenderingContext2D} ctx - 2D rendering context.
	 * @returns {void}
	 */
	const resize = (canvas) => {
		const { innerWidth, innerHeight } = window // Get window inner width and height

		canvas.width = innerWidth // Set canvas width to window width
		canvas.height = innerHeight // Set canvas height to window height

		center[0] = 0.5 * canvas.width // Update center X to canvas center
		center[1] = 0.5 * canvas.height // Update center Y to canvas center
	}

	/**
	 * Renders glow effect by repeatedly drawing the canvas back onto itself with filters.
	 * @function renderGlow
	 * @param {HTMLCanvasElement} canvas - Canvas element.
	 * @param {CanvasRenderingContext2D} ctx - 2D rendering context.
	 * @returns {void}
	 */
	const renderGlow = (canvas, ctx) => {
		ctx.save() // Save current canvas state
		ctx.filter = "blur(8px) brightness(200%)" // Apply blur and brightness filters for glow
		ctx.globalCompositeOperation = "lighter" // Set composite operation to lighter for additive blending
		ctx.drawImage(canvas, 0, 0) // Draw canvas onto itself to create glow effect
		ctx.restore() // Restore canvas state

		ctx.save() // Save current canvas state again for second glow pass
		ctx.filter = "blur(4px) brightness(200%)" // Apply different blur and brightness for layered glow
		ctx.globalCompositeOperation = "lighter" // Set composite operation to lighter
		ctx.drawImage(canvas, 0, 0) // Draw canvas onto itself again for stronger glow
		ctx.restore() // Restore canvas state
	}

	/**
	 * Composites the canvas render to the screen using 'lighter' blend mode.
	 * @function renderToScreen
	 * @param {HTMLCanvasElement} canvas - Canvas element.
	 * @param {CanvasRenderingContext2D} ctx - 2D rendering context.
	 * @returns {void}
	 */
	const renderToScreen = (canvas, ctx) => {
		ctx.save() // Save current canvas state
		ctx.globalCompositeOperation = "lighter" // Set global composite operation to 'lighter' for final render
		ctx.drawImage(canvas, 0, 0) // Draw canvas onto itself with lighter composite operation
		ctx.restore() // Restore canvas state
	}

	/**
	 * useEffect hook to initialize and setup the vortex animation after component mount.
	 * Also, it sets up an event listener for window resize to handle canvas resizing.
	 */
	useEffect(() => {
		setup() // Call setup function to initialize canvas and animation

		// Event listener for window resize to handle responsive canvas
		window.addEventListener("resize", () => {
			const canvas = canvasRef.current // Get canvas element from ref
			const ctx = canvas?.getContext("2d") // Get 2D rendering context

			if (canvas && ctx) {
				resize(canvas, ctx) // Resize canvas on window resize
			}
		})

		// useEffect cleanup function to remove event listener (not strictly needed in this case as component is typically not unmounted)
		return () => {
			window.removeEventListener("resize", () => {
				const canvas = canvasRef.current
				const ctx = canvas?.getContext("2d")
				if (canvas && ctx) {
					resize(canvas, ctx)
				}
			})
		}
	}, []) // Empty dependency array ensures this effect runs only once after component mount

	return (
		<div className={cn("relative h-24 w-full", props.containerClassName)}>
			{/* Container div for vortex and content, relative positioning, dynamic containerClassName */}
			<motion.div
				initial={{ opacity: 0 }} // Initial opacity for animation
				animate={{ opacity: 1 }} // Animate opacity to 1 for fade-in effect
				ref={containerRef} // Attach containerRef to this div
				className="absolute h-full w-full inset-0 z-0 bg-transparent flex items-center justify-center"
			>
				{/* Motion div for animated container: absolute positioning to cover parent, full height and width, transparent background, flex layout to center canvas, z-index 0 to stay behind content */}
				<canvas ref={canvasRef}></canvas>{" "}
				{/* Canvas element for vortex animation, ref attached */}
			</motion.div>
			{/* Div to hold content, positioned above the vortex animation */}
			<div className={cn("relative z-10", props.className)}>
				{props.children}{" "}
				{/* Render children components, positioned above vortex animation */}
			</div>
		</div>
	)
}
