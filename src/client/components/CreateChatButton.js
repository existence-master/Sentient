import { useEffect, useMemo, useState } from "react" // Importing React hooks: useEffect, useMemo, useState
import { Sparkle } from "lucide-react" // Importing Sparkle icon from lucide-react library
import { loadFull } from "tsparticles" // Importing loadFull function from tsparticles library for loading full tsparticles engine
import React from "react" // Importing React library
import Particles, { initParticlesEngine } from "@tsparticles/react" // Importing Particles component and initParticlesEngine function from @tsparticles/react

/**
 * Configuration object for tsparticles.
 * Defines the appearance and behavior of particles in the animation.
 * @type {object}
 */
const options = {
	key: "star", // Key identifier for this particles configuration
	name: "Star", // Name of this particles configuration
	particles: {
		number: {
			value: 20, // Number of particles to render
			density: {
				enable: false // Disable particle density adjustments
			}
		},
		color: {
			value: ["#00B2FE", "#005CFE", "#ffffff"] // Array of colors for particles, creating a multi-color effect
		},
		shape: {
			type: "star", // Shape of particles set to 'star'
			options: {
				star: {
					sides: 4 // Number of sides for the star shape (4-pointed star)
				}
			}
		},
		opacity: {
			value: 0.8 // Opacity of particles, 0.8 for slight transparency
		},
		size: {
			value: { min: 1, max: 4 } // Random particle size between 1px and 4px
		},
		rotate: {
			value: {
				min: 0,
				max: 360 // Rotation angle in degrees, from 0 to 360
			},
			enable: true, // Enable particle rotation
			direction: "clockwise", // Rotation direction set to clockwise
			animation: {
				enable: true, // Enable rotation animation
				speed: 10, // Rotation speed, higher value for faster rotation
				sync: false // Rotation animation of each particle is not synchronized
			}
		},
		links: {
			enable: false // Disable links between particles
		},
		reduceDuplicates: true, // Enable duplicate reduction for particle rendering
		move: {
			enable: true, // Enable particle movement
			center: {
				x: 120, // Center point X coordinate for particle movement
				y: 45 // Center point Y coordinate for particle movement
			}
		}
	},
	interactivity: {
		events: {} // No interactivity events defined
	},
	smooth: true, // Enable smooth particle rendering
	fpsLimit: 120, // Frame rate limit set to 120 FPS for smooth animation
	background: {
		color: "transparent", // Background color set to transparent
		size: "cover" // Background size set to cover
	},
	fullScreen: {
		enable: false // Full screen mode disabled
	},
	detectRetina: true, // Enable retina display detection for higher resolution rendering
	absorbers: [
		{
			enable: true, // Enable absorbers to attract particles
			opacity: 0, // Absorber opacity set to 0, making it invisible
			size: {
				value: 1, // Base size of absorber
				density: 1, // Absorber density
				limit: {
					radius: 5, // Limit radius of absorber effect
					mass: 5 // Mass of absorber
				}
			},
			position: {
				x: 110, // Absorber position X coordinate
				y: 45 // Absorber position Y coordinate
			}
		}
	],
	emitters: [
		{
			autoPlay: true, // Emitter auto play enabled
			fill: true, // Emitter fill enabled
			life: {
				wait: true // Emitter life wait enabled
			},
			rate: {
				quantity: 5, // Number of particles emitted per rate
				delay: 0.5 // Delay between emissions in seconds
			},
			position: {
				x: 110, // Emitter position X coordinate
				y: 45 // Emitter position Y coordinate
			}
		}
	]
}

/**
 * AiButton Component - Button with animated particles effect on hover.
 *
 * This component renders a button that, when hovered, displays an animated sparkle particle effect.
 * It uses the `tsparticles` library for particle animation and `lucide-react` for the sparkle icon.
 * The button is designed for interactive elements, providing visual feedback on hover and click.
 *
 * @param {object} props - Component props.
 * @param {function} props.onClick - Handler function to be called when the button is clicked.
 * @returns {React.ReactNode} - The AiButton component UI.
 */
export default function AiButton({ onClick }) {
	// State to manage the particle system readiness - particleState: "loading" | "ready" | "loaded"
	const [particleState, setParticlesReady] = useState("loaded") // Initial state set to 'loaded'
	// State to track hover status, controlling particle animation - isHovering: boolean
	const [isHovering, setIsHovering] = useState(false) // Initially set to false

	/**
	 * useEffect hook to initialize the particle engine on component mount.
	 *
	 * Loads the full tsparticles engine asynchronously and then sets the particleState to "loaded"
	 * to indicate that the particle system is initialized.
	 */
	useEffect(() => {
		// Initialize particles engine asynchronously
		initParticlesEngine(async (engine) => {
			await loadFull(engine) // Load the full tsparticles engine
		}).then(() => {
			setParticlesReady("loaded") // Set particleState to 'loaded' after engine initialization
		})
	}, []) // Empty dependency array ensures this effect runs only once on mount

	/**
	 * useMemo hook to modify particle options based on hover state.
	 *
	 * Memoizes the particle options configuration, updating the `autoPlay` property based on `isHovering` state.
	 * This optimization prevents unnecessary recalculations of options unless `isHovering` changes.
	 */
	const modifiedOptions = useMemo(() => {
		options.autoPlay = isHovering // Update autoPlay in options based on isHovering state
		return options // Return modified options
	}, [isHovering]) // Dependency array: isHovering - options are recalculated only when isHovering changes

	return (
		<button
			className="group relative my-8 rounded-md bg-lightblue p-1 text-white transition-transform hover:scale-110 active:scale-105"
			// Button styling: relative positioning, margin, rounded corners, light blue background, padding, white text, transition for hover and active states, scaling on interaction
			onMouseEnter={() => setIsHovering(true)} // Set isHovering to true on mouse enter to start particle animation
			onMouseLeave={() => setIsHovering(false)} // Set isHovering to false on mouse leave to stop particle animation
			onClick={onClick} // Calls the onClick handler passed as prop
		>
			<div className="relative flex items-center justify-center gap-2 rounded-md bg-white hover:border-lightblue px-4 py-2 text-black cursor-pointer">
				{/* Inner button content container: relative positioning, flex layout, center alignment, gap, rounded corners, white background, hover border, padding, black text */}
				<Sparkle className="size-6 -translate-y-0.5 animate-sparkle fill-white" />
				{/* Sparkle icon: size, slight vertical translation, sparkle animation, white fill color */}
				<Sparkle
					style={{
						animationDelay: "1s" // Animation delay for staggered effect
					}}
					className="absolute bottom-2.5 left-3.5 z-20 size-2 rotate-12 animate-sparkle fill-white"
					// Sparkle icon: absolute positioning, bottom-left placement, z-index to be above other sparkles, size, rotation, sparkle animation, white fill color
				/>
				<Sparkle
					style={{
						animationDelay: "1.5s", // Animation delay for staggered effect
						animationDuration: "2.5s" // Custom animation duration
					}}
					className="absolute left-5 top-2.5 size-1 -rotate-12 animate-sparkle fill-white"
					// Sparkle icon: absolute positioning, top-left placement, size, rotation, sparkle animation, white fill color, custom animation delay and duration
				/>
				<Sparkle
					style={{
						animationDelay: "0.5s", // Short animation delay for varied start times
						animationDuration: "2.5s" // Custom animation duration
					}}
					className="absolute left-3 top-3 size-1.5 animate-sparkle fill-white"
					// Sparkle icon: absolute positioning, top-left placement, size, sparkle animation, white fill color, custom animation delay and duration
				/>

				<span className="font-semibold">Create Chat</span>
				{/* Button text: bold font weight */}
			</div>
			{/* Conditional rendering of Particles component based on particleState */}
			{!!particleState && (
				<Particles
					id="whatever" // ID for the Particles component (optional)
					className={`pointer-events-none absolute -bottom-4 -left-4 -right-4 -top-4 z-0 opacity-0 transition-opacity ${
						particleState === "ready"
							? "group-hover:opacity-100"
							: ""
					}`}
					// Particles container styling: pointer-events disabled, absolute positioning to cover button, negative margins to extend beyond button bounds, z-index 0 to stay behind content, initial opacity 0, transition for opacity, dynamic class for hover effect based on particleState
					particlesLoaded={async () => {
						setParticlesReady("ready") // Update particleState to 'ready' when particles are loaded
					}}
					options={modifiedOptions} // Pass modified particle options to Particles component
				/>
			)}
		</button>
	)
}