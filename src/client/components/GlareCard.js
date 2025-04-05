import { useRef } from "react" // Importing useRef hook from React
import React from "react"

/**
 * GlareCard Component - A card component with a dynamic glare effect.
 *
 * This component creates a card with a 3D glare effect that responds to mouse movement.
 * It uses inline styles and CSS variables to achieve the dynamic lighting and rotation effects.
 * The glare effect is implemented using radial gradients and blend modes, creating a visually engaging UI element.
 *
 * @param {object} props - Component props.
 * @param {React.ReactNode} props.children - The content to be rendered inside the GlareCard.
 * @param {string} props.className - Optional CSS class names to apply to the GlareCard container.
 * @returns {React.ReactNode} - The GlareCard component UI.
 */
export const GlareCard = ({ children }) => {
	const isPointerInside = useRef(false) // useRef to track if the pointer is inside the card - isPointerInside: React.RefObject<boolean>
	const refElement = useRef(null) // useRef to get the DOM element of the card container - refElement: React.RefObject<HTMLDivElement>
	// useRef to hold state values for glare, background, and rotation, avoiding re-renders - state: React.RefObject<{ glare: { x: number; y: number }; background: { x: number; y: number }; rotate: { x: number; y: number } }>
	const state = useRef({
		glare: {
			x: 50, // Initial glare X position
			y: 50 // Initial glare Y position
		},
		background: {
			x: 50, // Initial background X position
			y: 50 // Initial background Y position
		},
		rotate: {
			x: 0, // Initial rotation X value
			y: 0 // Initial rotation Y value
		}
	})

	// Inline styles for the container using CSS variables for dynamic updates
	const containerStyle = {
		"--m-x": "50%", // Initial mouse X position for glare effect
		"--m-y": "50%", // Initial mouse Y position for glare effect
		"--r-x": "0deg", // Initial rotation X value
		"--r-y": "0deg", // Initial rotation Y value
		"--bg-x": "50%", // Initial background X position
		"--bg-y": "50%", // Initial background Y position
		"--duration": "300ms", // Transition duration for effects
		"--foil-size": "100%", // Foil size for background pattern
		"--opacity": "0", // Initial opacity for glare effect
		"--radius": "48px", // Border radius for card
		"--easing": "ease", // Transition easing function
		"--transition": "var(--duration) var(--easing)" // Combined transition property
	}

	// Inline styles for the background layers using CSS variables for dynamic backgrounds and blend modes
	const backgroundStyle = {
		"--step": "5%", // Step size for rainbow gradient
		"--foil-svg": `url("data:image/svg+xml,%3Csvg width='26' height='26' viewBox='0 0 26 26' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M2.99994 3.419C2.99994 3.419 21.6142 7.43646 22.7921 12.153C23.97 16.8695 3.41838 23.0306 3.41838 23.0306' stroke='blue' stroke-width='5' stroke-miterlimit='3.86874' stroke-linecap='round' style='mix-blend-mode:darken'/%3E%3C/svg%3E")`, // SVG data URL for foil pattern
		"--pattern": "var(--foil-svg) center/100% no-repeat", // Background pattern using foil SVG
		"--rainbow":
			"repeating-linear-gradient( 0deg,rgb(0,0,255) calc(var(--step) * 1),rgba(0,191,255,1) calc(var(--step) * 2),rgba(30,144,255,1) calc(var(--step) * 3),rgba(0,191,255,1) calc(var(--step) * 4),rgba(0,0,255,1) calc(var(--step) * 5),rgb(0,0,255) calc(var(--step) * 6),rgb(0,0,255) calc(var(--step) * 7) ) 0% var(--bg-y)/200% 700% no-repeat", // Rainbow gradient background
		"--diagonal":
			"repeating-linear-gradient( 128deg,#0e152e 0%,hsl(180,10%,60%) 3.8%,hsl(180,10%,60%) 4.5%,hsl(180,10%,60%) 5.2%,#0e152e 10%,#0e152e 12% ) var(--bg-x) var(--bg-y)/300% no-repeat", // Diagonal linear gradient background
		"--shade":
			"radial-gradient( farthest-corner circle at var(--m-x) var(--m-y),rgba(255,255,255,0.1) 12%,rgba(255,255,255,0.15) 20%,rgba(255,255,255,0.25) 120% ) var(--bg-x) var(--bg-y)/300% no-repeat", // Radial gradient for shade effect
		backgroundBlendMode: "hue, hue, hue, overlay" // Background blend modes for layered effects
	}

	/**
	 * Updates CSS variables based on mouse position to create glare effect.
	 *
	 * This function updates the CSS custom properties (variables) on the GlareCard element
	 * to dynamically adjust the glare and background positions, as well as the rotation of the card,
	 * based on the current mouse position.
	 *
	 * @function updateStyles
	 * @returns {void}
	 */
	const updateStyles = () => {
		if (refElement.current) {
			const { background, rotate, glare } = state.current // Destructure state values
			refElement.current?.style.setProperty("--m-x", `${glare.x}%`) // Update mouse X position CSS variable
			refElement.current?.style.setProperty("--m-y", `${glare.y}%`) // Update mouse Y position CSS variable
			refElement.current?.style.setProperty("--r-x", `${rotate.x}deg`) // Update rotation X CSS variable
			refElement.current?.style.setProperty("--r-y", `${rotate.y}deg`) // Update rotation Y CSS variable
			refElement.current?.style.setProperty("--bg-x", `${background.x}%`) // Update background X position CSS variable
			refElement.current?.style.setProperty("--bg-y", `${background.y}%`) // Update background Y position CSS variable
		}
	}

	return (
		<div
			style={containerStyle} // Apply container inline styles
			className="relative isolate [contain:layout_style] [perspective:600px] transition-transform duration-[var(--duration)] ease-[var(--easing)] delay-[var(--delay)] will-change-transform w-[320px] [aspect-ratio:17/21]"
			// Base container classes: relative positioning, isolation, layout containment, perspective for 3D, transitions, width, aspect ratio
			ref={refElement} // Attach refElement to this div
			onPointerMove={(event) => {
				// Pointer move event handler for dynamic glare update
				const rotateFactor = 0.4 // Rotation factor to control rotation intensity
				const rect = event.currentTarget.getBoundingClientRect() // Get bounding rectangle of the current target element
				const position = {
					x: event.clientX - rect.left, // Mouse X position relative to the card
					y: event.clientY - rect.top // Mouse Y position relative to the card
				}
				const percentage = {
					x: (100 / rect.width) * position.x, // Mouse X position as percentage of card width
					y: (100 / rect.height) * position.y // Mouse Y position as percentage of card height
				}
				const delta = {
					x: percentage.x - 50, // Delta X from center
					y: percentage.y - 50 // Delta Y from center
				}

				// Update state values based on mouse position
				const { background, rotate, glare } = state.current
				background.x = 50 + percentage.x / 4 - 12.5 // Calculate background X position
				background.y = 50 + percentage.y / 3 - 16.67 // Calculate background Y position
				rotate.x = -(delta.x / 3.5) // Calculate rotation X value
				rotate.y = delta.y / 2 // Calculate rotation Y value
				rotate.x *= rotateFactor // Apply rotation factor to X rotation
				rotate.y *= rotateFactor // Apply rotation factor to Y rotation
				glare.x = percentage.x // Set glare X position
				glare.y = percentage.y // Set glare Y position

				updateStyles() // Call updateStyles to apply changes to CSS variables
			}}
			onPointerEnter={() => {
				// Pointer enter event handler
				isPointerInside.current = true // Set isPointerInside to true
				if (refElement.current) {
					setTimeout(() => {
						if (isPointerInside.current) {
							refElement.current?.style.setProperty(
								"--duration",
								"0s"
							) // Set transition duration to 0s for immediate effect
						}
					}, 300) // Timeout to delay setting duration to 0s
				}
			}}
			onPointerLeave={() => {
				// Pointer leave event handler
				isPointerInside.current = false // Set isPointerInside to false
				if (refElement.current) {
					refElement.current.style.removeProperty("--duration") // Remove duration property to restore default transition
					refElement.current?.style.setProperty("--r-x", `0deg`) // Reset rotation X to 0deg
					refElement.current?.style.setProperty("--r-y", `0deg`) // Reset rotation Y to 0deg
				}
			}}
		>
			<div className="h-full grid will-change-transform origin-center transition-transform duration-[var(--duration)] ease-[var(--easing)] delay-[var(--delay)] [transform:rotateY(var(--r-x))_rotateX(var(--r-y))] rounded-[var(--radius)] border border-slate-800 hover:[--opacity:0.6] hover:[--duration:200ms] hover:[--easing:linear] hover:filter-none overflow-hidden pointer-events-none">
				{/* Inner card container: full height, grid layout, transform will change, transform origin, transitions, rotation transforms from CSS vars, rounded borders, border style, hover effects, overflow hidden, pointer events none */}
				<div
					className="w-full h-full grid [grid-area:1/1] mix-blend-soft-light [clip-path:inset(0_0_0_0_round_var(--radius))] pointer-events-none"
					// Content container: full width and height, grid layout, blend mode, clip path for rounded corners, pointer events none
				>
					<div className="relative z-10 pointer-events-auto">
						{children}
					</div>
					{/* Children container: relative positioning, z-index to be above other layers, pointer events auto to capture interactions */}
				</div>
				<div
					className="w-full h-full grid [grid-area:1/1] mix-blend-soft-light [clip-path:inset(0_0_1px_0_round_var(--radius))] opacity-[var(--opacity)] transition-opacity transition-background duration-[var(--duration)] ease-[var(--easing)] delay-[var(--delay)] will-change-background [background:radial-gradient(farthest-corner_circle_at_var(--m-x)_var(--m-y),_rgba(255,255,255,0.8)_10%,_rgba(255,255,255,0.65)_20%,_rgba(255,255,255,0)_90%)] pointer-events-none"
					// Glare effect layer: full width and height, grid layout, blend mode, clip path for rounded corners, opacity from CSS var, transitions, background radial gradient for glare, pointer events none
				/>
				<div
					className="w-full h-full grid [grid-area:1/1] mix-blend-color-dodge opacity-[var(--opacity)] will-change-background transition-opacity [clip-path:inset(0_0_1px_0_round_var(--radius))] [background-blend-mode:hue_hue_hue_overlay] [background:var(--pattern),_var(--rainbow),_var(--diagonal),_var(--shade)] relative pointer-events-none"
					style={{ ...backgroundStyle }} // Apply background inline styles
					// Background layers container: full width and height, grid layout, blend mode, opacity from CSS var, background blend modes, multiple backgrounds from CSS vars, relative positioning, pointer events none
				/>
			</div>
		</div>
	)
}
