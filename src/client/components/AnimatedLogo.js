import { motion } from "framer-motion" // Importing the motion component from framer-motion library
import React from "react"

/**
 * AnimatedLogo Component - Renders an animated logo using framer-motion.
 *
 * This component displays a logo image and applies a scale animation to it, creating a pulsing effect.
 * It utilizes the `motion.img` component from framer-motion to easily animate the logo's scale.
 * The animation is configured to scale the logo up and down in a loop, providing a subtle, engaging visual effect.
 *
 * @returns {React.ReactNode} - The AnimatedLogo component UI, which is an animated image.
 */
const AnimatedLogo = () => {
	return (
		<motion.img
			src="/images/half-logo-dark.svg" // Path to the logo image file
			alt="Logo" // Alt text for accessibility, describing the image as "Logo"
			animate={{ scale: [1, 1.5, 1] }} // Animation definition: scales from 1 to 1.5 and back to 1
			transition={{ duration: 2, repeat: Infinity }} // Transition properties: 2 seconds duration, repeats infinitely
		/>
	)
}

export default AnimatedLogo
