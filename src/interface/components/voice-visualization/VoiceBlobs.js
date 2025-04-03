"use client"
import { useEffect, useRef, useState } from "react"
import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import React from "react"

const cn = (...inputs) => {
	return twMerge(clsx(inputs))
}

export const VoiceBlobs = ({
	audioLevel = 0,
	isActive = false,
	className,
	containerClassName
}) => {
	const layerRefs = useRef([
		useRef(null),
		useRef(null),
		useRef(null),
		useRef(null),
		useRef(null)
	]).current

	// Dark theme colors (remains same)
	const gradientBackgroundStart = "rgb(18, 18, 18)"
	const gradientBackgroundEnd = "rgb(18, 18, 18)"
	const firstColor = "30, 80, 200"
	const secondColor = "80, 30, 220"
	const thirdColor = "50, 100, 250"
	const fourthColor = "100, 50, 240"
	const fifthColor = "40, 90, 230"
	const size = "80%"
	const blendingValue = "hard-light"

	// Set CSS variables (remains same)
	useEffect(() => {
		const rootStyle = document.documentElement.style
		rootStyle.setProperty(
			"--gradient-background-start",
			gradientBackgroundStart
		)
		rootStyle.setProperty(
			"--gradient-background-end",
			gradientBackgroundEnd
		)
		rootStyle.setProperty("--first-color", firstColor)
		rootStyle.setProperty("--second-color", secondColor)
		rootStyle.setProperty("--third-color", thirdColor)
		rootStyle.setProperty("--fourth-color", fourthColor)
		rootStyle.setProperty("--fifth-color", fifthColor)
		rootStyle.setProperty("--vb-size", size)
		rootStyle.setProperty("--blending-value", blendingValue)

		return () => {
			rootStyle.removeProperty("--gradient-background-start")
			rootStyle.removeProperty("--gradient-background-end")
			rootStyle.removeProperty("--first-color")
			rootStyle.removeProperty("--second-color")
			rootStyle.removeProperty("--third-color")
			rootStyle.removeProperty("--fourth-color")
			rootStyle.removeProperty("--fifth-color")
			rootStyle.removeProperty("--vb-size")
			rootStyle.removeProperty("--blending-value")
		}
	}, [])

	// Animation loop reacting to audioLevel
	useEffect(() => {
		let animationFrameId
		let time = 0

		const animate = () => {
			time += 0.003 // Base speed

			// MODIFIED: Increased sensitivity and adjusted base scale slightly
			const baseScale = 1.0 // Base size when inactive or silent
			// MODIFIED: Increased multiplier for more significant reaction to audio
			const activeScaleMultiplier = 2.5 // Amplify the effect of audioLevel
			// MODIFIED: Made minimum scale when active slightly larger than baseScale even with zero audio
			const minActiveScale = 1.05

			// Calculate target scale factor
			let scaleFactor = baseScale
			if (isActive) {
				// Scale increases from minActiveScale based on audio level
				scaleFactor =
					minActiveScale + audioLevel * activeScaleMultiplier
			}

			// Apply smoothing individually to each layer for potentially varied reaction speeds (or keep uniform)
			layerRefs.forEach((layer, index) => {
				if (layer.current) {
					// Get current scale
					const currentTransform = layer.current.style.transform
					const match = currentTransform.match(/scale\(([^)]+)\)/)
					const currentScale = match
						? parseFloat(match[1])
						: baseScale // Start from baseScale if no transform yet

					// MODIFIED: Faster smoothing (higher factor closer to 1 means less smoothing)
					const smoothingFactor = 0.25 // (e.g., 0.1 is very smooth, 0.5 is faster)
					const smoothedScaleFactor =
						currentScale +
						(scaleFactor - currentScale) * smoothingFactor

					// --- Base movement (remains same) ---
					const x =
						50 +
						15 * Math.sin(time * (1 + index * 0.3)) +
						10 * Math.cos(time * (2 + index * 0.5))
					const y =
						50 +
						15 * Math.cos(time * (1.5 + index * 0.3)) +
						10 * Math.sin(time * (2.5 + index * 0.5))

					layer.current.style.background = `radial-gradient(circle at ${x}% ${y}%, rgba(var(${layer.current.dataset.colorvar}), 0.8) 0%, rgba(var(${layer.current.dataset.colorvar}), 0) 50%) no-repeat`
					// --- End Base movement ---

					// --- Apply Pulsation ---
					layer.current.style.transform = `scale(${smoothedScaleFactor})`
					// MODIFIED: Slightly faster transition to match faster smoothing
					layer.current.style.transition = "transform 0.08s ease-out"
					// --- End Pulsation ---
				}
			})
			animationFrameId = requestAnimationFrame(animate)
		}

		animationFrameId = requestAnimationFrame(animate)

		return () => {
			cancelAnimationFrame(animationFrameId)
		}
	}, [audioLevel, isActive]) // Rerun effect if audioLevel or isActive changes

	// Safari check (remains same)
	const [isSafari, setIsSafari] = useState(false)
	useEffect(() => {
		if (typeof navigator !== "undefined") {
			setIsSafari(
				/^((?!chrome|android).)*safari/i.test(navigator.userAgent)
			)
		}
	}, [])

	// Layer configs (remains same)
	const layerConfigs = [
		{ colorVar: "--first-color", opacity: "100" },
		{ colorVar: "--second-color", opacity: "100" },
		{ colorVar: "--third-color", opacity: "100" },
		{ colorVar: "--fourth-color", opacity: "70" },
		{ colorVar: "--fifth-color", opacity: "100" }
	]

	return (
		// Container (remains same)
		<div
			className={cn(
				"relative h-full w-full overflow-hidden top-0 left-0", // REMOVED: Background gradient here, as blobs cover it
				containerClassName
			)}
			style={{ zIndex: 0, opacity: 0.8 }}
		>
			{/* SVG filter (remains same) */}
			<svg className="hidden">
				<defs>
					<filter id="blurMe">
						<feGaussianBlur
							in="SourceGraphic"
							stdDeviation="10"
							result="blur-sm"
						/>
						<feColorMatrix
							in="blur-sm"
							mode="matrix"
							values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 18 -8"
							result="goo"
						/>
						<feBlend in="SourceGraphic" in2="goo" />
					</filter>
				</defs>
			</svg>
			{/* Gradients container (remains same) */}
			<div
				className={cn(
					"gradients-container h-full w-full blur-lg",
					isSafari ? "blur-2xl" : "[filter:url(#blurMe)_blur(40px)]"
				)}
			>
				{layerConfigs.map((config, index) => (
					<div
						key={index}
						ref={layerRefs[index]}
						data-colorvar={config.colorVar}
						// Ensure initial scale is set if needed, or rely on JS to set it immediately
						className={`absolute [mix-blend-mode:var(--blending-value)] w-[var(--vb-size)] h-[var(--vb-size)] top-[calc(50%-var(--vb-size)/2)] left-[calc(50%-var(--vb-size)/2)] opacity-${config.opacity} transform scale-100`} // Start at scale 1
						style={{
							background: `radial-gradient(circle at 50% 50%, rgba(var(${config.colorVar}), 0.8) 0%, rgba(var(${config.colorVar}), 0) 50%) no-repeat`,
							transformOrigin: "center center"
						}}
					></div>
				))}
			</div>
		</div>
	)
}

export default VoiceBlobs
