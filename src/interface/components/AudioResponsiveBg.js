"use client"
import { useEffect, useRef, useState } from "react"
import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import React from "react"

const cn = (...inputs) => {
	return twMerge(clsx(inputs))
}

export const AudioVisualizer = ({
	analyser,
	isAI = false,
	gradientBackgroundStart = "rgb(33, 33, 33)",
	gradientBackgroundEnd = "rgb(33, 33, 33)",
	firstColor = "0, 92, 254",
	secondColor = "0, 178, 254",
	thirdColor = "0, 92, 254",
	fourthColor = "0, 178, 254",
	fifthColor = "0, 92, 254",
	size = "80%",
	blendingValue = "hard-light",
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

	const animationRef = useRef(null)
	const frequencyData = useRef(null)

	// Set up frequency data if we have an analyser
	useEffect(() => {
		if (analyser) {
			frequencyData.current = new Uint8Array(analyser.frequencyBinCount)
		}
		return () => {
			if (animationRef.current) {
				cancelAnimationFrame(animationRef.current)
			}
		}
	}, [analyser])

	const colors = [
		"--first-color",
		"--second-color",
		"--third-color",
		"--fourth-color",
		"--fifth-color"
	]

	useEffect(() => {
		document.body.style.setProperty(
			"--gradient-background-start",
			gradientBackgroundStart
		)
		document.body.style.setProperty(
			"--gradient-background-end",
			gradientBackgroundEnd
		)
		document.body.style.setProperty("--first-color", firstColor)
		document.body.style.setProperty("--second-color", secondColor)
		document.body.style.setProperty("--third-color", thirdColor)
		document.body.style.setProperty("--fourth-color", fourthColor)
		document.body.style.setProperty("--fifth-color", fifthColor)
		document.body.style.setProperty("--size", size)
		document.body.style.setProperty("--blending-value", blendingValue)
	}, [])

	useEffect(() => {
		let time = 0

		const animate = () => {
			if (!analyser || !frequencyData.current) {
				// Random movement when no audio is playing
				time += 0.005
				layerRefs.forEach((layer, index) => {
					if (layer.current) {
						const x =
							50 +
							15 * Math.sin(time * (1 + index)) +
							10 * Math.cos(time * (2 + index * 0.5))
						const y =
							50 +
							15 * Math.cos(time * (1.5 + index)) +
							10 * Math.sin(time * (2.5 + index * 0.5))
						layer.current.style.background = `radial-gradient(circle at ${x}% ${y}%, rgba(var(${colors[index]}), 0.8) 0%, rgba(var(${colors[index]}), 0) 50%) no-repeat`
					}
				})
			} else {
				// Audio-reactive movement
				analyser.getByteFrequencyData(frequencyData.current)

				// Calculate the average frequency for different bands
				const bands = [
					{ start: 0, end: 5 }, // Low
					{ start: 5, end: 10 }, // Low-mid
					{ start: 10, end: 20 }, // Mid
					{ start: 20, end: 40 }, // High-mid
					{ start: 40, end: 80 } // High
				]

				const averages = bands.map((band) => {
					let sum = 0
					for (
						let i = band.start;
						i < Math.min(band.end, frequencyData.current.length);
						i++
					) {
						sum += frequencyData.current[i]
					}
					// Normalize to 0-1
					return sum / ((band.end - band.start) * 255)
				})

				// Apply the frequency data to the layers
				layerRefs.forEach((layer, index) => {
					if (layer.current) {
						// Get the normalized average for this band (0-1)
						const avg = averages[index]

						// Scale radius based on audio intensity
						const scale = 0.5 + avg * 1.5 // 0.5 to 2.0 scale

						// Calculate positions with audio influence
						// Low frequencies affect position more for deeper layers
						const wobble = (20 * avg * (5 - index)) / 5

						const x = 50 + wobble * Math.sin(time)
						const y = 50 + wobble * Math.cos(time)

						// Update the radial gradient
						layer.current.style.background = `radial-gradient(circle at ${x}% ${y}%, rgba(var(${colors[index]}), ${0.5 + avg * 0.5}) 0%, rgba(var(${colors[index]}), 0) ${50 * scale}%) no-repeat`

						// Scale the layer
						const newSize = parseInt(size) * (1 + avg * 0.3)
						layer.current.style.width = `${newSize}%`
						layer.current.style.height = `${newSize}%`
						layer.current.style.top = `calc(50% - ${newSize / 2}%)`
						layer.current.style.left = `calc(50% - ${newSize / 2}%)`
					}
				})

				// Increment time for continuous motion
				time += 0.003
			}

			animationRef.current = requestAnimationFrame(animate)
		}

		animationRef.current = requestAnimationFrame(animate)

		return () => {
			if (animationRef.current) {
				cancelAnimationFrame(animationRef.current)
			}
		}
	}, [analyser])

	const [isSafari, setIsSafari] = React.useState(false)
	useEffect(() => {
		setIsSafari(/^((?!chrome|android).)*safari/i.test(navigator.userAgent))
	}, [])

	const layerConfigs = [
		{ color: "--first-color", opacity: "100" },
		{ color: "--second-color", opacity: "100" },
		{ color: "--third-color", opacity: "100" },
		{ color: "--fourth-color", opacity: "70" },
		{ color: "--fifth-color", opacity: "100" }
	]

	// For simple frequency visualization at the bottom
	const canvasRef = useRef(null)

	useEffect(() => {
		if (!analyser || !canvasRef.current) return

		const canvas = canvasRef.current
		const ctx = canvas.getContext("2d")
		const bufferLength = analyser.frequencyBinCount
		const dataArray = new Uint8Array(bufferLength)

		const drawFrequencyBars = () => {
			if (!analyser) return

			requestAnimationFrame(drawFrequencyBars)
			analyser.getByteFrequencyData(dataArray)

			ctx.clearRect(0, 0, canvas.width, canvas.height)

			const barWidth = (canvas.width / bufferLength) * 2.5
			let x = 0

			// Use different color for AI vs user
			ctx.fillStyle = isAI ? "rgb(0, 178, 254)" : "rgb(255, 179, 71)"

			for (let i = 0; i < bufferLength; i++) {
				const barHeight = (dataArray[i] / 255) * canvas.height
				ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight)
				x += barWidth + 1
			}
		}

		drawFrequencyBars()

		return () => {
			// No cleanup needed as we're using requestAnimationFrame
		}
	}, [analyser, isAI])

	return (
		<div className="w-full rounded-xl overflow-hidden">
			{/* Traditional frequency bars visualization */}
			<canvas
				ref={canvasRef}
				width={300}
				height={50}
				className="w-full h-12 rounded-md bg-black/20 mb-2"
			/>

			{/* Gradient background */}
			<div
				className={cn(
					"relative overflow-hidden h-40 rounded-xl bg-[linear-gradient(40deg,var(--gradient-background-start),var(--gradient-background-end))]",
					containerClassName
				)}
				style={{ zIndex: 0, opacity: 0.8 }}
			>
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
				<div className={cn("h-full w-full", className)}>
					{/* You can add content here if needed */}
				</div>
				<div
					className={cn(
						"gradients-container h-full w-full blur-lg",
						isSafari
							? "blur-2xl"
							: "[filter:url(#blurMe)_blur(40px)]"
					)}
				>
					{layerConfigs.map((config, index) => (
						<div
							key={index}
							ref={layerRefs[index]}
							className={`absolute [mix-blend-mode:var(--blending-value)] w-[var(--size)] h-[var(--size)] top-[calc(50%-var(--size)/2)] left-[calc(50%-var(--size)/2)] opacity-${config.opacity} transition-all duration-200`}
							style={{
								background: `radial-gradient(circle at 50% 50%, rgba(var(${config.color}), 0.8) 0%, rgba(var(${config.color}), 0) 50%) no-repeat`
							}}
						></div>
					))}
				</div>
			</div>
		</div>
	)
}
