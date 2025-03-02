"use client"
import { useEffect, useRef } from "react"
import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"
import React from "react"

const cn = (...inputs) => {
	return twMerge(clsx(inputs))
}

export const BackgroundGradientAnimation = ({
	gradientBackgroundStart = "rgb(33, 33, 33)",
	gradientBackgroundEnd = "rgb(33, 33, 33)",
	firstColor = "0, 92, 254",
	secondColor = "0, 178, 254",
	thirdColor = "0, 92, 254",
	fourthColor = "0, 178, 254",
	fifthColor = "0, 92, 254",
	size = "80%",
	blendingValue = "hard-light",
	children,
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
			time += 0.005 // Controls speed: smaller values = slower motion
			layerRefs.forEach((layer, index) => {
				if (layer.current) {
					// Calculate wave-like positions using sine and cosine
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
			requestAnimationFrame(animate)
		}

		requestAnimationFrame(animate)
	}, [])

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

	return (
		<div
			className={cn(
				"relative overflow-hidden top-0 left-0 bg-[linear-gradient(40deg,var(--gradient-background-start),var(--gradient-background-end))]",
				containerClassName
			)}
			style={{ zIndex: 0, opacity: 0.5 }}
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
			<div className={cn("", className)}>{children}</div>
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
						className={`absolute [mix-blend-mode:var(--blending-value)] w-[var(--size)] h-[var(--size)] top-[calc(50%-var(--size)/2)] left-[calc(50%-var(--size)/2)] opacity-${config.opacity}`}
						style={{
							background: `radial-gradient(circle at 50% 50%, rgba(var(${config.color}), 0.8) 0%, rgba(var(${config.color}), 0) 50%) no-repeat`
						}}
					></div>
				))}
			</div>
		</div>
	)
}
