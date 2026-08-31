"use client"
import React, { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import SiriSpheres from "@components/voice/SiriSpheres"
import FloatingIcons from "./FloatingIcons"
import InteractiveNetworkBackground from "@components/ui/InteractiveNetworkBackground"
import { IconArrowRight } from "@tabler/icons-react"
import { Button } from "@components/ui/button"

const introStages = [
	{
		id: "intro",
		text: "Hi. I'm <strong>BUYASOUL</strong>.",
		iconStage: null,
		duration: 1500
	},
	{
		id: "communication",
		text: "Text or call me...",
		iconStage: "communication",
		duration: 1000
	},
	{
		id: "work",
		text: "and I can get work done for you...",
		iconStage: "work",
		duration: 1000
	},
	{
		id: "integrations",
		text: "across your apps...",
		iconStage: "integrations",
		duration: 1000
	},
	{
		id: "memory",
		text: "while learning about you.",
		iconStage: "memory",
		duration: 1000
	}
]

const IntroSequence = ({ onComplete }) => {
	const [currentStageIndex, setCurrentStageIndex] = useState(0)
	const [audioLevel, setAudioLevel] = useState(0.1)
	const [sphereReacting, setSphereReacting] = useState(false)
	const [showNextButton, setShowNextButton] = useState(false)
	const [allVisitedStages, setAllVisitedStages] = useState([])

	const currentStage = introStages[currentStageIndex]

	// Gentle pulse animation for SiriSpheres
	useEffect(() => {
		let interval
		if (!sphereReacting) {
			interval = setInterval(() => {
				setAudioLevel(Math.sin(Date.now() / 500) * 0.1 + 0.15)
			}, 50)
		}
		return () => clearInterval(interval)
	}, [sphereReacting])

	// Handle stage progression
	const goToNextStage = () => {
		// Trigger sphere reaction
		setSphereReacting(true)
		setAudioLevel(0.8) // High impact

		setTimeout(() => {
			setSphereReacting(false)

			if (currentStageIndex < introStages.length - 1) {
				const nextIndex = currentStageIndex + 1
				setCurrentStageIndex(nextIndex)

				// Add the new stage to visited stages
				const nextStage = introStages[nextIndex]
				if (nextStage.iconStage) {
					setAllVisitedStages((prev) => [
						...prev,
						nextStage.iconStage
					])
				}
			}
		}, 600)
	}

	// Show the next button after a delay
	useEffect(() => {
		const buttonDelay = introStages[currentStageIndex].duration
		const timer = setTimeout(() => {
			setShowNextButton(true)
		}, buttonDelay)

		return () => {
			clearTimeout(timer)
			setShowNextButton(false) // Hide button immediately on stage change
		}
	}, [currentStageIndex])

	const handleContinue = () => {
		setSphereReacting(true)
		setAudioLevel(0.8)

		setTimeout(() => {
			onComplete()
		}, 300)
	}

	return (
		<motion.div
			key="intro-sequence"
			className="relative w-full h-screen text-center overflow-hidden"
			initial={{ opacity: 0 }}
			animate={{ opacity: 1 }}
			exit={{ opacity: 0 }}
		>
			<InteractiveNetworkBackground />
			<FloatingIcons
				currentStage={currentStage.iconStage}
				allStages={allVisitedStages}
			/>

			{/* SiriSpheres Container - centered and behind text */}
			<div className="absolute inset-0 -top-[10vh] flex items-center justify-center pointer-events-none">
				<motion.div
					layoutId="onboarding-sphere"
					className="w-[300px] h-[300px] md:w-[450px] md:h-[450px] flex items-center justify-center"
					animate={{
						y: sphereReacting ? 0 : [0, -15, 0]
					}}
					transition={{
						y: sphereReacting
							? { duration: 0.3, ease: "easeOut" }
							: {
									duration: 4,
									repeat: Infinity,
									ease: "easeInOut"
								}
					}}
				>
					<SiriSpheres status="connected" audioLevel={audioLevel} />
				</motion.div>
			</div>

			{/* Text Content Container - positioned at the bottom */}
			<div className="absolute bottom-0 left-0 right-0 w-full flex flex-col items-center justify-center px-4 pb-16 md:pb-24">
				<div className="min-h-[14rem] flex flex-col items-center justify-center">
					<AnimatePresence mode="wait">
						<motion.div
							key={currentStageIndex}
							initial={{ opacity: 0, y: 20 }}
							animate={{ opacity: 1, y: 0 }}
							exit={{ opacity: 0, y: -20 }}
							transition={{ duration: 0.5 }}
							className="text-3xl md:text-5xl font-medium text-neutral-200"
							dangerouslySetInnerHTML={{
								__html: currentStage.text
							}}
						/>
					</AnimatePresence>

					{/* Continue Button */}
					<AnimatePresence>
						{showNextButton && (
							<Button
								initial={{ opacity: 0, y: 20 }}
								animate={{ opacity: 1, y: 0 }}
								exit={{ opacity: 0, y: -20 }}
								onClick={
									currentStageIndex === introStages.length - 1
										? handleContinue
										: goToNextStage
								}
								className="mt-8 text-base md:text-lg bg-brand-orange text-brand-black font-semibold hover:bg-brand-orange/90 flex items-center gap-2"
								size="lg"
							>
								<span>
									{currentStageIndex ===
									introStages.length - 1
										? "Let's get started"
										: "Next"}
								</span>
								<IconArrowRight size={22} strokeWidth={2.5} />
							</Button>
						)}
					</AnimatePresence>
				</div>
			</div>
		</motion.div>
	)
}

export default IntroSequence
