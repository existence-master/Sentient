"use client"

import React, { useState, useRef } from "react"
import { motion } from "framer-motion"
import { cn } from "@utils/cn"
import toast from "react-hot-toast"
import { useRouter } from "next/navigation"
import {
	IconBrandWhatsapp,
	IconLoader,
	IconSparkles,
	IconCheck,
	IconX
} from "@tabler/icons-react"
import InteractiveNetworkBackground from "@components/ui/InteractiveNetworkBackground"

const questions = [
	{
		id: "needs-pa",
		question:
			"Do you often juggle multiple priorities, manage a small team, lead projects, and handle countless day-to-day tasks on your own? Many professionals spend too much time scheduling meetings, organizing their calendar, responding to emails, and doing other administrative work that eats into their day. Do you ever wish you had someone to take these repetitive tasks off your plate?\n\nDo you ever feel the need for a personal assistant?",
		type: "yes-no",
		required: true
	},
	{
		id: "whatsapp_notifications_number",
		question:
			"I will send you important notifications, task updates, and reminders on WhatsApp. We're in the process of getting an official number, so for now, messages will come from our co-founder Sarthak (+91827507823), who may also occasionally reach out for feedback. \n\nPlease enter your number with the country code.",
		type: "text-input",
		required: true,
		placeholder: "+14155552671",
		icon: <IconBrandWhatsapp />
	}
]

const CompleteProfilePage = () => {
	const [answers, setAnswers] = useState({})
	const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
	const [isSubmitting, setIsSubmitting] = useState(false)
	const router = useRouter()
	const [whatsappStatus, setWhatsappStatus] = useState("idle") // idle, checking, valid, invalid
	const [whatsappError, setWhatsappError] = useState("")
	const debounceTimeoutRef = useRef(null)

	const verifyWhatsappNumber = async (number) => {
		if (!/^\+[1-9]\d{1,14}$/.test(number.trim())) {
			setWhatsappStatus("invalid")
			setWhatsappError(
				"Please use E.164 format with country code (e.g., +14155552671)."
			)
			return
		}
		setWhatsappStatus("checking")
		setWhatsappError("")
		try {
			const response = await fetch("/api/settings/whatsapp-notifications/verify", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ phone_number: number })
			})
			const result = await response.json()
			if (!response.ok) {
				throw new Error(result.detail || "Verification request failed.")
			}
			if (result.numberExists) {
				setWhatsappStatus("valid")
				setWhatsappError("")
			} else {
				setWhatsappStatus("invalid")
				setWhatsappError("This number does not appear to be on WhatsApp.")
			}
		} catch (error) {
			setWhatsappStatus("invalid")
			setWhatsappError(error.message)
		}
	}

	const handleAnswer = (questionId, answer) => {
		setAnswers((prev) => ({ ...prev, [questionId]: answer }))
		if (questionId === "whatsapp_notifications_number") {
			setWhatsappStatus("idle")
			if (debounceTimeoutRef.current) {
				clearTimeout(debounceTimeoutRef.current)
			}
			if (answer.trim()) {
				debounceTimeoutRef.current = setTimeout(() => {
					verifyWhatsappNumber(answer)
				}, 800)
			} else {
				setWhatsappError("")
			}
		}
	}

	const handleNext = () => {
		if (currentQuestionIndex < questions.length - 1) {
			setCurrentQuestionIndex((prev) => prev + 1)
		}
	}

	const handleSubmit = async () => {
		if (!answers["needs-pa"] || !answers["whatsapp_notifications_number"]) {
			toast.error("Please answer all questions to continue.")
			return
		}
		if (whatsappStatus !== "valid") {
			toast.error("Please provide a valid WhatsApp number to continue.")
			return
		}
		setIsSubmitting(true)
		try {
			const response = await fetch("/api/settings/complete-profile", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(answers)
			})
			if (!response.ok) {
				const result = await response.json()
				throw new Error(result.detail || "Failed to update profile.")
			}
			toast.success("Thank you! Your profile is now complete.")
			window.location.href = "/chat"
		} catch (error) {
			toast.error(`Error: ${error.message}`)
		} finally {
			setIsSubmitting(false)
		}
	}

	const currentQuestion = questions[currentQuestionIndex]

	return (
		<div className="relative flex flex-col items-center justify-center min-h-screen w-full p-4 sm:p-8 text-white overflow-hidden">
			<div className="absolute inset-0 z-[-1]">
				<InteractiveNetworkBackground />
			</div>
			<motion.div
				key="complete-profile"
				initial={{ opacity: 0, y: 20 }}
				animate={{ opacity: 1, y: 0 }}
				className="relative z-10 w-full max-w-2xl text-center"
			>
				<IconSparkles
					size={60}
					className="mx-auto text-brand-orange mb-4"
				/>
				<h1 className="text-3xl sm:text-4xl font-bold mb-2">
					Just a quick update...
				</h1>
				<p className="text-neutral-300 mb-8">
					We've added some new features and need a couple more details
					to personalize your experience.
				</p>

				<div className="bg-neutral-900/50 border border-neutral-700/50 rounded-2xl p-6 sm:p-8 text-left space-y-6">
					{currentQuestion.id === "needs-pa" ? (
						<div className="text-neutral-200 space-y-4">
							<p>
								Are you someone who finds themselves spending
								too much time on administrative work? For
								example:
							</p>
							<ul className="list-disc list-inside pl-4 space-y-2 text-neutral-300">
								<li>Juggling multiple priorities</li>
								<li>
									Managing a small team or leading projects
								</li>
								<li>
									Scheduling meetings and organizing calendars
								</li>
								<li>Responding to routine emails</li>
							</ul>
							<p className="font-semibold pt-2">
								Do you ever feel the need for a personal
								assistant (human or AI) to handle these
								repetitive tasks?
							</p>
						</div>
					) : (
						<p className="whitespace-pre-wrap text-neutral-200">
							{currentQuestion.question}
						</p>
					)}

					{currentQuestion.type === "yes-no" && (
						<div className="flex gap-4 pt-2">
							<button
								onClick={() => {
									handleAnswer(currentQuestion.id, "yes")
									setTimeout(handleNext, 100)
								}}
								className="px-6 py-2 rounded-lg font-semibold bg-neutral-700 hover:bg-neutral-600"
							>
								Yes
							</button>
							<button
								onClick={() => {
									handleAnswer(currentQuestion.id, "no")
									setTimeout(handleNext, 100)
								}}
								className="px-6 py-2 rounded-lg font-semibold bg-neutral-700 hover:bg-neutral-600"
							>
								No
							</button>
						</div>
					)}

					{currentQuestion.type === "text-input" && (
						<div className="relative pt-2">
							<div className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-400">
								{currentQuestion.icon}
							</div>
							<input
								type="text"
								value={answers[currentQuestion.id] || ""}
								onChange={(e) =>
									handleAnswer(
										currentQuestion.id,
										e.target.value
									)
								}
								placeholder={currentQuestion.placeholder}
								className="w-full pl-10 pr-10 py-3 bg-neutral-800 border border-neutral-700 rounded-lg focus:ring-2 focus:ring-brand-orange"
								autoFocus
							/>
							{currentQuestion.id ===
								"whatsapp_notifications_number" && (
								<div className="absolute right-3 top-1/2 -translate-y-1/2">
									{whatsappStatus === "checking" && (
										<IconLoader
											size={18}
											className="animate-spin text-neutral-400"
										/>
									)}
									{whatsappStatus === "valid" && (
										<IconCheck
											size={18}
											className="text-green-500"
										/>
									)}
									{whatsappStatus === "invalid" && (
										<IconX
											size={18}
											className="text-red-500"
										/>
									)}
								</div>
							)}
						</div>
					)}
					{currentQuestion.id === "whatsapp_notifications_number" &&
						whatsappStatus === "invalid" &&
						whatsappError && (
							<p className="text-red-500 text-sm mt-2">
								{whatsappError}
							</p>
						)}
				</div>

				<div className="mt-8">
					<button
						onClick={handleSubmit}
						disabled={
							isSubmitting ||
							currentQuestionIndex !== questions.length - 1 ||
							whatsappStatus !== "valid"
						}
						className="w-full max-w-xs py-3 px-6 rounded-lg bg-brand-orange text-brand-black font-semibold text-lg transition-all hover:bg-brand-orange/90 disabled:opacity-50 disabled:cursor-not-allowed"
					>
						{isSubmitting ? (
							<IconLoader className="animate-spin mx-auto" />
						) : (
							"Continue to App"
						)}
					</button>
				</div>
			</motion.div>
		</div>
	)
}

export default CompleteProfilePage
