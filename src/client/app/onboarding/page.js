"use client"
import React, { useState, useEffect, useCallback, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@utils/cn"
import toast from "react-hot-toast"
import { usePostHog } from "posthog-js/react"
import { useRouter } from "next/navigation"
import {
	IconLoader,
	IconCheck,
	IconSparkles,
	IconHeart,
	IconBrandWhatsapp
} from "@tabler/icons-react"
import InteractiveNetworkBackground from "@components/ui/InteractiveNetworkBackground"
import ProgressBar from "@components/onboarding/ProgressBar"
import SparkleEffect from "@components/ui/SparkleEffect"
import { GridBackground } from "@components/ui/GridBackground"
import { UseCaseCarousel } from "@components/onboarding/UseCaseCarousel"

// --- Helper Components ---

const TypingIndicator = () => (
	<motion.div
		initial={{ opacity: 0 }}
		animate={{ opacity: 1 }}
		exit={{ opacity: 0 }}
		className="flex items-center gap-2"
	>
		<span className="text-brand-orange">[SENTIENT]:</span>
		<motion.div
			className="w-2 h-4 bg-brand-orange"
			animate={{ opacity: [0, 1, 0] }}
			transition={{ duration: 1, repeat: Infinity, ease: "easeInOut" }}
		/>
	</motion.div>
)

const NavigationHint = ({ onBack, onNext, isNextDisabled }) => (
	<div className="mt-6 text-center">
		{/* Mobile Buttons */}
		<div className="md:hidden flex justify-center gap-4">
			<button
				onClick={onBack}
				className="py-2 px-5 rounded-md bg-neutral-700 hover:bg-neutral-600 text-sm font-semibold"
			>
				Back
			</button>
			<button
				onClick={onNext}
				disabled={isNextDisabled}
				className="py-2 px-5 rounded-md bg-brand-orange/80 hover:bg-brand-orange text-brand-black text-sm font-semibold disabled:opacity-50"
			>
				Next
			</button>
		</div>

		{/* Desktop Hint */}
		<p className="hidden md:block text-xs text-neutral-500">
			Press{" "}
			<kbd className="px-2 py-1 text-xs font-semibold text-neutral-400 bg-neutral-800 border border-neutral-700 rounded-md">
				Enter
			</kbd>{" "}
			to continue, or{" "}
			<kbd className="px-2 py-1 text-xs font-semibold text-neutral-400 bg-neutral-800 border border-neutral-700 rounded-md">
				←
			</kbd>{" "}
			to go back.
		</p>
	</div>
)

// --- Onboarding Data ---

const questions = [
	{
		id: "user-name",
		question: "First, what should I call you?",
		type: "text-input",
		required: true,
		placeholder: "e.g., Alex"
	},
	{
		id: "timezone",
		question: "What's your timezone?",
		type: "select",
		required: true,
		options: [
			{ value: "", label: "Select your timezone..." },
			{ value: "UTC", label: "(GMT+00:00) Coordinated Universal Time" },
			{
				value: "America/New_York",
				label: "(GMT-04:00) Eastern Time (US & Canada)"
			},
			{
				value: "America/Chicago",
				label: "(GMT-05:00) Central Time (US & Canada)"
			},
			{
				value: "America/Denver",
				label: "(GMT-06:00) Mountain Time (US & Canada)"
			},
			{
				value: "America/Los_Angeles",
				label: "(GMT-07:00) Pacific Time (US & Canada)"
			},
			{ value: "America/Anchorage", label: "(GMT-08:00) Alaska" },
			{ value: "America/Phoenix", label: "(GMT-07:00) Arizona" },
			{ value: "Pacific/Honolulu", label: "(GMT-10:00) Hawaii" },
			{ value: "America/Sao_Paulo", label: "(GMT-03:00) Brasilia" },
			{
				value: "America/Buenos_Aires",
				label: "(GMT-03:00) Buenos Aires"
			},
			{
				value: "Europe/London",
				label: "(GMT+01:00) London, Dublin, Lisbon"
			},
			{
				value: "Europe/Berlin",
				label: "(GMT+02:00) Amsterdam, Berlin, Paris, Rome"
			},
			{
				value: "Europe/Helsinki",
				label: "(GMT+03:00) Helsinki, Kyiv, Riga, Sofia"
			},
			{
				value: "Europe/Moscow",
				label: "(GMT+03:00) Moscow, St. Petersburg"
			},
			{ value: "Africa/Cairo", label: "(GMT+02:00) Cairo" },
			{ value: "Africa/Johannesburg", label: "(GMT+02:00) Johannesburg" },
			{ value: "Asia/Dubai", label: "(GMT+04:00) Abu Dhabi, Muscat" },
			{ value: "Asia/Kolkata", label: "(GMT+05:30) India Standard Time" },
			{
				value: "Asia/Shanghai",
				label: "(GMT+08:00) Beijing, Hong Kong, Shanghai"
			},
			{ value: "Asia/Singapore", label: "(GMT+08:00) Singapore" },
			{ value: "Asia/Tokyo", label: "(GMT+09:00) Tokyo, Seoul" },
			{
				value: "Australia/Sydney",
				label: "(GMT+10:00) Sydney, Melbourne"
			},
			{ value: "Australia/Brisbane", label: "(GMT+10:00) Brisbane" },
			{ value: "Australia/Adelaide", label: "(GMT+09:30) Adelaide" },
			{ value: "Australia/Perth", label: "(GMT+08:00) Perth" },
			{
				value: "Pacific/Auckland",
				label: "(GMT+12:00) Auckland, Wellington"
			}
		]
	},
	{
		id: "location",
		question: "Where are you located?",
		description:
			"This helps with local info like weather. You can type a city or detect it automatically.",
		type: "location",
		required: true
	},
	{
		id: "professional-context",
		question: "What is your professional background?",
		type: "textarea",
		required: true,
		placeholder: "e.g., I'm a software developer at a startup..."
	},
	{
		id: "personal-context",
		question: "What about your personal life and interests?",
		type: "textarea",
		required: true,
		placeholder: "e.g., I enjoy hiking, learning guitar, and soccer.",
		icon: <IconHeart />
	},
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
			"Please enter your WhatsApp number with the country code.",
		type: "text-input",
		required: true,
		placeholder: "+14155552671",
		icon: <IconBrandWhatsapp />
	}
]

const sentientComments = [
	"To get started, I just need to ask a few questions to personalize your experience.",
	"Great to meet you, {user-name}! To make sure I'm always on your time...",
	"Perfect. Now, to help with local info like weather and places...",
	"This helps me understand your professional goals and context.",
	"And when you're not working? Tell me about your hobbies.",
	"Got it. One more thing before we get to the last step...",
	"Finally, I will send you important notifications, task updates, and reminders on WhatsApp. We're in the process of getting an official number, so for now, messages will come from our co-founder Sarthak (+91827507823), who may also occasionally reach out for feedback.",
	"Awesome! That's all I need. Let's get you set up."
]

// --- Main Component ---

const OnboardingPage = () => {
	const [stage, setStage] = useState("intro") // 'intro', 'questions', 'submitting', 'complete'
	const [answers, setAnswers] = useState({})
	const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
	const [isLoading, setIsLoading] = useState(true)
	const [conversation, setConversation] = useState([])
	const [score, setScore] = useState(0)
	const [maxQuestionIndexReached, setMaxQuestionIndexReached] = useState(0)
	const [sparkleTrigger, setSparkleTrigger] = useState(0)
	const [isAiTyping, setIsAiTyping] = useState(false)
	const posthog = usePostHog()
	const router = useRouter()
	const chatEndRef = useRef(null)
	const statusChecked = useRef(false)

	const [locationState, setLocationState] = useState({
		loading: false,
		data: null,
		error: null
	})

	const addAiMessage = useCallback(
		(index) => {
			let message = sentientComments[index]
			if (message.includes("{user-name}")) {
				message = message.replace(
					"{user-name}",
					answers["user-name"] || "friend"
				)
			}
			// Append the question text if it exists for the current step
			if (index < questions.length) {
				const question = questions[index]
				if (question) {
					message += `\n\n${question.question}`
				}
			}

			setConversation((prev) => [
				...prev,
				{ sender: "ai", text: message }
			])
		},
		[answers]
	)

	const handleAnswer = (questionId, answer) => {
		setAnswers((prev) => ({ ...prev, [questionId]: answer }))
	}

	const handleMultiChoice = (questionId, option) => {
		const currentAnswers = answers[questionId] || []
		const limit = questions.find((q) => q.id === questionId)?.limit || 1
		let newAnswers
		if (currentAnswers.includes(option)) {
			newAnswers = currentAnswers.filter((item) => item !== option)
		} else {
			if (currentAnswers.length < limit) {
				newAnswers = [...currentAnswers, option]
			} else {
				toast.error(`You can select up to ${limit} options.`)
				newAnswers = currentAnswers
			}
		}
		setAnswers((prev) => ({ ...prev, [questionId]: newAnswers }))
	}

	const handleGetLocation = () => {
		if (navigator.geolocation) {
			setLocationState({ loading: true, data: null, error: null })
			navigator.geolocation.getCurrentPosition(
				async (position) => {
					const { latitude, longitude } = position.coords
					try {
						const response = await fetch(
							`https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}`
						)
						if (!response.ok) {
							throw new Error("Failed to fetch location details.")
						}
						const data = await response.json()
						const address = data.address
						// Construct a readable location string
						const locationString = [
							address.city || address.town || address.village,
							address.state,
							address.country
						]
							.filter(Boolean) // Remove any null/undefined parts
							.join(", ")

						if (!locationString) {
							throw new Error(
								"Could not determine location name from coordinates."
							)
						}

						// Update state with the text location
						setLocationState({
							loading: false,
							data: locationString, // Store the string
							error: null
						})
						handleAnswer("location", locationString) // Save the string
					} catch (error) {
						setLocationState({
							loading: false,
							data: null,
							error: error.message
						})
						toast.error(
							`Could not convert coordinates to location: ${error.message}`
						)
					}
				},
				(error) => {
					let userMessage =
						"An unknown error occurred while detecting your location."
					switch (error.code) {
						case error.PERMISSION_DENIED:
							userMessage =
								"Location permission denied. Please enable location access for this site in your browser settings and try again."
							break
						case error.POSITION_UNAVAILABLE:
							userMessage =
								"Location information is unavailable. This can happen if location services are turned off in your operating system (e.g., Windows or macOS). Please check your system settings and network connection."
							break
						case error.TIMEOUT:
							userMessage =
								"The request to get your location timed out. Please try again."
							break
					}
					setLocationState({
						loading: false,
						data: null,
						error: userMessage
					})
					toast.error(userMessage)
				}
			)
		}
	}

	const isCurrentQuestionAnswered = useCallback(() => {
		if (stage !== "questions" || currentQuestionIndex >= questions.length)
			return false
		const currentQuestion = questions[currentQuestionIndex]
		if (!currentQuestion.required) return true
		const answer = answers[currentQuestion.id]
		if (answer === undefined || answer === null || answer === "")
			return false
		if (Array.isArray(answer) && answer.length === 0) return false
		return true
	}, [answers, currentQuestionIndex, stage, questions.length])

	const handleSubmit = async () => {
		setStage("submitting")
		const mainOnboardingData = { ...answers }

		// Save WhatsApp number if provided
		const whatsappNumber = mainOnboardingData.whatsapp_notifications_number
		if (whatsappNumber && whatsappNumber.trim() !== "") {
			try {
				const whatsappResponse = await fetch(
					"/api/settings/whatsapp-notifications",
					{
						method: "POST",
						headers: { "Content-Type": "application/json" },
						body: JSON.stringify({
							whatsapp_notifications_number: whatsappNumber
						})
					}
				)
				if (!whatsappResponse.ok) {
					// Don't block onboarding for this, just show a toast.
					toast.error(
						"Could not save WhatsApp number, but onboarding will continue."
					)
					console.error(
						"Failed to save WhatsApp number during onboarding."
					)
				}
			} catch (error) {
				toast.error("An error occurred while saving WhatsApp number.")
				console.error(
					"Error saving WhatsApp number during onboarding:",
					error
				)
			}
		}

		try {
			const response = await fetch("/api/onboarding", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ data: mainOnboardingData })
			})
			if (!response.ok) {
				const result = await response.json()
				throw new Error(
					result.message || "Failed to save onboarding data"
				)
			}
			// Identify the user in PostHog as soon as we have their name
			posthog?.identify(
				(await (await fetch("/api/user/profile")).json()).sub, // Fetch user ID from session
				{ name: mainOnboardingData["user-name"] }
			)
			posthog?.capture("user_signed_up", {
				signup_method: "auth0", // or derive from user profile if available
				referral_source: "direct" // Placeholder, can be populated from URL params
			})
			posthog?.capture("onboarding_completed")
			router.push("/chat?show_demo=true")
		} catch (error) {
			toast.error(`Error: ${error.message}`)
			setStage("questions") // Go back to questions on error
		}
	}

	const handleNext = useCallback(() => {
		if (!isCurrentQuestionAnswered() || isAiTyping) return

		// Slice conversation history to the correct point before adding the new answer.
		// This ensures that if a user goes back and changes an answer, the subsequent
		// conversation history is cleared.
		const conversationSliceIndex = 1 + currentQuestionIndex * 2
		const currentQuestion = questions[currentQuestionIndex]
		const answer = answers[currentQuestion.id]

		// Format answer for display
		let displayAnswer = answer
		if (Array.isArray(answer)) {
			displayAnswer = answer.join(", ")
		}

		setConversation((prev) => {
			const slicedPrev = prev.slice(0, conversationSliceIndex)
			return [...slicedPrev, { sender: "user", text: displayAnswer }]
		})

		if (currentQuestionIndex >= maxQuestionIndexReached) {
			setScore((s) => s + 10)
			setMaxQuestionIndexReached(currentQuestionIndex + 1)
		}

		setSparkleTrigger((c) => c + 1)
		setIsAiTyping(true)

		setTimeout(() => {
			setIsAiTyping(false)
			if (currentQuestionIndex < questions.length - 1) {
				setCurrentQuestionIndex((prev) => prev + 1)
				addAiMessage(currentQuestionIndex + 1)
			} else {
				addAiMessage(questions.length) // Final comment
				setTimeout(handleSubmit, 1500)
			}
		}, 1500) // 1.5 second typing delay
	}, [
		currentQuestionIndex,
		answers,
		isCurrentQuestionAnswered,
		handleSubmit,
		addAiMessage,
		locationState.data,
		maxQuestionIndexReached,
		isAiTyping
	])

	const handleBack = useCallback(() => {
		if (currentQuestionIndex > 0 && !isAiTyping) {
			setCurrentQuestionIndex((prev) => {
				const newIndex = prev - 1
				// Slice conversation to remove the last user answer and the AI question that followed.
				const conversationSliceIndex = 1 + newIndex * 2
				setConversation((prevConv) =>
					prevConv.slice(0, conversationSliceIndex)
				)
				return newIndex
			})
		}
	}, [currentQuestionIndex, isAiTyping])

	// --- Effects ---

	useEffect(() => {
		if (statusChecked.current) return
		statusChecked.current = true

		const checkStatus = async () => {
			try {
				const response = await fetch("/api/user/data")
				if (!response.ok) throw new Error("Could not fetch user data.")
				const result = await response.json()
				if (result?.data?.onboardingComplete) {
					router.push("/chat?show_demo=true")
				} else {
					const firstQuestion = questions[0]?.question || ""
					setConversation([
						{
							sender: "ai",
							text: `${sentientComments[0]}\n\n${firstQuestion}`
						}
					])
					setIsLoading(false)
				}
			} catch (error) {
				toast.error(error.message)
				setIsLoading(false)
			}
		}
		checkStatus()
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [router])

	useEffect(() => {
		try {
			const userTimezone =
				Intl.DateTimeFormat().resolvedOptions().timeZone
			if (userTimezone) handleAnswer("timezone", userTimezone)
		} catch (e) {
			console.warn("Could not detect user timezone.")
		}
	}, [])

	useEffect(() => {
		const handleKeyDown = (e) => {
			if (stage === "intro" && e.key === "Enter") {
				e.preventDefault()
				setStage("questions")
			} else if (stage === "questions") {
				if (e.key === "ArrowLeft") {
					e.preventDefault()
					handleBack()
				} else if (e.key === "Enter") {
					const currentQuestion = questions[currentQuestionIndex]
					if (currentQuestion.type === "textarea" && e.shiftKey) {
						return
					}
					e.preventDefault()
					handleNext()
				}
			}
		}

		window.addEventListener("keydown", handleKeyDown)
		return () => window.removeEventListener("keydown", handleKeyDown)
	}, [stage, handleBack, handleNext, currentQuestionIndex])

	useEffect(() => {
		chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
	}, [conversation, isAiTyping])

	// --- Render Logic ---

	if (isLoading) {
		return (
			<div className="flex flex-col items-center justify-center min-h-screen bg-brand-black text-brand-white">
				<IconLoader className="w-10 h-10 animate-spin text-[var(--color-accent-blue)]" />
			</div>
		)
	}

	const renderContent = () => {
		const introVariants = {
			hidden: { opacity: 0 },
			visible: {
				opacity: 1,
				transition: {
					staggerChildren: 0.3
				}
			}
		}

		const itemVariants = {
			hidden: { y: 20, opacity: 0 },
			visible: {
				y: 0,
				opacity: 1,
				transition: { type: "spring", stiffness: 100 }
			}
		}

		switch (stage) {
			case "intro":
				return (
					<motion.div
						key="intro"
						variants={introVariants}
						initial="hidden"
						animate="visible"
						exit={{ opacity: 0, y: -20 }}
						className="text-center flex flex-col items-center"
					>
						<motion.div variants={itemVariants} className="mb-8">
							<IconSparkles
								size={80}
								className="mx-auto text-brand-orange drop-shadow-[0_0_15px_rgba(0,173,181,0.5)]"
							/>
						</motion.div>
						<motion.h1
							variants={itemVariants}
							className="text-4xl sm:text-5xl md:text-6xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-b from-neutral-50 to-neutral-400"
						>
							Welcome. I'm Sentient.
						</motion.h1>
						<motion.p
							variants={itemVariants}
							className="text-lg md:text-xl text-neutral-300 max-w-xl mx-auto"
						>
							I'm excited to get to know you.
						</motion.p>
						<motion.div
							variants={itemVariants}
							className="mt-12 flex flex-col items-center gap-4"
						>
							<motion.button
								onClick={() => {
									setStage("questions")
								}}
								className="rounded-lg bg-brand-orange px-8 py-3 text-lg font-semibold text-brand-black transition-colors hover:bg-brand-orange/90"
								whileHover={{ scale: 1.05 }}
								whileTap={{ scale: 0.95 }}
							>
								Let's Begin
							</motion.button>
						</motion.div>
					</motion.div>
				)

			case "questions":
				const currentQuestion = questions[currentQuestionIndex] ?? null
				return (
					<motion.div
						key="questions"
						initial={{ opacity: 0 }}
						animate={{ opacity: 1 }}
						exit={{ opacity: 0 }}
						className="w-full h-full flex flex-col font-mono"
					>
						<div className="pt-8 pb-4 flex-shrink-0">
							<ProgressBar
								score={score}
								totalQuestions={questions.length}
							/>
						</div>
						<div className="flex-1 w-full max-w-4xl mx-auto overflow-y-auto custom-scrollbar p-4 space-y-2 text-sm md:text-base bg-black/30 border border-brand-gray rounded-lg">
							{conversation.map((msg, index) => (
								<div key={index}>
									{msg.sender === "ai" ? (
										<p className="whitespace-pre-wrap">
											<span className="text-brand-orange">
												[SENTIENT]:
											</span>
											<span className="text-brand-white">
												{" "}
												{msg.text}
											</span>
										</p>
									) : (
										<p className="whitespace-pre-wrap">
											<span className="text-green-400">
												[YOU]:
											</span>
											<span className="text-neutral-300">
												{" "}
												{msg.text}
											</span>
										</p>
									)}
								</div>
							))}
							{isAiTyping && <TypingIndicator />}
							<div ref={chatEndRef} />
						</div>
						<div className="w-full max-w-4xl mx-auto p-4 flex-shrink-0">
							<AnimatePresence mode="wait">
								{currentQuestion && !isAiTyping && (
									<motion.div
										key={currentQuestionIndex}
										initial={{ opacity: 0, y: 20 }}
										animate={{ opacity: 1, y: 0 }}
										exit={{ opacity: 0, y: -20 }}
									>
										<div className="flex items-center gap-2">
											<span className="text-green-400">
												{">"}
											</span>
											<div className="flex-1">
												{renderInput(currentQuestion)}
											</div>
										</div>
									</motion.div>
								)}
							</AnimatePresence>
							{currentQuestion && !isAiTyping && (
								<NavigationHint
									onBack={handleBack}
									onNext={handleNext}
									isNextDisabled={
										!isCurrentQuestionAnswered()
									}
								/>
							)}
						</div>
					</motion.div>
				)

			case "submitting":
				return (
					<motion.div
						key="submitting"
						initial={{ opacity: 0 }}
						animate={{ opacity: 1 }}
						exit={{ opacity: 0 }}
						className="text-center"
					>
						<IconLoader className="w-16 h-16 animate-spin text-brand-orange mx-auto mb-6" />
						<h1 className="text-3xl font-bold">
							Personalizing your experience...
						</h1>
					</motion.div>
				)

			case "whatsNext":
				// This stage is removed as per the simplified flow.
				// The app will redirect directly after submitting.
				return null

			case "complete":
				return (
					<motion.div
						key="complete"
						initial={{ opacity: 0, y: 20 }}
						animate={{ opacity: 1, y: 0 }}
						className="text-center"
					>
						<IconCheck className="w-24 h-24 text-brand-green mx-auto mb-6" />
						<h1 className="text-5xl font-bold mb-4">
							All Set, {answers["user-name"] || "Friend"}!
						</h1>
						<p className="text-xl text-neutral-400">
							Your personal AI companion is ready.
						</p>
						<p className="text-lg text-neutral-500 mt-4">
							Redirecting you to home...
						</p>
					</motion.div>
				)

			default:
				return null
		}
	}

	const renderInput = (currentQuestion) => {
		switch (currentQuestion.type) {
			case "text-input":
				return (
					<input
						type="text"
						value={answers[currentQuestion.id] || ""}
						onChange={(e) =>
							handleAnswer(currentQuestion.id, e.target.value)
						}
						placeholder={currentQuestion.placeholder}
						required={currentQuestion.required}
						autoFocus
						className="w-full px-4 py-2 bg-transparent text-brand-white placeholder:text-neutral-500 focus:ring-0 border-none p-0"
					/>
				)
			case "select":
				return (
					<select
						value={answers[currentQuestion.id] || ""}
						onChange={(e) =>
							handleAnswer(currentQuestion.id, e.target.value)
						}
						required={currentQuestion.required}
						className="w-full px-4 py-2 bg-transparent text-brand-white placeholder:text-neutral-500 focus:ring-0 border-none p-0 appearance-none"
					>
						{currentQuestion.options.map((option) => (
							<option
								key={option.value}
								value={option.value}
								disabled={option.disabled}
								className="bg-brand-gray text-brand-white"
							>
								{option.label}
							</option>
						))}
					</select>
				)
			case "textarea":
				return (
					<textarea
						value={answers[currentQuestion.id] || ""}
						onChange={(e) =>
							handleAnswer(currentQuestion.id, e.target.value)
						}
						className="w-full px-4 py-2 bg-transparent text-brand-white placeholder:text-neutral-500 focus:ring-0 border-none p-0 resize-none"
						placeholder={currentQuestion.placeholder}
						autoFocus
						rows={1}
					/>
				)
			case "location":
				return (
					<div className="flex flex-col sm:flex-row gap-4 items-start">
						<input
							type="text"
							placeholder="Enter Locality, City, State..."
							value={
								typeof answers[currentQuestion.id] === "string"
									? answers[currentQuestion.id]
									: ""
							}
							onChange={(e) =>
								handleAnswer("location", e.target.value)
							}
							className="w-full px-4 py-2 bg-transparent text-brand-white placeholder:text-neutral-500 focus:ring-0 border-none p-0"
						/>
						<button
							type="button"
							onClick={handleGetLocation}
							disabled={locationState.loading}
							className="translate-y-3 text-sm text-center text-brand-orange hover:underline whitespace-nowrap"
						>
							{locationState.loading
								? "Detecting..."
								: "or [Detect Current Location]"}
						</button>
						{locationState.data && !isAiTyping && (
							<p className="text-sm text-green-400">
								Location captured!
							</p>
						)}
					</div>
				)
			case "yes-no":
				return (
					<div className="flex gap-4">
						<button
							onClick={() => {
								handleAnswer(currentQuestion.id, "yes")
								setTimeout(handleNext, 100)
							}}
							className={cn(
								"px-6 py-2 rounded-lg font-semibold transition-colors",
								answers[currentQuestion.id] === "yes"
									? "bg-brand-orange text-brand-black"
									: "bg-neutral-700 hover:bg-neutral-600"
							)}
						>
							Yes
						</button>
						<button
							onClick={() => {
								handleAnswer(currentQuestion.id, "no")
								setTimeout(handleNext, 100)
							}}
							className={cn(
								"px-6 py-2 rounded-lg font-semibold transition-colors",
								answers[currentQuestion.id] === "no"
									? "bg-brand-orange text-brand-black"
									: "bg-neutral-700 hover:bg-neutral-600"
							)}
						>
							No
						</button>
					</div>
				)
			default:
				return null
		}
	}

	return (
		<div className="grid md:grid-cols-20 min-h-screen w-full text-brand-white overflow-hidden">
			{/* Left Column: Onboarding Flow */}
			<div className="relative flex flex-col items-center justify-center w-full p-4 sm:p-8 overflow-hidden md:col-span-13">
				<div className="absolute inset-0 z-[-1]">
					<InteractiveNetworkBackground />
				</div>
				<div className="absolute -top-[250px] left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-brand-orange/10 rounded-full blur-3xl -z-10" />
				<div className="relative z-10 w-full h-full flex flex-col items-center justify-center">
					<SparkleEffect trigger={sparkleTrigger} />
					<AnimatePresence mode="wait">
						{renderContent()}
					</AnimatePresence>
				</div>
			</div>

			{/* Right Column: Use Case Carousel (Desktop Only) */}
			<div className="hidden md:flex flex-col items-center justify-center relative md:col-span-7">
				<GridBackground>
					<motion.div
						initial={{ opacity: 0, scale: 0.9 }}
						animate={{ opacity: 1, scale: 1 }}
						transition={{
							duration: 0.5,
							delay: 0.2,
							ease: "easeOut"
						}}
					>
						<UseCaseCarousel />
					</motion.div>
				</GridBackground>
			</div>
		</div>
	)
}

export default OnboardingPage
