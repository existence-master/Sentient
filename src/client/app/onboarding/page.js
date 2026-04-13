"use client"
import React, { useState, useEffect, useCallback, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@utils/cn"
import toast from "react-hot-toast"
import { usePostHog } from "posthog-js/react"
import { useRouter } from "next/navigation"
import {
	IconSparkles,
	IconHeart,
	IconLoader,
	IconCheck,
	IconBrain
} from "@tabler/icons-react"
import InteractiveNetworkBackground from "@components/ui/InteractiveNetworkBackground"
import { useMutation } from "@tanstack/react-query"
import { useUserStore } from "@stores/app-stores"
import ProgressBar from "@components/onboarding/ProgressBar" // Assuming this component exists
import SparkleEffect from "@components/ui/SparkleEffect"
import SiriSpheres from "@components/voice/SiriSpheres"
import IntroSequence from "@components/onboarding/IntroSequence"
import { Button } from "@components/ui/button"
import { Input } from "@components/ui/input"
import { Select } from "@components/ui/select"
import { Textarea } from "@components/ui/textarea"

// Standard typography styles for questions
const questionStyles = {
	title: "text-xl md:text-2xl text-white font-medium leading-relaxed",
	description:
		"text-sm md:text-base text-neutral-400 mt-3 max-w-2xl mx-auto leading-relaxed",
	container:
		"min-h-[100px] md:min-h-[120px] flex items-center justify-center w-full"
}

// --- Onboarding Data ---

const questions = [
	{
		id: "user-name",
		question: "First, what should I call you?",
		type: "text-input",
		required: true,
		placeholder: "Your name"
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
		question: "Tell me about your professional background",
		type: "textarea",
		required: true,
		placeholder: "e.g., I'm a software developer at a startup..."
	},
	{
		id: "working-hours",
		question: "What are your usual working hours?",
		description: "This helps me know when to proactively reach out",
		type: "text-input",
		required: false,
		placeholder: "e.g., Mon-Fri, 9 AM to 6 PM"
	},
	{
		id: "key-people",
		question: "Who are the key people in your life I should remember?",
		description:
			"Family members, colleagues, or assistants I should know about",
		type: "textarea",
		required: false,
		placeholder: "e.g., Jane Doe - spouse, John Smith - assistant"
	},
	{
		id: "personal-context",
		question: "Any personal details you'd like me to remember?",
		description:
			"Birthdays, anniversaries, preferences, or anything important to you",
		type: "textarea",
		required: false,
		placeholder: "e.g., My anniversary is on June 5th. I love Italian food."
	}
]

// --- Main Component ---

const OnboardingPage = () => {
	const [stage, setStage] = useState("intro") // 'intro', 'questions', 'submitting', 'complete'
	const [answers, setAnswers] = useState({})
	const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
	const [isLoading, setIsLoading] = useState(true)
	const [score, setScore] = useState(0)
	const [sparkleTrigger, setSparkleTrigger] = useState(0)
	const posthog = usePostHog()
	const { fetchUserData } = useUserStore()
	const router = useRouter()
	const statusChecked = useRef(false)
	const [modelReacting, setModelReacting] = useState(false)
	const [audioLevel, setAudioLevel] = useState(0.1)
	const [timezoneDetected, setTimezoneDetected] = useState(null) // null: checking, true: detected, false: not detected

	const [locationState, setLocationState] = useState({
		loading: false,
		data: null,
		error: null
	})

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

		// Check for undefined, null, or empty values
		if (answer === undefined || answer === null) return false

		// For string values, check if empty or whitespace-only
		if (typeof answer === "string" && answer.trim() === "") return false

		// For arrays, check if empty
		if (Array.isArray(answer) && answer.length === 0) return false

		return true
	}, [answers, currentQuestionIndex, stage])

	const submitOnboardingMutation = useMutation({
		mutationFn: (onboardingData) =>
			fetch("/api/onboarding", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ data: onboardingData })
			}).then(async (res) => {
				if (!res.ok) {
					const result = await res.json()
					throw new Error(
						result.message || "Failed to save onboarding data"
					)
				}
				return res.json()
			}),
		onSuccess: async (data, submittedAnswers) => {
			posthog?.identify(
				(await (await fetch("/api/user/profile")).json()).sub, // Fetch user ID from session
				{ name: submittedAnswers["user-name"] }
			)
			posthog?.capture("user_signed_up", { signup_method: "auth0" })
			posthog?.capture("onboarding_completed")
			await fetchUserData() // Refresh user data in the store
			router.push("/chat?show_demo=true")
		},
		onError: (error) => {
			toast.error(`Error: ${error.message}`)
			setStage("questions") // Go back to questions on error
		}
	})

	const handleNext = useCallback(() => {
		if (!isCurrentQuestionAnswered()) return

		// Trigger sphere reaction immediately
		setModelReacting(true)
		setAudioLevel(0.9) // High impact
		setSparkleTrigger((c) => c + 1)

		setTimeout(() => setModelReacting(false), 400)

		if (currentQuestionIndex < questions.length - 1) {
			setCurrentQuestionIndex((prev) => prev + 1)
		} else {
			setStage("submitting")
			submitOnboardingMutation.mutate(answers)
		}
	}, [
		currentQuestionIndex,
		isCurrentQuestionAnswered,
		submitOnboardingMutation,
		answers
	])
	// --- Effects ---

	useEffect(() => {
		if (statusChecked.current) return
		statusChecked.current = true

		const checkStatus = async () => {
			try {
				const response = await fetch("/api/user/data", {
					method: "POST"
				})
				if (!response.ok) throw new Error("Could not fetch user data.")
				const result = await response.json()
				if (result?.data?.onboardingComplete) {
					router.push("/chat")
				} else {
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
			if (userTimezone) {
				handleAnswer("timezone", userTimezone)
				setTimezoneDetected(true)
			} else {
				setTimezoneDetected(false)
			}
		} catch (e) {
			console.warn("Could not detect user timezone.")
			setTimezoneDetected(false)
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [])

	useEffect(() => {
		const handleKeyDown = (e) => {
			if (stage === "questions") {
				if (e.key === "Enter") {
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
	}, [stage, handleNext, currentQuestionIndex])

	useEffect(() => {
		let interval
		if (modelReacting) {
			setAudioLevel(0.8) // Spike the level for reaction
		} else {
			// Gentle pulse
			interval = setInterval(() => {
				setAudioLevel(Math.sin(Date.now() / 400) * 0.05 + 0.1)
			}, 50)
		}
		return () => clearInterval(interval)
	}, [modelReacting])

	// --- Render Logic ---

	if (isLoading) {
		return (
			<div className="flex flex-col items-center justify-center min-h-screen bg-brand-black text-brand-white">
				<IconLoader className="w-10 h-10 animate-spin text-[var(--color-accent-blue)]" />
			</div>
		)
	}

	const renderContent = () => {
		switch (stage) {
			case "questions":
				const currentQuestion = questions[currentQuestionIndex] ?? null
				return (
					// Use a motion.div for AnimatePresence transitions
					<motion.div
						key="questions-view"
						className="w-full h-full relative"
						initial={{ opacity: 0 }}
						animate={{ opacity: 1 }}
						exit={{ opacity: 0 }}
					>
						{/* SiriSpheres at top */}
						<motion.div
							layoutId="onboarding-sphere"
							initial={{ scale: 1, y: 0 }}
							animate={{ scale: 0.7, y: -20 }}
							transition={{ duration: 0.8, ease: "easeInOut" }}
							className="absolute md:top-[-30px] sm:top-[100px] left-1/2 -translate-x-1/2 right-1/2 w-[300px] h-[300px] md:w-[450px] md:h-[450px] pointer-events-none z-0"
						>
							<div className="w-full h-full opacity-90">
								<SiriSpheres
									status="connected"
									audioLevel={audioLevel}
								/>
							</div>
						</motion.div>

						{/* Progress Bar */}
						<div className="fixed bottom-0 left-0 right-0 w-full px-4 py-6 md:py-8 z-20 pointer-events-none">
							<div className="max-w-4xl mx-auto">
								<ProgressBar
									score={score}
									totalQuestions={questions.length}
								/>
							</div>
						</div>

						{/* Questions Container */}
						<div className="relative z-10 w-full h-full flex flex-col items-center justify-center pt-32 md:pt-40 pb-8">
							<motion.div
								key="questions-stage"
								initial={{ opacity: 0, y: 30 }}
								animate={{ opacity: 1, y: 0 }}
								transition={{ delay: 0.3, duration: 0.6 }}
								className="w-full max-w-4xl flex flex-col items-center gap-6 md:gap-8 text-center px-4"
							>
								{/* Question Text */}
								<AnimatePresence mode="wait" initial={false}>
									<motion.div
										key={currentQuestionIndex}
										initial={{ opacity: 0, y: 20 }}
										animate={{ opacity: 1, y: 0 }}
										exit={{ opacity: 0, y: -20 }}
										transition={{ duration: 0.4 }}
										className={questionStyles.container}
									>
										<div className="w-full">
											<h2
												className={questionStyles.title}
											>
												{currentQuestion.question}
											</h2>
											{currentQuestion.description && (
												<p
													className={
														questionStyles.description
													}
												>
													{currentQuestion.description}
												</p>
											)}
										</div>
									</motion.div>
								</AnimatePresence>

								{/* Answer Input */}
								<div className="w-full max-w-2xl min-h-[80px] flex items-center justify-center">
									{currentQuestion &&
										renderInput(currentQuestion)}
								</div>

								{/* Navigation */}
								<div className="mt-4 md:mt-6">
									<Button
										onClick={handleNext}
										disabled={!isCurrentQuestionAnswered()}
										size="lg"
										className="rounded-xl bg-brand-orange text-brand-black text-base md:text-lg font-semibold transition-all duration-300 hover:bg-brand-orange/90 hover:scale-105 shadow-lg shadow-brand-orange/25"
									>
										{currentQuestionIndex ===
										questions.length - 1
											? "Finish"
											: "Next"}
									</Button>
								</div>
							</motion.div>
						</div>
					</motion.div>
				)

			case "submitting":
				return (
					// prettier-ignore
					<motion.div
						key="submitting"
						initial={{ opacity: 0 }}
						animate={{ opacity: 1 }}
						exit={{ opacity: 0 }}
						className="w-full h-full flex flex-col items-center justify-center text-center"
					>
						<div className="w-[300px] h-[300px] md:w-[400px] md:h-[400px]">
							<SiriSpheres status="connecting" />
						</div>
						<h1 className="text-2xl md:text-3xl font-medium text-neutral-200 mt-8">
							Personalizing your experience...
						</h1>
					</motion.div>
				)

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
					<div className="relative w-full max-w-lg mx-auto">
						<Input
							type="text"
							value={answers[currentQuestion.id] || ""}
							onChange={(e) =>
								handleAnswer(currentQuestion.id, e.target.value)
							}
							placeholder={currentQuestion.placeholder}
							required={currentQuestion.required}
							autoFocus
							className="px-6 py-4 md:py-5 bg-neutral-900/60 backdrop-blur-sm border-neutral-700/50 rounded-xl focus:ring-brand-orange/50 focus:border-brand-orange/50 transition-all duration-300 text-center text-base md:text-lg placeholder:text-neutral-500 shadow-lg shadow-black/20"
						/>
					</div>
				)
			case "select":
				// Special handling for timezone question
				if (currentQuestion.id === "timezone") {
					const detectedTimezone = answers[currentQuestion.id]
					const isTimezoneInOptions = currentQuestion.options.some(
						(opt) => opt.value === detectedTimezone
					)

					// Create a dynamic options list
					let timezoneOptions = [...currentQuestion.options]

					// If detected timezone is not in the list, add it
					if (
						timezoneDetected &&
						detectedTimezone &&
						!isTimezoneInOptions
					) {
						timezoneOptions.unshift({
							value: detectedTimezone,
							label: detectedTimezone.replace(/_/g, " ")
						})
					}

					// Modify placeholder if detection failed
					if (timezoneDetected === false) {
						timezoneOptions[0] = {
							value: "",
							label: "Couldn't detect. Please select..."
						}
					}

					return (
						<div className="w-full max-w-xl mx-auto text-center">
							<Select
								value={answers[currentQuestion.id] || ""}
								onChange={(e) =>
									handleAnswer(
										currentQuestion.id,
										e.target.value
									)
								}
								required={currentQuestion.required}
								disabled={timezoneDetected === true}
								className="px-6 py-4 md:py-5 bg-neutral-900/60 backdrop-blur-sm border-neutral-700/50 rounded-xl focus:ring-brand-orange/50 focus:border-brand-orange/50 transition-all duration-300 text-center text-base md:text-lg placeholder:text-neutral-500 shadow-lg shadow-black/20 appearance-none"
							>
								{timezoneOptions.map((option) => (
									<option
										key={option.value}
										value={option.value}
										disabled={option.disabled}
										className="bg-brand-gray text-brand-white"
									>
										{option.label}
									</option>
								))}
							</Select>
							{timezoneDetected === true && (
								<p className="text-green-400 text-sm mt-3 bg-green-400/10 border border-green-400/20 rounded-lg px-4 py-2">
									We've automatically detected your timezone.
								</p>
							)}
							{timezoneDetected === false && (
								<p className="text-yellow-400 text-sm mt-3 bg-yellow-400/10 border border-yellow-400/20 rounded-lg px-4 py-2">
									We couldn't detect your timezone
									automatically.
								</p>
							)}
						</div>
					)
				}
				// Default select rendering for other questions
				return (
					<div className="w-full max-w-xl mx-auto">
						<Select
							value={answers[currentQuestion.id] || ""}
							onChange={(e) =>
								handleAnswer(currentQuestion.id, e.target.value)
							}
							required={currentQuestion.required}
							className="px-6 py-4 md:py-5 bg-neutral-900/60 backdrop-blur-sm border-neutral-700/50 rounded-xl focus:ring-brand-orange/50 focus:border-brand-orange/50 transition-all duration-300 text-center text-base md:text-lg placeholder:text-neutral-500 shadow-lg shadow-black/20 appearance-none"
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
						</Select>
					</div>
				)
			case "textarea":
				return (
					<div className="w-full max-w-3xl mx-auto">
						<Textarea
							value={answers[currentQuestion.id] || ""}
							onChange={(e) =>
								handleAnswer(currentQuestion.id, e.target.value)
							}
							className="w-full h-32 md:h-40 px-6 py-4 md:py-5 bg-neutral-900/60 backdrop-blur-sm border-neutral-700/50 rounded-xl focus:ring-brand-orange/50 focus:border-brand-orange/50 resize-none transition-all duration-300 text-center text-base md:text-lg placeholder:text-neutral-500 shadow-lg shadow-black/20"
							placeholder={currentQuestion.placeholder}
							autoFocus
							rows={4}
						/>
					</div>
				)
			case "location":
				return (
					<div className="flex flex-col sm:flex-row items-center justify-center gap-4 md:gap-6 w-full max-w-3xl mx-auto">
						<Input
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
							className="px-6 py-4 md:py-5 bg-neutral-900/60 backdrop-blur-sm border-neutral-700/50 rounded-xl focus:ring-brand-orange/50 focus:border-brand-orange/50 transition-all duration-300 text-center text-base md:text-lg placeholder:text-neutral-500 shadow-lg shadow-black/20 sm:flex-grow"
						/>
						<span className="hidden sm:inline text-neutral-400 text-base font-medium">
							or
						</span>
						<span className="sm:hidden text-neutral-400 text-base">
							or
						</span>
						<Button
							type="button"
							onClick={handleGetLocation}
							disabled={locationState.loading}
							variant="outline"
							className="px-6 py-3 md:py-4 rounded-xl transition-all duration-300 whitespace-nowrap disabled:opacity-50 font-medium border-brand-orange/30 text-brand-orange hover:bg-brand-orange/10"
						>
							{locationState.loading
								? "Detecting..."
								: "Detect Current Location"}
						</Button>
					</div>
				)

			default:
				return null
		}
	}

	return (
		<div className="relative flex flex-col items-center min-h-screen w-full text-brand-white overflow-hidden">
			<div className="absolute inset-0 z-[-1]">
				<InteractiveNetworkBackground />
			</div>
			<div className="absolute -top-[250px] left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-brand-orange/10 rounded-full blur-3xl -z-10" />
			<div className={cn("relative z-10 w-full h-screen")}>
				<SparkleEffect trigger={sparkleTrigger} />
				<AnimatePresence mode="wait">
					{stage === "intro" ? (
						<IntroSequence
							onComplete={() => setStage("questions")}
						/>
					) : (
						renderContent()
					)}
				</AnimatePresence>
			</div>
		</div>
	)
}

export default OnboardingPage
