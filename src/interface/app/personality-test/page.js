"use client"

import React from "react"
import { useEffect, useState } from "react" // Importing necessary React hooks
import { useRouter } from "next/navigation" // Hook for client-side navigation in Next.js
import { motion } from "framer-motion" // Animation library for React
import { IconInfoCircle } from "@tabler/icons-react" // Icon component for info circle
import AnimatedLogo from "@components/AnimatedLogo" // Component for displaying an animated logo
import ShiningButton from "@components/ShiningButton" // Custom button component with a shining effect
import AnimatedBeam from "@components/AnimatedBeam" // Component for animated beam effect
import ShinyCard from "@components/ShinyCard" // Custom card component with a shiny effect
import ModelSelection from "@components/ModelSelection"
import toast from "react-hot-toast" // Library for displaying toast notifications

/**
 * BasicInfoForm Component - Collects basic user information: name, location, and date of birth.
 *
 * This form is displayed during the onboarding process to gather essential user details.
 * It validates input fields and uses a callback function (`onSubmit`) to pass data to the parent component.
 *
 * @param {object} props - Component props.
 * @param {function} props.onSubmit - Callback function to handle form submission with basic info data.
 * @returns {React.ReactNode} - The BasicInfoForm component UI.
 */
const BasicInfoForm = ({ onSubmit }) => {
	// State to manage basic information input fields
	const [basicInfo, setBasicInfo] = useState({
		// basicInfo: { name: string, location: string, dateOfBirth: string }
		name: "",
		location: "",
		dateOfBirth: ""
	})

	/**
	 * Handles changes to input fields in the form.
	 *
	 * Updates the `basicInfo` state whenever an input field value changes.
	 *
	 * @function handleChange
	 * @param {React.ChangeEvent<HTMLInputElement>} e - The change event.
	 * @returns {void}
	 */
	const handleChange = (e) => {
		const { name, value } = e.target // Destructure name and value from the event target
		setBasicInfo((prev) => ({ ...prev, [name]: value })) // Update basicInfo state for the changed input field
	}

	/**
	 * Handles form submission for basic information.
	 *
	 * Validates if all fields are filled and then calls the `onSubmit` callback with the collected data.
	 * Prevents default form submission behavior.
	 *
	 * @function handleSubmit
	 * @param {React.FormEvent} e - The form submit event.
	 * @returns {void}
	 */
	const handleSubmit = (e) => {
		e.preventDefault() // Prevent default form submission

		// Check if any of the basic info fields are empty
		if (!basicInfo.name || !basicInfo.location || !basicInfo.dateOfBirth) {
			alert("Please fill in all fields.") // Alert user to fill in all fields
			return // Stop form submission if fields are empty
		}

		onSubmit(basicInfo) // Call onSubmit callback function with basic info data
	}

	return (
		<AnimatedBeam className={"w-screen h-screen"}>
			<div className="min-h-screen flex flex-col items-center justify-center">
				<div className="w-4/5 max-w-lg font-Poppins bg-matteblack p-10 rounded-2xl shadow-2xl">
					<h2 className="text-3xl font-bold text-white mb-6 text-center">
						Lets get to know you
					</h2>
					<form
						className="space-y-6 text-white"
						onSubmit={handleSubmit}
					>
						<div>
							<label className="block text-xl font-medium text-gray-300 mb-2">
								Name
							</label>
							<input
								type="text"
								name="name"
								value={basicInfo.name}
								onChange={handleChange}
								className="text-input focus:ring-lightblue"
								placeholder="Enter your name"
							/>
						</div>
						<div>
							<label className="block text-xl font-medium text-gray-300 mb-2">
								Location
							</label>
							<input
								type="text"
								name="location"
								value={basicInfo.location}
								onChange={handleChange}
								className="text-input focus:ring-lightblue"
								placeholder="Enter your location"
							/>
						</div>
						<div>
							<label className="block text-xl font-medium text-gray-300 mb-2">
								Date of Birth
							</label>
							<input
								type="date"
								name="dateOfBirth"
								value={basicInfo.dateOfBirth}
								onChange={handleChange}
								className="text-input focus:ring-lightblue"
							/>
						</div>
						<div className="w-full flex items-center justify-center">
							<ShiningButton type="submit" className={"w-full"}>
								Continue
							</ShiningButton>
						</div>
					</form>
				</div>
			</div>
		</AnimatedBeam>
	)
}

/**
 * PersonalityTest Component - Main component for rendering the personality test.
 *
 * This component manages the state for the personality test, including questions, scores,
 * form data, and UI states like loading, tooltips, and results display.
 * It fetches existing personality data, handles question progression, score calculation,
 * and submission of personality type to the backend.
 *
 * @returns {React.ReactNode} - The PersonalityTest component UI.
 */
const PersonalityTest = () => {
	const router = useRouter() // Hook to get the router object for navigation

	// State to control visibility of the basic info form, shown after referral (or directly).
	const [showBasicInfoForm, setShowBasicInfoForm] = useState(false) // showBasicInfoForm: boolean
	// State to keep track of scores for each personality trait category (E, I, S, N, T, F, J, P).
	const [scores, setScores] = useState({
		// scores: { E: number, I: number, S: number, N: number, T: number, F: number, J: number, P: number }
		E: 0,
		I: 0,
		S: 0,
		N: 0,
		T: 0,
		F: 0,
		J: 0,
		P: 0
	})

	// State to manage the index of the current question being displayed.
	const [currentQuestion, setCurrentQuestion] = useState(0) // currentQuestion: number
	// State to indicate if personality data is loaded from the backend.
	const [isPersonalityLoaded, setIsPersonalityLoaded] = useState(false) // isPersonalityLoaded: boolean
	// State to store form data, specifically answers to personality test questions.
	const [formData, setFormData] = useState({}) // formData: { [question: string]: number }
	// State to control the visibility of the information tooltip.
	const [showTooltip, setShowTooltip] = useState(false) // showTooltip: boolean
	// State to control the visibility of the personality test results screen.
	const [showResults, setShowResults] = useState(false) // showResults: boolean
	// State to manage whether the user was referred, null: initial, true: referred, false: not referred.
	const [isReferred, setIsReferred] = useState(null) // isReferred: boolean | null
	// State to store the referral code entered by the user.
	const [referralCode, setReferralCode] = useState("") // referralCode: string
	const [showModelSelection, setShowModelSelection] = useState(false)

	/**
	 * useEffect hook to fetch existing personality data on component mount.
	 *
	 * Checks if personality data already exists in the database. If it does, it redirects the user to the integrations page,
	 * assuming they have already completed the personality test. Sets `isPersonalityLoaded` to true after attempt.
	 */
	useEffect(() => {
		/**
		 * Fetches personality data from the database using electron invoke.
		 * If personality data exists, redirects to '/integrations'. Sets 'isPersonalityLoaded' to true afterwards.
		 */
		const fetchPersonality = async () => {
			try {
				const response = await window.electron?.invoke("get-db-data") // Invoke electron to get data from database
				if (response.status === 200 && response.data) {
					const { personality } = response.data // Extract personality data from response
					if (personality && Object.keys(personality).length > 0) {
						router.push("/integrations") // Redirect to integrations page if personality data exists
					}
				}
			} catch (error) {
				toast.error(`Error fetching personality: ${error}`) // Show error toast if fetching fails
			}
		}

		fetchPersonality() // Call fetchPersonality on component mount
		setIsPersonalityLoaded(true) // Set isPersonalityLoaded to true after attempting to fetch personality data
	}, []) // Empty dependency array ensures this effect runs only once on mount

	/**
	 * Array of personality test questions.
	 *
	 * Each question is an object with `text` (the question itself) and `area` (personality trait area it tests).
	 * Areas correspond to personality dimensions: E/I, S/N, T/F, J/P.
	 */
	const questions = [
		{ text: "You regularly make new friends.", area: "E" },
		{
			text: "Complex and novel ideas excite you more than simple and straightforward ones.",
			area: "N"
		},
		{
			text: "You usually feel more persuaded by what resonates emotionally with you than by factual arguments.",
			area: "F"
		},
		{
			text: "Your living and working spaces are clean and organized.",
			area: "J"
		},
		{
			text: "You usually stay calm, even under a lot of pressure.",
			area: "T"
		},
		{
			text: "You find the idea of networking or promoting yourself to strangers very daunting.",
			area: "I"
		},
		{
			text: "You prioritize facts over people’s feelings when determining a course of action.",
			area: "T"
		},
		{
			text: "You often allow the day to unfold without any schedule at all.",
			area: "P"
		},
		{
			text: "You rarely worry about whether you make a good impression on people you meet.",
			area: "I"
		},
		{
			text: "You enjoy participating in team-based activities.",
			area: "E"
		},
		{
			text: "You enjoy experimenting with new and untested approaches.",
			area: "N"
		},
		{
			text: "You prioritize being sensitive over being completely honest.",
			area: "F"
		},
		{
			text: "In disagreements, you prioritize proving your point over preserving the feelings of others.",
			area: "T"
		},
		{
			text: "You usually wait for others to introduce themselves first at social gatherings.",
			area: "I"
		},
		{ text: "Your mood can change very quickly.", area: "F" },
		{
			text: "You are not easily swayed by emotional arguments.",
			area: "T"
		},
		{
			text: "You often end up doing things at the last possible moment.",
			area: "P"
		},
		{ text: "You enjoy debating ethical dilemmas.", area: "T" },
		{
			text: "You usually prefer to be around others rather than on your own.",
			area: "E"
		},
		{
			text: "You become bored or lose interest when the discussion gets highly theoretical.",
			area: "S"
		},
		{
			text: "When facts and feelings conflict, you usually find yourself following your heart.",
			area: "F"
		},
		{
			text: "You find it challenging to maintain a consistent work or study schedule.",
			area: "P"
		},
		{
			text: "You rarely second-guess the choices that you have made.",
			area: "J"
		},
		{
			text: "Your friends would describe you as lively and outgoing.",
			area: "E"
		},
		{
			text: "You are drawn to various forms of creative expression, such as writing.",
			area: "N"
		},
		{
			text: "You usually base your choices on objective facts rather than emotional impressions.",
			area: "T"
		},
		{ text: "You like to have a to-do list for each day.", area: "J" },
		{ text: "You rarely feel insecure.", area: "T" },
		{ text: "You avoid making phone calls.", area: "I" },
		{
			text: "You enjoy exploring unfamiliar ideas and viewpoints.",
			area: "N"
		},
		{
			text: "You can easily connect with people you have just met.",
			area: "E"
		},
		{
			text: "If your plans are interrupted, your top priority is to get back on track as soon as possible.",
			area: "J"
		},
		{
			text: "You are still bothered by mistakes that you made a long time ago.",
			area: "F"
		},
		{
			text: "You are not too interested in discussing theories on what the world could look like in the future.",
			area: "S"
		},
		{
			text: "Your emotions control you more than you control them.",
			area: "F"
		},
		{
			text: "When making decisions, you focus more on how the affected people might feel than on what is most logical or efficient.",
			area: "F"
		},
		{
			text: "Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.",
			area: "P"
		},
		{
			text: "When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.",
			area: "I"
		},
		{
			text: "You would love a job that requires you to work alone most of the time.",
			area: "I"
		},
		{
			text: "You believe that pondering abstract philosophical questions is a waste of time.",
			area: "S"
		},
		{
			text: "You feel more drawn to busy, bustling atmospheres than to quiet, intimate places.",
			area: "E"
		},
		{
			text: "If a decision feels right to you, you often act on it without needing further proof.",
			area: "N"
		},
		{ text: "You often feel overwhelmed.", area: "F" },
		{
			text: "You complete things methodically without skipping over any steps.",
			area: "J"
		},
		{
			text: "You prefer tasks that require you to come up with creative solutions rather than follow concrete steps.",
			area: "N"
		},
		{
			text: "You are more likely to rely on emotional intuition than logical reasoning when making a choice.",
			area: "F"
		},
		{ text: "You struggle with deadlines.", area: "P" },
		{
			text: "You feel confident that things will work out for you.",
			area: "J"
		}
	]

	/**
	 * useEffect hook to automatically progress to the next question if an answer is already in formData for the current question.
	 *
	 * This logic ensures that if the component re-renders and the current question already has a selected option,
	 * it automatically advances to the next question for a smoother user experience.
	 */
	useEffect(() => {
		// Check if formData for the current question is already set (not undefined)
		if (formData[questions[currentQuestion]?.text] !== undefined) {
			handleNext() // Automatically proceed to the next question
		}
	}, [formData, currentQuestion]) // Effect dependencies: formData, currentQuestion

	/**
	 * useEffect hook to conditionally show the BasicInfoForm after determining referral status.
	 *
	 * Sets `showBasicInfoForm` to true if `isReferred` is explicitly set to false, indicating
	 * that the referral process is completed (or skipped) and basic info form should now be displayed.
	 */
	useEffect(() => {
		if (isReferred === false) {
			setShowBasicInfoForm(true) // Show BasicInfoForm when isReferred is false
		}
	}, [isReferred]) // Effect dependency: isReferred

	/**
	 * Handles submission of basic information collected in BasicInfoForm.
	 *
	 * Sends the basic info data to the backend using electron invoke to be stored in the database.
	 * On success, hides the BasicInfoForm; on failure, shows an error toast.
	 *
	 * @async
	 * @function handleBasicInfoSubmit
	 * @param {object} basicInfo - Basic information data submitted from the form.
	 * @returns {Promise<void>}
	 */
	const handleBasicInfoSubmit = async (basicInfo) => {
		try {
			const response = await window.electron?.invoke("set-db-data", {
				data: { personalInfo: basicInfo }
			})

			if (response.status === 200) {
				setShowBasicInfoForm(false)
				setShowModelSelection(true) // Show model selection after basic info
			} else {
				throw new Error("Error saving basic information")
			}
		} catch (error) {
			toast.error(`Error saving basic information: ${error}`)
		}
	}

	/**
	 * Handles click on a personality test option.
	 *
	 * Updates the score for the personality area corresponding to the current question,
	 * records the answer in `formData`, and progresses to the next question or submits the test
	 * if it's the last question.
	 *
	 * @function handleOptionClick
	 * @param {number} value - The value associated with the clicked option (e.g., -1 or 1).
	 * @returns {void}
	 */
	const handleOptionClick = (value) => {
		const question = questions[currentQuestion] // Get current question object
		const { area } = question // Extract the personality area from the question

		setScores((prevScores) => ({
			// Update scores state
			...prevScores,
			[area]: prevScores[area] + value // Increment score for the question's area
		}))

		setFormData((prevData) => ({
			// Update formData state
			...prevData,
			[question.text]: value // Store the answer value for the question text
		}))

		if (currentQuestion < questions.length - 1) {
			setCurrentQuestion((prev) => prev + 1) // Move to the next question if not the last question
		} else {
			handleSubmit() // Submit the test if it's the last question
		}
	}

	/**
	 * Handles moving to the next question in the personality test.
	 *
	 * Increments the `currentQuestion` state to display the next question.
	 * If already at the last question, it calls `handleSubmit` to finalize the test.
	 *
	 * @function handleNext
	 * @returns {void}
	 */
	const handleNext = () => {
		if (currentQuestion < questions.length - 1) {
			setCurrentQuestion((prev) => prev + 1) // Increment currentQuestion to move to the next question
		} else {
			handleSubmit() // If it's the last question, submit the test
		}
	}

	/**
	 * Handles moving back to the previous question in the personality test.
	 *
	 * Decrements `currentQuestion` to go back, and clears the answer for the question being moved back from
	 * in `formData` to allow re-selection.
	 *
	 * @function handleBack
	 * @returns {void}
	 */
	const handleBack = () => {
		if (currentQuestion > 0) {
			const previousQuestion = questions[currentQuestion - 1] // Get the question object of the previous question
			setFormData((prevData) => {
				// Update formData state
				const newData = { ...prevData } // Create a copy of the previous formData
				delete newData[previousQuestion.text] // Delete the answer for the question we are moving back from
				return newData // Return the updated formData
			})

			setCurrentQuestion((prev) => prev - 1) // Decrement currentQuestion to move back to the previous question
		}
	}

	/**
	 * Calculates the personality type based on the accumulated scores.
	 *
	 * Determines the 4-letter personality type (e.g., ESTJ, INFP) by comparing scores for each of the four dimensions
	 * (E/I, S/N, T/F, J/P) and selecting the dominant trait for each.
	 *
	 * @function calculatePersonalityType
	 * @returns {string} - The calculated 4-letter personality type.
	 */
	const calculatePersonalityType = () => {
		const { E, I, S, N, T, F, J, P } = scores // Destructure scores for each personality dimension
		return `${E > I ? "E" : "I"}${S > N ? "S" : "N"}${T > F ? "T" : "F"}${
			J > P ? "J" : "P" // Determine each letter of the personality type based on score comparison
		}`
	}

	/**
	 * Handles the final submission of the personality test.
	 *
	 * Calculates the personality type, sends it to the backend using electron invoke to store in the database,
	 * and then sets `showResults` to true to display the personality results.
	 *
	 * @async
	 * @function handleSubmit
	 * @returns {Promise<void>}
	 */
	const handleSubmit = async () => {
		const personalityType = calculatePersonalityType() // Calculate personality type from scores

		try {
			const response = await window.electron?.invoke("set-db-data", {
				// Invoke electron to set data in database
				data: { personalityType } // Data to set is the calculated personalityType
			})

			if (response.status === 200) {
				setShowResults(true) // Show personality results on successful submission
			} else {
				throw new Error("Error saving personality type") // Throw error if response status is not 200
			}
		} catch (error) {
			toast.error(`Error saving personality type: ${error}`) // Show error toast if any error occurs
		}
	}

	/**
	 * PersonalityResults Component - Displays the personality test results.
	 *
	 * This component takes the calculated personality type and a proceed callback as props.
	 * It renders the 4-letter personality type, descriptions for each letter, and a button to proceed.
	 *
	 * @param {object} props - Component props.
	 * @param {string} props.personalityType - The calculated 4-letter personality type.
	 * @param {function} props.onProceed - Callback function to handle proceeding after viewing results.
	 * @returns {React.ReactNode} - The PersonalityResults component UI.
	 */
	const PersonalityResults = ({ personalityType, onProceed }) => {
		// Descriptions for each personality trait letter
		const descriptions = {
			E: "Extroverts are outgoing and gain energy from being around others.",
			I: "Introverts are reserved and gain energy from spending time alone.",
			S: "Sensors focus on facts and details, preferring practical approaches.",
			N: "Intuitives look at the bigger picture and focus on abstract concepts.",
			T: "Thinkers base decisions on logic and objectivity.",
			F: "Feelers prioritize emotions and value empathy and harmony.",
			J: "Judgers prefer structure, organization, and planning.",
			P: "Perceivers are flexible and enjoy spontaneity and adaptability."
		}

		return (
			<div className="w-4/5 max-w-5xl bg-matteblack border p-10 rounded-2xl shadow-2xl text-white">
				<h2 className="text-3xl font-bold mb-8 text-center">
					Your Personality Type: {personalityType}
				</h2>
				<div className="flex justify-between flex-wrap gap-4">
					{/* Map over each letter of the personality type to display trait cards */}
					{personalityType.split("").map((char, index) => (
						<div
							key={index}
							className="flex flex-col items-center bg-matteblack hover-button p-6 rounded-lg shadow-lg w-1/5 min-w-[150px] grow"
						>
							<h3 className="text-5xl font-extrabold text-blue-400 mb-4">
								{char}
							</h3>
							<p className="text-sm text-center">
								{descriptions[char]}
							</p>
						</div>
					))}
				</div>
				<div className="mt-8 text-center ">
					<ShiningButton onClick={onProceed}>Proceed</ShiningButton>
				</div>
			</div>
		)
	}

	/**
	 * Handles submission of the referral code.
	 *
	 * If `isReferred` is true and a `referralCode` is provided, it attempts to set the referrer in the backend
	 * using electron invoke. On success, sets `isReferred` to false, proceeding to the next step.
	 * If `isReferred` is false (user not referred), it directly sets `isReferred` to false to skip referral step.
	 *
	 * @async
	 * @function handleReferralSubmit
	 * @returns {Promise<void>}
	 */
	const handleReferralSubmit = async () => {
		if (isReferred && referralCode) {
			// Check if user was referred and if a referral code is entered
			try {
				let response = await window.electron?.invoke("set-referrer", {
					referralCode
				}) // Invoke electron to set referrer with referral code

				if (response.status === 200) {
					setIsReferred(false) // Set isReferred to false on successful referral submission
				}
			} catch (error) {
				toast.error(`Error submitting referral code: ${error}`) // Show error toast if referral submission fails
			}
		} else {
			setIsReferred(false) // If not referred, directly set isReferred to false to proceed
		}
	}

	/**
	 * Conditional rendering for asking if the user was referred.
	 *
	 * If `isReferred` state is null, this initial question screen is displayed, asking if the user was referred.
	 *
	 * @returns {React.ReactNode | null} - Returns the referral question UI if `isReferred` is null, otherwise null.
	 */
	if (isReferred === null) {
		return (
			<div className="min-h-screen flex flex-col items-center justify-center bg-black text-white">
				<h1 className="text-3xl font-bold mb-6">
					Were you referred by someone?
				</h1>
				<div className="flex gap-4">
					<ShiningButton onClick={() => setIsReferred(true)}>
						Yes
					</ShiningButton>
					<ShiningButton onClick={() => setIsReferred(false)}>
						No
					</ShiningButton>
				</div>
			</div>
		)
	}

	/**
	 * Conditional rendering for the referral code input screen.
	 *
	 * If `isReferred` is true, this screen is shown, prompting the user to enter a referral code.
	 * Includes input for referral code and a submit button.
	 *
	 * @returns {React.ReactNode | null} - Returns the referral code input UI if `isReferred` is true, otherwise null.
	 */
	if (isReferred === true) {
		return (
			<div className="min-h-screen flex flex-col items-center justify-center bg-black text-white">
				<div className="mt-8">
					<label className="block mb-4 text-lg">
						Enter Referral Code:
						<input
							type="text"
							className="block w-full px-4 py-2 mt-2 rounded-lg bg-gray-800 text-white"
							value={referralCode}
							onChange={(e) => setReferralCode(e.target.value)}
						/>
					</label>
					<ShiningButton
						onClick={async () => {
							// Async onClick handler to call handleReferralSubmit and then show BasicInfoForm
							await handleReferralSubmit() // Submit referral code
							setShowBasicInfoForm(true) // Show BasicInfoForm after referral submission
						}}
					>
						Submit
					</ShiningButton>
				</div>
			</div>
		)
	}

	/**
	 * Conditional rendering for loading state.
	 *
	 * Displays a loading message if personality data is not yet loaded (`isPersonalityLoaded` is false).
	 *
	 * @returns {React.ReactNode | null} - Returns the loading UI if `isPersonalityLoaded` is false, otherwise null.
	 */
	if (!isPersonalityLoaded) {
		return (
			<div className="flex flex-col items-center justify-center min-h-screen bg-black">
				<h1 className="text-white text-4xl mb-6">Loading...</h1>
			</div>
		)
	}

	/**
	 * Conditional rendering for the basic information form.
	 *
	 * Displays the BasicInfoForm component if `showBasicInfoForm` is true.
	 *
	 * @returns {React.ReactNode | null} - Returns the BasicInfoForm UI if `showBasicInfoForm` is true, otherwise null.
	 */
	if (showBasicInfoForm) {
		return (
			<div className="min-h-screen flex justify-center items-center bg-black">
				<BasicInfoForm onSubmit={handleBasicInfoSubmit} />
			</div>
		)
	}

	/**
	 * Main return statement for the PersonalityTest component, rendering the personality test UI.
	 *
	 * Includes animated beam background, header with animated logo and info tooltip,
	 * conditional rendering for personality results or question cards, and motion-animated elements for transitions.
	 *
	 * @returns {React.ReactNode} - The main UI for the PersonalityTest component.
	 */

	if (showModelSelection) {
		return (
			<AnimatedBeam className={"w-screen h-screen"}>
				<div className="min-h-screen flex justify-center items-center">
					<ModelSelection
						onProceed={() => setShowModelSelection(false)}
					/>
				</div>
			</AnimatedBeam>
		)
	}

	return (
		<AnimatedBeam className={"w-screen h-screen"}>
			<div className="relative overflow-hidden">
				<motion.div
					className="min-h-screen flex flex-col items-center py-10"
					initial={{ opacity: 0 }} // Initial animation state: opacity 0
					animate={{ opacity: 1 }} // Animation target state: opacity 1
					transition={{ duration: 0.5 }} // Animation transition duration: 0.5 seconds
				>
					<div className="flex h-1/5 justify-between items-center w-full px-10 mb-10">
						<div className="flex items-center">
							<AnimatedLogo /> {/* Animated logo component */}
							<h1 className="text-white text-3xl ml-5 font-Poppins font-normal">
								Sentient
							</h1>
						</div>
						<div
							className="relative"
							onMouseEnter={() => setShowTooltip(true)} // Show tooltip on mouse enter
							onMouseLeave={() => setShowTooltip(false)} // Hide tooltip on mouse leave
						>
							<IconInfoCircle
								className="w-6 h-6 text-white cursor-pointer"
								aria-label="More info"
							/>
							{showTooltip && ( // Conditional rendering for tooltip
								<div className="absolute top-10 right-0 bg-gray-800 text-white text-sm p-3 rounded-xl shadow-lg w-72">
									<h2 className="font-bold mb-2">
										Why do we need this?
									</h2>
									<p>
										We use this information to personalize
										your experience. All data stays local.
									</p>
								</div>
							)}
						</div>
					</div>

					{showResults ? ( // Conditional rendering for personality results screen
						<div className="flex flex-col h-[70vh] items-center justify-center">
							<PersonalityResults
								personalityType={calculatePersonalityType()} // Pass calculated personality type
								onProceed={() => router.push("/integrations")} // Callback to proceed to integrations page
							/>
						</div>
					) : (
						// Conditional rendering for personality test questions UI
						<motion.div
							className="w-4/5 max-w-4xl min-h-[70vh] flex items-center justify-center"
							initial={{ opacity: 0, y: 50 }} // Initial animation state: opacity 0, y 50px down
							animate={{ opacity: 1, y: 0 }} // Animation target state: opacity 1, y 0
							transition={{ duration: 0.5 }} // Animation transition duration: 0.5 seconds
						>
							<ShinyCard
								questionText={questions[currentQuestion].text} // Question text for the current question
								currentQuestionIndex={currentQuestion} // Index of the current question
								totalQuestions={questions.length} // Total number of questions
								onOptionClick={(value) => {
									// Handler for option click, progresses test or submits
									if (
										currentQuestion <
										questions.length - 1
									) {
										handleOptionClick(value)
									} else {
										handleSubmit() // Submit test on last question
									}
								}}
								selectedOption={
									formData[
										questions[currentQuestion]?.text
									] ?? null // Selected option for the current question, if any
								}
								onBackClick={handleBack} // Handler for back button click
							/>
						</motion.div>
					)}
				</motion.div>
			</div>
		</AnimatedBeam>
	)
}

export default PersonalityTest
