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
	// State to indicate if initial user data (like personalInfo) is loaded from the backend.
	const [isInitialDataLoaded, setIsInitialDataLoaded] = useState(false) // isInitialDataLoaded: boolean
	// State to control the visibility of the information tooltip.
	const [showTooltip, setShowTooltip] = useState(false) // showTooltip: boolean
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
		 * Fetches initial user data from the database using electron invoke.
		 * If essential personalInfo (e.g., name) exists, redirects to '/integrations'.
		 * Sets 'isInitialDataLoaded' to true afterwards.
		 */
		const fetchInitialUserData = async () => {
			try {
				const response = await window.electron?.invoke("get-user-data")
				if (response.status === 200 && response.data) {
					// Check if personalInfo and specifically personalInfo.name exists
					if (response.data.personalInfo && response.data.personalInfo.name) {
						toast.success("Welcome back! Redirecting...")
						router.push("/chat")
					}
				}
			} catch (error) {
				toast.error(`Error fetching user data: ${error.message || String(error)}`)
			} finally {
				setIsInitialDataLoaded(true) // Set loading to false after attempt
			}
		}

		fetchInitialUserData()
	}, [router]) // router is a dependency for router.push

	/**
	 * useEffect hook to automatically progress to the next question if an answer is already in formData for the current question.
	 *
	 * This logic ensures that if the component re-renders and the current question already has a selected option,
	 * it automatically advances to the next question for a smoother user experience.
	 */
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
			const response = await window.electron?.invoke("set-user-data", {
				data: { personalInfo: basicInfo }
			})

			if (response.status === 200) {
				setShowBasicInfoForm(false) // Hide form
				toast.success("Information saved! Proceeding to chat...")
				router.push("/chat") // Navigate to chat page
			} else {
				throw new Error(`Error saving basic information: ${response.message || "Unknown error"}`)
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
	 * Displays a loading message if initial user data is not yet loaded (`isInitialDataLoaded` is false).
	 *
	 * @returns {React.ReactNode | null} - Returns the loading UI if `isInitialDataLoaded` is false, otherwise null.
	 */
	if (!isInitialDataLoaded) {
		return (
			<div className="flex flex-col items-center justify-center min-h-screen bg-black">
				{/* Optionally, add a spinner or a more sophisticated loading indicator here */}
				<AnimatedLogo />
				<h1 className="text-white text-2xl mt-4">Loading initial data...</h1>
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
										your experience.
									</p>
								</div>
							)}
						</div>
					</div>

					{/* The main content area will now primarily be for the BasicInfoForm if needed, 
					    or it will be blank if the user is immediately redirected or after referral.
						The BasicInfoForm is already handled by the conditional rendering block:
						if (showBasicInfoForm) { ... } 
						So, this section can be simplified or used for a general container if other UI elements
						were to be added here that are not BasicInfoForm or Referral.
						For now, if not showing BasicInfoForm or referral forms, this part of the UI will be empty,
						which is fine as the user would have been redirected or is in the process of filling forms.
					*/}
					{!showBasicInfoForm && isInitialDataLoaded && isReferred === false && (
						<div className="flex flex-col h-[70vh] items-center justify-center text-white">
							<p>Loading next step...</p> 
							{/* This state should ideally be very short-lived due to navigation in handleBasicInfoSubmit */}
						</div>
					)}
				</motion.div>
			</div>
		</AnimatedBeam>
	)
}

export default PersonalityTest
