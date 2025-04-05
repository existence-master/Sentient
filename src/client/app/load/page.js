"use client"

import { useState, useEffect } from "react" // Importing necessary React hooks
import AnimatedLogo from "@components/AnimatedLogo" // Component for displaying an animated logo
import Link from "next/link" // Component for client-side navigation in Next.js
import { MultiStepLoader } from "../../components/MultiStep" // Component for displaying a multi-step loading animation
import AnimatedBeam from "@components/AnimatedBeam" // Component for an animated beam effect
import ShiningButton from "@components/ShiningButton" // Custom button component with a shining effect
import { IconRocket, IconMessage } from "@tabler/icons-react" // Icons from tabler-icons-react library
import toast from "react-hot-toast" // Library for displaying toast notifications
import React from "react"

/**
 * Load Component - The initial loading screen of the application.
 *
 * This component is responsible for initializing the application by fetching user data,
 * and determining whether to navigate the user to the personality test or directly to the chat interface
 * based on their onboarding status. It also displays a loading animation while initializing.
 *
 * @returns {React.ReactNode} - The Load component UI.
 */
const Load = () => {
	// State to track the onboarding status of the user.
	// null: initial state, false: user not onboarded, true: user onboarded
	const [onboarded, setOnboarded] = useState(null) // onboarded: boolean | null
	// State to manage the loading state of the component, displayed during initialization.
	const [loading, setLoading] = useState(true) // loading: boolean

	/**
	 * Initializes the application by fetching user data from the database.
	 *
	 * This function uses Electron's `invoke` to call backend functions to:
	 * 1. Fetch user data from the database to check onboarding status.
	 * 2. Optionally start Neo4j and initiate the FastAPI server (currently commented out).
	 * 3. Update the `loading` and `onboarded` states based on the fetched data.
	 *
	 * @async
	 * @function initializeApp
	 * @returns {Promise<void>}
	 */
	const initializeApp = async () => {
		try {
			// Fetch user data from the database using electron invoke
			const { data: userData } =
				await window.electron?.invoke("get-user-data")

			setLoading(false) // Set loading to false once initialization is complete (or attempted)
			setOnboarded(userData?.firstRunCompleted) // Set onboarded state based on 'firstRunCompleted' from user data
		} catch (error) {
			toast.error(`Error initializing app: ${error}`) // Display error toast if initialization fails
			setLoading(false) // Ensure loading is set to false even if there's an error
		}
	}

	/**
	 * useEffect hook to call `initializeApp` when the component mounts.
	 *
	 * This ensures that the application initialization process starts as soon as the Load component is rendered.
	 */
	useEffect(() => {
		initializeApp() // Call initializeApp function when component mounts
	}, []) // Empty dependency array ensures this effect runs only once on mount

	/**
	 * Array of loading state messages for the MultiStepLoader component.
	 *
	 * Each object in this array defines a text message to be displayed during the loading animation,
	 * providing user feedback during the initialization process.
	 */
	const loadingStates = [
		{
			text: "loading your memories"
		},
		{
			text: "loading my brain"
		},
		{
			text: "loading our past conversations"
		},
		{
			text: "loading your secrets"
		},
		{
			text: "hello, friend"
		}
	]

	return (
		<AnimatedBeam>
			<div className="min-h-screen flex flex-col items-center justify-center">
				<div className="flex flex-col items-center justify-center h-full backdrop-blur-xs">
					{/* Conditional rendering based on loading state */}
					{loading && (
						<>
							{/* Spacer to maintain layout when loader is visible */}
							<div className="h-72 w-72" />
							{/* MultiStepLoader component for displaying loading animation with messages */}
							<MultiStepLoader
								loadingStates={loadingStates} // Array of loading messages
								loading={loading} // Loading state boolean
								duration={2000} // Duration for each loading step in milliseconds
							/>
						</>
					)}
					{/* Render AnimatedLogo and title when not loading */}
					{!loading && (
						<>
							<AnimatedLogo /> {/* Animated logo component */}
							<h1 className="text-white text-4xl mt-4">
								Sentient
							</h1>{" "}
							{/* Application title */}
						</>
					)}
					{/* Conditional rendering based on onboarding status (after loading) */}
					{onboarded !== null ? (
						onboarded === false ? (
							// If not onboarded, link to personality test
							<Link href="/personality-test" className="mt-10">
								<ShiningButton className="" icon={IconRocket}>
									Get Started
								</ShiningButton>
							</Link>
						) : (
							// If onboarded, link to chat interface
							<Link href="/chat" className="mt-10">
								<ShiningButton className="" icon={IconMessage}>
									Go to Chat
								</ShiningButton>
							</Link>
						)
					) : null}
				</div>
			</div>
		</AnimatedBeam>
	)
}

export default Load
