"use client"

import { useEffect } from "react" // Importing the useEffect hook from React
import { useRouter } from "next/navigation" // Importing the useRouter hook from next/navigation for client-side routing
import AnimatedLogo from "@components/AnimatedLogo" // Importing the AnimatedLogo component
import toast from "react-hot-toast" // Importing the toast library for displaying notifications
import React from "react"

/**
 * Home Component - The landing page of the application.
 *
 * This component serves as the initial entry point of the application.
 * Upon loading, it immediately redirects the user to the '/update' page to check for updates.
 * It displays an animated logo and the application title while the redirection is being processed.
 *
 * @returns {React.ReactNode} - The Home component UI, which includes an animated logo and application title.
 */
const Home = () => {
	const router = useRouter() // Initializing the router for navigation

	/**
	 * useEffect hook to initialize the app and navigate to the update page on component mount.
	 *
	 * This hook runs once after the component is mounted. It defines an async function `initializeApp`
	 * that attempts to navigate the user to the '/update' route. Any errors during initialization
	 * are caught and displayed as a toast notification.
	 */
	useEffect(() => {
		/**
		 * Initializes the application and navigates to the update page.
		 *
		 * This asynchronous function is designed to be called once when the component mounts.
		 * It uses the Next.js router to redirect the user to the '/update' page to initiate the update process.
		 * Any errors during this initialization and navigation process are caught and displayed to the user as a toast error message.
		 *
		 * @async
		 * @function initializeApp
		 * @returns {Promise<void>} - A promise that resolves after attempting to redirect to the update page or after handling any errors.
		 */
		const initializeApp = async () => {
			try {
				router.push("/update") // Programmatically navigate to the '/update' route
			} catch (error) {
				toast.error(`Error initializing app: ${error}`) // Display a toast error message if navigation fails
			}
		}
		initializeApp() // Call initializeApp when the component mounts
	}, [router]) // Dependency array includes 'router' to satisfy hook dependencies, but effect is intended to run only once on mount

	return (
		<div className="flex flex-col items-center justify-center min-h-screen bg-black">
			<AnimatedLogo /> {/* Render the AnimatedLogo component */}
			<h1 className="text-white text-4xl mt-4">Sentient</h1>{" "}
			{/* Display the title of the application */}
		</div>
	)
}

export default Home
