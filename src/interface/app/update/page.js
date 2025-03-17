"use client"

import { useRouter } from "next/navigation" // Hook for client-side navigation in Next.js
import { useState, useEffect, useRef } from "react" // Importing necessary React hooks and useRef
import toast from "react-hot-toast" // Library for displaying toast notifications
import React from "react"

/**
 * Update Component - Handles application updates and the update checking screen.
 *
 * This component checks for application updates, manages the update process, and displays a loading/update screen to the user.
 * It listens for update-related events from the Electron backend and navigates the user to the main application after updates are handled.
 *
 * @returns {React.ReactNode} - The Update component UI.
 */
const Update = () => {
	// State to track if an update is available - updateAvailable: boolean
	const [updateAvailable, setUpdateAvailable] = useState(false)
	// useRef to persist updateAvailable status across renders without causing re-renders.
	const updateAvailableRef = useRef(false) // updateAvailableRef: React.RefObject<boolean>
	const router = useRouter() // Hook to get the router object for navigation
	// useRef to track if event listeners have been added to avoid adding them multiple times.
	const eventListenersAdded = useRef(false) // eventListenersAdded: React.RefObject<boolean>

	/**
	 * Handles the update process, including checking for updates and applying them.
	 *
	 * Sets up listeners for 'update-available' and 'update-downloaded' events from the electron backend.
	 * When an update is available, it sets `updateAvailable` to true.
	 * When an update is downloaded, it invokes backend to setup server, alerts user to restart app,
	 * and resolves a promise after a timeout, indicating update handling completion.
	 * Implements a timeout to reject the promise if update handling takes too long.
	 *
	 * @async
	 * @function handleUpdates
	 * @returns {Promise<void>}
	 * @throws {string} - Throws "Update handling timed out" if update process exceeds timeout.
	 */
	const handleUpdates = async () => {
		try {
			// Create a promise to handle asynchronous update process with timeout
			const updatePromise = new Promise((resolve, reject) => {
				// Add event listeners only once
				if (!eventListenersAdded.current) {
					// Listener for 'update-available' event from electron backend
					window.electron.onUpdateAvailable(() => {
						// Prevent state update if already updated or during initial check
						if (!updateAvailableRef.current) {
							setUpdateAvailable(true) // Set state to indicate update is available
							updateAvailableRef.current = true // Set ref to prevent redundant state updates
						}
					})

					// Listener for 'update-downloaded' event from electron backend
					window.electron.onUpdateDownloaded(async () => {
						try {

							// Alert user to restart application to apply update
							alert(
								"Update downloaded. Please restart the application to apply the update."
							)

							// Timeout to allow user to read alert before resolving promise and resetting state
							setTimeout(() => {
								setUpdateAvailable(false) // Reset updateAvailable state
								updateAvailableRef.current = false // Reset updateAvailableRef
								resolve() // Resolve promise to indicate update handling is complete
							}, 3000) // 3 seconds timeout
						} catch (setupError) {
							toast.error(
								`Error in setting up server: ${setupError}`
							) // Show error toast if server setup fails
							reject(setupError) // Reject promise if server setup fails
						}
					})

					eventListenersAdded.current = true // Mark event listeners as added
				}

				// Timeout to reject promise if update handling takes too long (e.g., network issues)
				setTimeout(() => {
					reject("Update handling timed out") // Reject promise after timeout
				}, 10000) // 10 seconds timeout for update handling
			})

			await updatePromise // Wait for the update promise to resolve or reject
		} catch (error) {
			// Handle errors during update process, but ignore timeout errors as they are expected
			if (error !== "Update handling timed out") {
				toast.error(`Error during update handling: ${error}`) // Show error toast for non-timeout errors
			}
		}
	}

	/**
	 * Handles the page flow after checking for updates.
	 *
	 * Calls `handleUpdates` to initiate the update check process. After updates are handled (or if no update is available),
	 * it navigates the user to the '/load' page using next/navigation router, unless an update is in progress (`updateAvailableRef.current` is true).
	 *
	 * @async
	 * @function handlePageFlow
	 * @returns {Promise<void>}
	 */
	const handlePageFlow = async () => {
		await handleUpdates() // Await handleUpdates to complete update check and handling
		if (!updateAvailableRef.current) {
			router.push("/load") // Navigate to '/load' page if no update is in progress
		}
	}

	/**
	 * useEffect hook to initiate page flow on component mount and when updateAvailable state changes.
	 *
	 * Calls `handlePageFlow` when the component is first rendered and whenever the `updateAvailable` state changes.
	 * The dependency array `[updateAvailable]` ensures that page flow is re-evaluated in response to update availability changes.
	 */
	useEffect(() => {
		handlePageFlow() // Call handlePageFlow to start update check and page navigation
	}, [updateAvailable]) // Effect dependency: updateAvailable - effect runs when updateAvailable changes

	/**
	 * Main return statement for the Update component, rendering the update checking UI.
	 *
	 * Displays a full-screen loading spinner and message indicating update check or update in progress.
	 * The message dynamically updates based on the `updateAvailable` state.
	 *
	 * @returns {React.ReactNode} - The main UI for the Update component.
	 */
	return (
		<div
			className="flex flex-col items-center justify-center min-h-screen"
			style={{
				position: "fixed", // Fixed position to cover entire viewport
				top: 0,
				left: 0,
				width: "100%",
				height: "100%",
				display: "flex", // Flexbox for centering content
				justifyContent: "center", // Center content horizontally
				alignItems: "center", // Center content vertically
				zIndex: 1000 // High z-index to ensure it's on top of other content
			}}
		>
			<div
				style={{
					textAlign: "center" // Center text within the container
				}}
			>
				<div
					className="spinner"
					style={{
						width: "50px", // Width of the spinner
						height: "50px", // Height of the spinner
						border: "5px solid rgba(255, 255, 255, 0.2)", // Border style
						borderTop: "5px solid #fff", // Top border color to create spinner effect
						borderRadius: "50%", // Make it round
						margin: "0 auto 20px", // Center spinner horizontally and add margin below
						animation: "spin 1s linear infinite" // Animation for spinner
					}}
				></div>
				<h2
					style={{
						fontSize: "20px", // Font size for heading
						fontWeight: "bold", // Bold font weight
						marginBottom: "10px", // Margin at the bottom
						color: "#fff" // Text color white
					}}
				>
					{updateAvailable
						? "An update is in progress..." // Display message if update is available
						: "Checking for updates..."}{" "}
					{/* Default message when checking for updates */}
				</h2>
				<p style={{ fontSize: "16px", color: "#fff" }}>
					{updateAvailable
						? "Please wait while we configure the application." // Message during update configuration
						: "Ensuring your app is up-to-date."}{" "}
					{/* Default message during update check */}
				</p>
			</div>
			{/* Inline style for keyframes animation */}
			<style>
				{`
          @keyframes spin {
            0% {
              transform: rotate(0deg);
            }
            100% {
              transform: rotate(360deg);
            }
          }
        `}
			</style>
		</div>
	)
}

export default Update
