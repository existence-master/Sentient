"use client"

import { useRouter } from "next/navigation"
import { useState, useEffect, useRef } from "react"
import toast from "react-hot-toast"
import React from "react"

/**
 * Update Component - Handles application updates and displays progress.
 */
const Update = () => {
	const [updateAvailable, setUpdateAvailable] = useState(false)
	const [progress, setProgress] = useState(0) // State for download progress
	const [error, setError] = useState(null) // State for update errors

	const updateAvailableRef = useRef(false)
	const router = useRouter()
	const eventListenersAdded = useRef(false)
	const cleanupProgressListener = useRef(null) // Ref to store cleanup function

	/**
	 * Handles the update process, including checking for updates and applying them.
	 */

	/**
	 * useEffect hook to initiate update checks and set up listeners on component mount.
	 */
	useEffect(() => {
		let isMounted = true // Flag to check if component is still mounted in async operations

		if (!eventListenersAdded.current) {
			console.log("Setting up Electron update listeners...")

			// Listener for 'update-available'
			window.electron.onUpdateAvailable(() => {
				if (isMounted && !updateAvailableRef.current) {
					console.log("Update available detected.")
					setUpdateAvailable(true)
					updateAvailableRef.current = true
					setError(null) // Clear any previous error
					setProgress(0) // Reset progress
				}
			})

			// Listener for 'update-downloaded'
			window.electron.onUpdateDownloaded(async () => {
				if (isMounted) {
					console.log("Update downloaded detected.")
					setProgress(100) // Show 100% briefly
					// Alert user to restart application to apply update
					// Consider using a more integrated UI element than alert()
					alert(
						"Update downloaded. Please restart the application to apply the update."
					)
					// Optionally trigger restart automatically or provide a button
					// window.electron.restartApp();

					// Reset state and potentially navigate away or wait for user action
					// For now, we stay on this page showing the message.
					// If you want to navigate after the alert:
					// setTimeout(() => {
					//     if (isMounted) {
					// 		   setUpdateAvailable(false);
					//         updateAvailableRef.current = false;
					//         setProgress(0);
					//         // router.push('/load'); // Or wherever needed
					// 	   }
					// }, 3000);
				}
			})

			// Listener for 'update-progress'
			// Store the cleanup function returned by onUpdateProgress
			cleanupProgressListener.current = window.electron.onUpdateProgress(
				(progressObj) => {
					if (isMounted && updateAvailableRef.current) {
						// console.log("Update progress:", progressObj.percent); // Can be verbose
						setProgress(Math.round(progressObj.percent))
					}
				}
			)

			// Listener for 'update-error'
			window.electron.onUpdateError((event, errorMessage) => {
				if (isMounted) {
					console.error("Update error detected:", errorMessage)
					setError(`Update failed: ${errorMessage}`)
					toast.error(`Update failed: ${errorMessage}`)
					setUpdateAvailable(false) // No longer updating
					updateAvailableRef.current = false
					setProgress(0) // Reset progress
					// Maybe navigate away after a delay or show error state
					setTimeout(() => {
						if (isMounted && !updateAvailableRef.current) {
							// Check again in case a new update became available
							router.push("/load") // Navigate after showing error
						}
					}, 5000) // Wait 5 seconds before navigating
				}
			})

			eventListenersAdded.current = true
			console.log("Electron update listeners added.")
		}

		// Initial check simulation or trigger (if needed)
		// The main process's checkForUpdates usually triggers the events automatically.
		// handlePageFlow(); // Call if it does essential setup

		// --- Navigation Logic ---
		// Decide when to navigate away. Example: navigate after 10s if no update found.
		const navigationTimeout = setTimeout(() => {
			// Check if component is still mounted and no update is active/pending
			if (isMounted && !updateAvailableRef.current && !error) {
				console.log(
					"No update detected or process finished without error, navigating..."
				)
				router.push("/load")
			} else if (isMounted && error) {
				console.log(
					"Update error occurred, navigation handled by error listener."
				)
			} else if (isMounted && updateAvailableRef.current) {
				console.log("Update in progress, staying on update page.")
			}
		}, 10000) // Navigate after 10 seconds if nothing happens

		// Cleanup function
		return () => {
			console.log("Cleaning up Update component listeners...")
			isMounted = false // Set flag to false when component unmounts
			clearTimeout(navigationTimeout) // Clear the navigation timeout
			// Call the cleanup function returned by onUpdateProgress if it exists
			if (cleanupProgressListener.current) {
				cleanupProgressListener.current()
			}
			// Note: The current preload structure doesn't provide easy ways to remove
			// listeners added via onUpdateAvailable/Downloaded/Error.
			// If this component mounts/unmounts frequently, you might need
			// to enhance preload.js to return cleanup functions for all listeners
			// or use ipcRenderer.removeAllListeners('channel-name') carefully.
			// For a typical update screen shown once, this might be acceptable.
		}
	}, [router]) // Dependencies: router (stable)

	/**
	 * Main return statement for the Update component UI.
	 */
	return (
		<div
			className="flex flex-col items-center justify-center min-h-screen bg-black text-white" // Added bg/text color
			style={{
				position: "fixed",
				top: 0,
				left: 0,
				width: "100%",
				height: "100%",
				display: "flex",
				justifyContent: "center",
				alignItems: "center",
				zIndex: 1000
			}}
		>
			<div
				style={{
					textAlign: "center",
					padding: "20px" // Added padding
				}}
			>
				{!error ? (
					<>
						<div
							className="spinner"
							style={{
								width: "50px",
								height: "50px",
								border: "5px solid rgba(255, 255, 255, 0.2)",
								borderTop: "5px solid #fff",
								borderRadius: "50%",
								margin: "0 auto 20px",
								animation: "spin 1s linear infinite",
								display:
									updateAvailable &&
									progress > 0 &&
									progress < 100
										? "none"
										: "block" // Hide spinner when progress bar shows
							}}
						></div>

						{/* Progress Bar and Text */}
						{updateAvailable && progress > 0 && progress < 100 && (
							<div
								style={{
									marginBottom: "20px",
									width: "80%",
									maxWidth: "300px",
									margin: "0 auto 20px"
								}}
							>
								<div
									style={{
										height: "10px",
										backgroundColor:
											"rgba(255, 255, 255, 0.2)",
										borderRadius: "5px",
										overflow: "hidden",
										marginBottom: "5px"
									}}
								>
									<div
										style={{
											width: `${progress}%`,
											height: "100%",
											backgroundColor: "#fff",
											borderRadius: "5px",
											transition: "width 0.2s ease-in-out" // Smooth transition
										}}
									></div>
								</div>
								<p style={{ fontSize: "14px", color: "#ccc" }}>
									Downloading update... {progress}%
								</p>
							</div>
						)}

						<h2
							style={{
								fontSize: "20px",
								fontWeight: "bold",
								marginBottom: "10px",
								color: "#fff"
							}}
						>
							{updateAvailable
								? progress === 100
									? "Update Downloaded"
									: "Update Available"
								: "Checking for updates..."}
						</h2>
						<p style={{ fontSize: "16px", color: "#ccc" }}>
							{updateAvailable
								? progress > 0 && progress < 100
									? "Downloading the latest version."
									: progress === 100
										? "Preparing update. Please restart the app when prompted."
										: "An update is available and will be downloaded."
								: "Ensuring your app is up-to-date. Please wait..."}
						</p>
					</>
				) : (
					// Error State Display
					<div style={{ color: "#ff6b6b" }}>
						<h2
							style={{
								fontSize: "20px",
								fontWeight: "bold",
								marginBottom: "10px"
							}}
						>
							Update Error
						</h2>
						<p style={{ fontSize: "16px" }}>{error}</p>
						<p
							style={{
								fontSize: "14px",
								marginTop: "15px",
								color: "#ccc"
							}}
						>
							The application will proceed with the current
							version shortly.
						</p>
					</div>
				)}
			</div>
			{/* Inline style for keyframes animation */}
			<style>
				{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}
			</style>
		</div>
	)
}

export default Update