"use client"
import React, { useState, useEffect, useCallback } from "react"
// Import necessary icons
import { IconLoader, IconBell, IconAlertCircle } from "@tabler/icons-react" // Added IconAlertCircle
import toast from "react-hot-toast"
import Sidebar from "@components/Sidebar"
import { cn } from "@utils/cn"

const Notifications = () => {
	const [notifications, setNotifications] = useState([])
	const [isLoading, setIsLoading] = useState(true) // Start loading initially
	const [userDetails, setUserDetails] = useState({})
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	const [error, setError] = useState(null)

	// Fetch notifications function wrapped in useCallback
	const fetchNotifications = useCallback(async () => {
		console.log("Fetching notifications...")
		// Don't set loading here based on previous state - set it unconditionally
		// We handle the initial loading state separately
		setIsLoading(true) // Indicate loading start
		setError(null) // Clear previous errors
		try {
			const response = await window.electron?.invoke("get-notifications")

			if (!response) {
				// Handle case where electron API is not available
				const errorMsg = "Notification service not available."
				toast.error(errorMsg)
				setNotifications([])
				setError(errorMsg)
			} else if (
				response.status === 200 &&
				Array.isArray(response.notifications)
			) {
				// Sort notifications by timestamp, newest first
				const sortedNotifications = response.notifications.sort(
					(a, b) => {
						try {
							return (
								new Date(b.timestamp).getTime() -
								new Date(a.timestamp).getTime()
							)
						} catch {
							return 0
						} // Keep order if dates invalid
					}
				)
				setNotifications(sortedNotifications)
				console.log(
					"Notifications fetched:",
					sortedNotifications.length
				)
			} else {
				// Handle backend errors or unexpected response structure
				const errorMsg =
					response?.error || "Unknown error fetching notifications"
				console.error("Error fetching notifications:", errorMsg)
				toast.error(errorMsg)
				setNotifications([])
				setError(errorMsg)
			}
		} catch (error) {
			// Handle network or other exceptions during fetch
			console.error("Catch Error fetching notifications:", error)
			const errorMsg = `Error fetching notifications: ${error.message}`
			toast.error(errorMsg)
			setNotifications([])
			setError(errorMsg)
		} finally {
			// Always set loading to false after fetch attempt completes
			setIsLoading(false)
		}
		// MODIFIED: Removed isLoading from dependency array
	}, []) // useCallback dependency array is now empty

	// Function to fetch user details (remains the same)
	const fetchUserDetails = async () => {
		try {
			const response = await window.electron?.invoke("get-profile")
			setUserDetails(response || {})
		} catch (error) {
			toast.error("Error fetching user details for sidebar.")
			console.error("Error fetching user details for sidebar:", error)
			setUserDetails({})
		}
	}

	// Initial fetch and interval setup effect
	useEffect(() => {
		fetchUserDetails()
		fetchNotifications() // Call fetch on initial mount

		// Setup interval for subsequent refreshes
		const intervalId = setInterval(fetchNotifications, 120000) // Refresh every 2 minutes
		return () => clearInterval(intervalId) // Cleanup interval on unmount
	}, [fetchNotifications]) // Dependency array correctly contains fetchNotifications

	// Helper function to format timestamp (remains the same)
	const formatTimestamp = (timestamp) => {
		if (!timestamp) return "No timestamp"
		try {
			return new Date(timestamp).toLocaleString(undefined, {
				dateStyle: "medium",
				timeStyle: "short"
			})
		} catch (e) {
			console.warn("Error formatting timestamp:", timestamp, e)
			return timestamp
		}
	}

	// --- Render Logic ---
	return (
		<div className="h-screen bg-matteblack flex relative overflow-hidden dark">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
			/>

			<div className="flex-grow flex flex-col h-full bg-matteblack text-white relative overflow-hidden p-6 md:p-8">
				<h1 className="text-3xl md:text-4xl font-light text-white mb-6 md:mb-8 px-4 text-center md:text-left">
					Notifications
				</h1>

				<div className="w-full max-w-3xl mx-auto flex-grow overflow-hidden flex flex-col">
					{/* MODIFIED: Show loader if isLoading is true, regardless of notifications.length */}
					{isLoading ? (
						<div className="flex-grow flex flex-col justify-center items-center text-center p-10">
							<IconLoader className="w-10 h-10 text-lightblue animate-spin" />
							<span className="ml-3 text-gray-400 text-lg mt-4">
								Loading Notifications...
							</span>
						</div>
					) : error ? ( // Display error message if error state is set
						<div className="flex-grow flex flex-col justify-center items-center text-center p-10">
							<IconAlertCircle className="w-12 h-12 text-red-500 mb-4" />
							<p className="text-red-400 text-lg mb-4">
								Could not load notifications
							</p>
							<p className="text-gray-500 text-sm mb-6">
								{error}
							</p>
							<button
								onClick={fetchNotifications} // Allow retry
								className="py-2 px-5 rounded bg-lightblue hover:bg-blue-700 text-white text-sm font-medium transition-colors"
							>
								Retry
							</button>
						</div>
					) : notifications.length === 0 ? ( // Display empty state if no error and not loading
						<div className="flex-grow flex flex-col justify-center items-center text-center p-10">
							<IconBell className="w-16 h-16 text-neutral-700 mb-4" />
							<p className="text-gray-500 text-xl">
								No new notifications
							</p>
							<p className="text-gray-600 text-sm mt-2">
								Check back later for updates.
							</p>
						</div>
					) : (
						// Display notification list
						<div className="flex-grow overflow-y-auto space-y-3 pr-2 custom-scrollbar">
							{notifications.map((notif) => (
								<div
									key={notif.id ?? Math.random()}
									className="flex items-start gap-4 bg-neutral-800 p-4 rounded-lg border border-neutral-700/50 shadow-sm"
								>
									<div className="flex-shrink-0 pt-1">
										<IconBell className="w-5 h-5 text-lightblue" />
									</div>
									<div className="flex-grow">
										<p className="text-white text-sm leading-relaxed mb-1">
											{notif.message ||
												"No message content."}
										</p>
										<p className="text-gray-500 text-xs">
											{formatTimestamp(notif.timestamp)}
										</p>
									</div>
								</div>
							))}
						</div>
					)}
				</div>
			</div>
		</div>
	)
}

export default Notifications
