"use client"
import React, { useState, useEffect } from "react"
import { IconLoader } from "@tabler/icons-react"
import toast from "react-hot-toast"
import Sidebar from "@components/Sidebar" // Import the Sidebar component

const Notifications = () => {
	const [notifications, setNotifications] = useState([])
	const [isLoading, setIsLoading] = useState(true)
	const [userDetails, setUserDetails] = useState({}) // State for user details
	const [isSidebarVisible, setSidebarVisible] = useState(false) // State for sidebar visibility

	const fetchNotifications = async () => {
		setIsLoading(true)
		try {
			// Assuming you have an IPC handler named "get-notifications"
			// If not, you'll need to create one in your Electron main process (index.js)
			// that fetches notifications, perhaps from a database or state management.
			// For now, we'll simulate a response or assume it exists.

			// Example simulation (replace with actual IPC call):
			// const response = { status: 200, notifications: [
			//     { id: 1, message: "Task 'Analyze report' completed.", timestamp: "2023-10-27 10:00:00" },
			//     { id: 2, message: "Memory 'Project X details' added.", timestamp: "2023-10-27 09:30:00" },
			// ] };

			// Replace simulation with actual call if handler exists:
			const response = await window.electron?.invoke("get-notifications") // Make sure this handler exists

			if (!response) {
				// Handle case where electron API is not available or handler doesn't exist
				toast.error("Notification service not available.")
				setNotifications([]) // Set to empty array
			} else if (
				response.status === 200 &&
				Array.isArray(response.notifications)
			) {
				setNotifications(response.notifications)
			} else {
				console.error(
					"Error fetching notifications:",
					response?.error || "Unknown error"
				)
				toast.error(response?.error || "Error fetching notifications.")
				setNotifications([]) // Ensure state is an array even on error
			}
		} catch (error) {
			console.error("Catch Error fetching notifications:", error)
			toast.error(`Error fetching notifications: ${error.message}`)
			setNotifications([]) // Ensure state is an array on catch
		} finally {
			setIsLoading(false)
		}
	}

	// Function to fetch user details for the sidebar
	const fetchUserDetails = async () => {
		try {
			const response = await window.electron?.invoke("get-profile")
			if (response) {
				setUserDetails(response)
			} else {
				// Handle case where profile couldn't be fetched (e.g., logged out in dev mode)
				setUserDetails({}) // Set to empty object or default state
			}
		} catch (error) {
			toast.error("Error fetching user details for sidebar.")
			console.error("Error fetching user details for sidebar:", error)
			setUserDetails({}) // Set to empty object on error
		}
	}

	useEffect(() => {
		fetchNotifications()
		fetchUserDetails() // Fetch user details when component mounts

		// Optional: Set up an interval or WebSocket listener to refresh notifications
		// const intervalId = setInterval(fetchNotifications, 60000); // Example: refresh every minute
		// return () => clearInterval(intervalId);
	}, [])

	return (
		// Main flex container for sidebar and content
		<div className="h-screen bg-matteblack flex relative">
			{/* Sidebar Component */}
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
				fromChat={false} // Assuming this page is not the chat interface
			/>

			{/* Content Area */}
			<div className="w-4/5 flex flex-col items-start h-full bg-matteblack ml-5 py-8 px-6">
				{" "}
				{/* Added padding */}
				<h1 className="text-4xl font-bold text-white mb-6">
					Notifications
				</h1>
				{isLoading ? (
					<div className="flex justify-center items-center w-full h-full">
						{" "}
						{/* Center loader within content area */}
						<IconLoader className="w-10 h-10 text-white animate-spin" />
						<span className="ml-3 text-white text-lg">
							Loading Notifications...
						</span>
					</div>
				) : notifications.length === 0 ? (
					<div className="flex justify-center items-center w-full h-full">
						<p className="text-gray-500 text-lg">
							No new notifications
						</p>
					</div>
				) : (
					// Container for the list with scrolling
					<div className="w-full bg-gray-900 rounded-lg shadow-xl max-h-[80vh] overflow-y-auto no-scrollbar">
						<ul className="p-4">
							{" "}
							{/* Padding inside the scrollable list */}
							{notifications.map((notif) => (
								<li
									key={notif.id} // Ensure notifications have a unique ID
									className="mb-3 p-4 bg-gray-800 rounded-md border-l-4 border-blue-500 shadow-sm" // Added border for emphasis
								>
									<p className="text-white text-md mb-1">
										{notif.message}
									</p>
									<p className="text-gray-400 text-xs">
										{/* Format timestamp if needed */}
										{new Date(
											notif.timestamp
										).toLocaleString() || "No timestamp"}
									</p>
								</li>
							))}
						</ul>
					</div>
				)}
			</div>
		</div>
	)
}

export default Notifications