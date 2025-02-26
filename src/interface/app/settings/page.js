"use client"

import { useState, useEffect } from "react" // Importing necessary React hooks
import Disclaimer from "@components/Disclaimer" // Component for displaying a disclaimer for data usage
import AppCard from "@components/AppCard" // Reusable component to display app integration cards
import Sidebar from "@components/Sidebar" // Sidebar component for navigation
import ProIcon from "@components/ProIcon" // Component to indicate Pro features
import toast from "react-hot-toast" // Library for displaying toast notifications
import ShiningButton from "@components/ShiningButton" // Custom button component with a shining effect
import ModalDialog from "@components/ModalDialog"
import {
	IconGift,
	IconBeta,
	IconRocket
} from "@node_modules/@tabler/icons-react/dist/esm/tabler-icons-react" // Icons from tabler-icons-react library
import React from "react"

/**
 * Settings Component - Manages app settings, including social media integrations and referral program.
 *
 * This component allows users to connect/disconnect their social media profiles (LinkedIn, Reddit, Twitter),
 * view referral information, and toggle beta user status. It also handles disclaimers and loading states for UI interactions.
 *
 * @returns {React.ReactNode} - The Settings component UI.
 */
const Settings = () => {
	// State to control visibility of the disclaimer modal - showDisclaimer: boolean
	const [showDisclaimer, setShowDisclaimer] = useState(false)
	// State to store LinkedIn profile URL input - linkedInProfileUrl: string
	const [linkedInProfileUrl, setLinkedInProfileUrl] = useState("")
	// State to store Reddit profile URL input - redditProfileUrl: string
	const [redditProfileUrl, setRedditProfileUrl] = useState("")
	// State to store Twitter profile URL input - twitterProfileUrl: string
	const [twitterProfileUrl, setTwitterProfileUrl] = useState("")
	// State to track connection status of social media profiles - isProfileConnected: { LinkedIn: boolean, Reddit: boolean, Twitter: boolean }
	const [isProfileConnected, setIsProfileConnected] = useState({
		LinkedIn: false,
		Reddit: false,
		Twitter: false
	})
	// State to define the action type (connect or disconnect) - action: "connect" | "disconnect" | ""
	const [action, setAction] = useState("")
	// State to manage loading status for each app card - loading: { LinkedIn: boolean, Reddit: boolean, Twitter: boolean }
	const [loading, setLoading] = useState({
		LinkedIn: false,
		Reddit: false,
		Twitter: false
	})
	// State to store the name of the selected app for actions - selectedApp: string ("LinkedIn" | "Reddit" | "Twitter" | "")
	const [selectedApp, setSelectedApp] = useState("")
	// State to store user details, fetched from the backend - userDetails: any - Structure depends on backend response
	const [userDetails, setUserDetails] = useState({})
	// State to manage sidebar visibility - isSidebarVisible: boolean
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	// State to store the user's pricing plan, defaults to "free" - pricing: string ("free" | "pro")
	const [pricing, setPricing] = useState("free")
	// State to control visibility of the referral dialog - showReferralDialog: boolean
	const [showReferralDialog, setShowReferralDialog] = useState(false)
	// State to store the referral code - referralCode: string
	const [referralCode, setReferralCode] = useState("")
	// State to track referrer status - referrerStatus: boolean
	const [referrerStatus, setReferrerStatus] = useState(false)
	// State to track beta user status - betaUser: boolean
	const [betaUser, setBetaUser] = useState(false)
	// State to control visibility of the beta user dialog - showBetaDialog: boolean
	const [showBetaDialog, setShowBetaDialog] = useState(false)

	const [availableModels, setAvailableModels] = useState([])
	const [selectedModel, setSelectedModel] = useState("llama3.2:3b") // Default model

	/**
	 * Fetches user details from the backend.
	 *
	 * Uses electron invoke to call the backend function to get user profile information and updates the `userDetails` state.
	 * Handles errors by displaying a toast notification.
	 *
	 * @async
	 * @function fetchUserDetails
	 * @returns {Promise<void>}
	 */
	const fetchUserDetails = async () => {
		try {
			const response = await window.electron?.invoke("get-profile") // Invoke electron backend to get user profile
			setUserDetails(response) // Update userDetails state with fetched data
		} catch (error) {
			toast.error("Error fetching user details.") // Show error toast if fetching fails
		}
	}

	/**
	 * Fetches the user's pricing plan.
	 *
	 * Retrieves the pricing plan information from the backend using electron invoke and sets the `pricing` state.
	 * Defaults to "free" if no plan is fetched or in case of an error.
	 *
	 * @async
	 * @function fetchPricingPlan
	 * @returns {Promise<void>}
	 */
	const fetchPricingPlan = async () => {
		try {
			const response = await window.electron?.invoke("fetch-pricing-plan") // Invoke electron backend to get pricing plan
			setPricing(response || "free") // Set pricing state with fetched plan or default to 'free'
		} catch (error) {
			toast.error("Error fetching pricing plan.") // Show error toast if fetching fails
		}
	}

	/**
	 * Fetches the beta user status from the backend.
	 *
	 * Calls the electron backend to check if the user is a beta user and updates the `betaUser` state accordingly.
	 * Handles errors with a toast notification.
	 *
	 * @async
	 * @function fetchBetaUserStatus
	 * @returns {Promise<void>}
	 */
	const fetchBetaUserStatus = async () => {
		try {
			const response = await window.electron?.invoke(
				"get-beta-user-status"
			) // Invoke electron backend to get beta user status
			setBetaUser(response === true) // Set betaUser state based on backend response
		} catch (error) {
			toast.error("Error fetching beta user status.") // Show error toast if fetching fails
		}
	}

	/**
	 * Fetches referral details including referral code and referrer status.
	 *
	 * Retrieves referral code and referrer status from the backend using electron invoke and updates corresponding states.
	 * Defaults referral code to "N/A" if not available. Handles errors with a toast notification.
	 *
	 * @async
	 * @function fetchReferralDetails
	 * @returns {Promise<void>}
	 */
	const fetchReferralDetails = async () => {
		try {
			const referral = await window.electron?.invoke(
				"fetch-referral-code"
			) // Invoke electron backend to get referral code
			const referrer = await window.electron?.invoke(
				"fetch-referrer-status"
			) // Invoke electron backend to get referrer status
			setReferralCode(referral || "N/A") // Set referralCode state with fetched code or default to 'N/A'
			setReferrerStatus(referrer === true) // Set referrerStatus state based on backend response
		} catch (error) {
			toast.error("Error fetching referral details.") // Show error toast if fetching fails
		}
	}

	/**
	 * Toggles the beta user status.
	 *
	 * Invokes the electron backend to invert the current beta user status, updates the `betaUser` state,
	 * and displays a success toast message indicating the status change.
	 * Handles errors with a toast notification and closes the beta dialog.
	 *
	 * @async
	 * @function handleBetaUserToggle
	 * @returns {Promise<void>}
	 */
	const handleBetaUserToggle = async () => {
		try {
			await window.electron?.invoke("invert-beta-user-status") // Invoke electron backend to invert beta user status
			setBetaUser((prev) => !prev) // Toggle betaUser state
			toast.success(
				betaUser
					? "You have exited the Beta User Program."
					: "You are now a Beta User!" // Display success toast based on new beta user status
			)
		} catch (error) {
			toast.error("Error updating beta user status.") // Show error toast if update fails
		}
		setShowBetaDialog(false) // Close beta dialog after attempting to toggle status
	}

	/**
	 * Fetches user data including connected social media profiles.
	 *
	 * Retrieves user data from the database using electron invoke and updates the `isProfileConnected` state
	 * to reflect whether LinkedIn, Reddit, and Twitter profiles are connected.
	 * Handles errors with a toast notification.
	 *
	 * @async
	 * @function fetchData
	 * @returns {Promise<void>}
	 */
	const fetchData = async () => {
		try {
			const response = await window.electron?.invoke("get-db-data")
			if (response.status === 200 && response.data) {
				const {
					linkedInProfile,
					redditProfile,
					twitterProfile,
					selectedModel
				} = response.data
				setIsProfileConnected({
					LinkedIn:
						linkedInProfile &&
						Object.keys(linkedInProfile).length > 0,
					Reddit:
						redditProfile && Object.keys(redditProfile).length > 0,
					Twitter:
						twitterProfile && Object.keys(twitterProfile).length > 0
				})
				// Set selectedModel, default to "llama3.2:3b" if not present
				setSelectedModel(selectedModel || "llama3.2:3b")
			}
		} catch (error) {
			toast.error("Error fetching user data.")
		}
	}

	// Fetch available models from Ollama
	const fetchAvailableModels = async () => {
		try {
			const models = await window.electron.invoke("get-ollama-models")
			setAvailableModels(models)
			// If selectedModel isn't in availableModels, reset to default or first available
			if (models.length > 0 && !models.includes(selectedModel)) {
				const newModel = models[0] // Default to first available model
				await window.electron.invoke("set-db-data", {
					data: { selectedModel: newModel }
				})
				setSelectedModel(newModel)
			}
		} catch (error) {
			toast.error(
				"Error fetching available models. Ensure Ollama is running."
			)
			setAvailableModels([])
		}
	}

	/**
	 * useEffect hook to fetch user details, pricing plan, referral, and beta user status on component mount.
	 *
	 * Calls various data fetching functions to initialize settings page with user-specific information.
	 */
	useEffect(() => {
		fetchData()
		fetchUserDetails() // Fetch user details on component mount
		fetchPricingPlan() // Fetch pricing plan on component mount
		fetchReferralDetails() // Fetch referral details on component mount
		fetchBetaUserStatus() // Fetch beta user status on component mount
		fetchAvailableModels()
	}, []) // Empty dependency array ensures this effect runs only once on mount

	/**
	 * Handles the click event for connecting a social media app.
	 *
	 * For free users, connecting Reddit or Twitter is restricted and displays an error toast.
	 * For all users, it sets the `selectedApp`, `action` to "connect", and shows the disclaimer modal.
	 *
	 * @function handleConnectClick
	 * @param {string} appName - The name of the app to connect (e.g., "LinkedIn", "Reddit", "Twitter").
	 * @returns {void}
	 */
	const handleConnectClick = (appName) => {
		if (
			pricing === "free" &&
			(appName === "Reddit" || appName === "Twitter")
		) {
			toast.error("This feature is only available for Pro users.") // Show error toast for free users trying to connect Pro features
			return // Prevent further action for free users on Pro features
		}
		setShowDisclaimer(true) // Show disclaimer modal before connecting
		setSelectedApp(appName) // Set selectedApp state to the app being connected
		setAction("connect") // Set action state to "connect"
	}

	/**
	 * Handles the click event for disconnecting a social media app.
	 *
	 * Sets the `selectedApp`, `action` to "disconnect", and shows the disclaimer modal to confirm disconnection.
	 *
	 * @function handleDisconnectClick
	 * @param {string} appName - The name of the app to disconnect (e.g., "LinkedIn", "Reddit", "Twitter").
	 * @returns {void}
	 */
	const handleDisconnectClick = (appName) => {
		setShowDisclaimer(true) // Show disclaimer modal before disconnecting
		setSelectedApp(appName) // Set selectedApp state to the app being disconnected
		setAction("disconnect") // Set action state to "disconnect"
	}

	/**
	 * Handles the acceptance of the disclaimer and proceeds with connecting or disconnecting the selected app profile.
	 *
	 * Sets the `showDisclaimer` state to false to hide the disclaimer modal.
	 * Sets the loading state for the selected app to true to indicate the action is in progress.
	 * Depending on the `action` state ("connect" or "disconnect") and the `selectedApp`, it performs the following:
	 * - For "connect" action, it scrapes the profile data using electron invoke based on the `selectedApp` (LinkedIn, Reddit, Twitter),
	 *   stores the scraped data in the database, updates the `isProfileConnected` state, and shows a success toast message.
	 * - For "disconnect" action, it clears the corresponding profile data in the database, deletes the subgraph for the disconnected profile,
	 *   updates the `isProfileConnected` state, and shows a success toast message.
	 * After successfully connecting or disconnecting, it also invokes electron to create/update document and graph.
	 * In case of any error during the process, it shows an error toast message.
	 * Finally, it resets the loading state for the selected app to false in the finally block.
	 *
	 * @async
	 * @function handleDisclaimerAccept
	 * @returns {Promise<void>}
	 */
	const handleDisclaimerAccept = async () => {
		setShowDisclaimer(false) // Hide the disclaimer modal
		setLoading((prev) => ({ ...prev, [selectedApp]: true })) // Set loading state for the selected app to true

		try {
			let successMessage = "" // Variable to hold success message
			if (action === "connect") {
				// If action is "connect"
				let response = null // Variable to hold electron invoke response

				if (selectedApp === "LinkedIn") {
					// If selected app is LinkedIn
					response = await window.electron?.invoke(
						"scrape-linkedin",
						{
							// Invoke electron to scrape LinkedIn profile
							linkedInProfileUrl // Pass LinkedIn profile URL to backend
						}
					)

					if (response.status === 200) {
						// If response status is 200 (success)
						await window.electron?.invoke("set-db-data", {
							// Invoke electron to set data in database
							data: { linkedInProfile: response.profile } // Store scraped LinkedIn profile data in database
						})
						successMessage =
							"LinkedIn profile connected successfully." // Set success message for LinkedIn connection
					} else {
						throw new Error("Error scraping LinkedIn profile") // Throw error if response status is not 200
					}
				} else if (selectedApp === "Reddit") {
					// If selected app is Reddit
					response = await window.electron?.invoke("scrape-reddit", {
						// Invoke electron to scrape Reddit profile
						redditProfileUrl // Pass Reddit profile URL to backend
					})

					if (response.status === 200) {
						// If response status is 200 (success)
						await window.electron?.invoke("set-db-data", {
							// Invoke electron to set data in database
							data: { redditProfile: response.topics } // Store scraped Reddit topics data in database
						})
						successMessage =
							"Reddit profile connected successfully." // Set success message for Reddit connection
					} else {
						throw new Error("Error scraping Reddit profile") // Throw error if response status is not 200
					}
				} else if (selectedApp === "Twitter") {
					// If selected app is Twitter
					response = await window.electron?.invoke("scrape-twitter", {
						// Invoke electron to scrape Twitter profile
						twitterProfileUrl // Pass Twitter profile URL to backend
					})

					if (response.status === 200) {
						// If response status is 200 (success)
						await window.electron?.invoke("set-db-data", {
							// Invoke electron to set data in database
							data: { twitterProfile: response.topics } // Store scraped Twitter topics data in database
						})
						successMessage =
							"Twitter profile connected successfully." // Set success message for Twitter connection
					} else {
						throw new Error("Error scraping Twitter profile") // Throw error if response status is not 200
					}
				}

				await window.electron?.invoke("create-document-and-graph") // Invoke electron to create document and graph in backend
			} else if (action === "disconnect") {
				// If action is "disconnect"
				await window.electron?.invoke("set-db-data", {
					// Invoke electron to set data in database
					data: { [`${selectedApp.toLowerCase()}Profile`]: {} } // Clear profile data in database for selected app
				})

				await window.electron?.invoke("delete-subgraph", {
					// Invoke electron to delete subgraph
					source_name: selectedApp.toLowerCase() // Pass source name to backend to delete subgraph
				})

				successMessage = `${selectedApp} profile disconnected successfully.` // Set success message for profile disconnection
			}

			toast.success(successMessage) // Show success toast with success message

			setIsProfileConnected((prev) => ({
				// Update isProfileConnected state
				...prev,
				[selectedApp]: action === "connect" // Set connection status for selected app based on action
			}))
		} catch (error) {
			toast.error(`Error processing ${selectedApp} profile.`) // Show error toast if any error occurs
		} finally {
			setLoading((prev) => ({ ...prev, [selectedApp]: false })) // Set loading state for the selected app to false regardless of success or failure
		}
	}

	/**
	 * Handles the decline action from the disclaimer component.
	 * Simply hides the disclaimer modal by setting `showDisclaimer` state to false.
	 *
	 * @function handleDisclaimerDecline
	 * @returns {void}
	 */
	const handleDisclaimerDecline = () => {
		setShowDisclaimer(false) // Hide the disclaimer modal
	}

	/**
	 * Main return statement for the Settings component, rendering the settings UI.
	 *
	 * Includes sidebar, settings page header, app cards for LinkedIn, Reddit, and Twitter with connect/disconnect actions,
	 * referral dialog, beta program dialog, and disclaimer component.
	 *
	 * @returns {React.ReactNode} - The main UI for the Settings component.
	 */
	return (
		<div className="flex h-screen w-screen bg-matteblack text-white">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
				fromChat={false} // Indicates sidebar is not in chat view
			/>
			<div className="w-4/5 flex ml-5 flex-col pb-9 justify-center items-start h-full bg-matteblack">
				<div className="w-4/5 flex justify-between px-4 py-4">
					<h1 className="font-Poppins text-white text-6xl py-4">
						Settings
					</h1>
					<div className="flex gap-3">
						<ShiningButton
							text
							bgColor="bg-lightblue"
							borderColor="border-lightblue"
							className="rounded-lg cursor-pointer"
							borderClassName=""
							dataTooltipId="upgrade-tooltip"
							dataTooltipContent="Logout and login after upgrading to enjoy Pro features."
							icon={IconRocket}
							onClick={() =>
								window.open(
									"https://existence-sentient.vercel.app/dashboard",
									"_blank"
								)
							}
						>
							{pricing === "free"
								? "Upgrade to Pro"
								: "Current Plan: Pro"}
						</ShiningButton>
						<ShiningButton
							text
							bgColor="bg-lightblue"
							borderColor="border-lightblue"
							className="rounded-lg cursor-pointer"
							dataTooltipId="refer-tooltip"
							dataTooltipContent="Logout and login again after your friend has entered the code to reset your referrer status. Credits will be refreshed tomorrow"
							icon={IconGift}
							onClick={() => setShowReferralDialog(true)}
						>
							Refer Sentient
						</ShiningButton>
						<ShiningButton
							text
							bgColor="bg-lightblue"
							borderColor="border-lightblue"
							className="rounded-lg cursor-pointer"
							dataTooltipId="beta-tooltip"
							dataTooltipContent="Logout and login again after this step."
							icon={IconBeta}
							onClick={() => setShowBetaDialog(true)}
						>
							{betaUser
								? "Exit Beta Program"
								: "Become a Beta User"}
						</ShiningButton>
					</div>
				</div>
				<div className="flex items-center justify-center h-full w-4/5 space-x-8">
					{/* AppCard component for LinkedIn integration settings */}
					<AppCard
						logo="/images/linkedin-logo.png" // LinkedIn logo image path
						name="LinkedIn" // App name - LinkedIn
						description={
							// Description based on connection status
							isProfileConnected.LinkedIn
								? "Disconnect your LinkedIn profile and erase your professional info"
								: "Connect your LinkedIn account to pull in your professional profile and enhance your experience.."
						}
						onClick={
							// OnClick handler for LinkedIn card, toggles between connect and disconnect
							isProfileConnected.LinkedIn
								? () => handleDisconnectClick("LinkedIn")
								: () => handleConnectClick("LinkedIn")
						}
						action={
							isProfileConnected.LinkedIn
								? "disconnect"
								: "connect"
						} // Action text based on connection status
						loading={loading.LinkedIn} // Loading state for LinkedIn card
						disabled={Object.values(loading).some(
							(status) => status
						)} // Disable card if any app is loading
					/>
					{/* AppCard component for Reddit integration settings */}
					<AppCard
						logo="/images/reddit-logo.png" // Reddit logo image path
						name="Reddit" // App name - Reddit
						description={
							// Description based on connection status
							isProfileConnected.Reddit
								? "Disconnect your Reddit profile"
								: "Connect your Reddit account to analyze your activity and identify topics of interest."
						}
						onClick={
							// OnClick handler for Reddit card, toggles between connect and disconnect
							isProfileConnected.Reddit
								? () => handleDisconnectClick("Reddit")
								: () => handleConnectClick("Reddit")
						}
						action={
							// Action text based on connection status and pricing plan
							isProfileConnected.Reddit ? (
								"disconnect"
							) : pricing === "free" ? (
								<ProIcon /> // Show ProIcon for free plan users
							) : (
								"connect"
							)
						}
						loading={loading.Reddit} // Loading state for Reddit card
						disabled={
							pricing === "free" ||
							Object.values(loading).some((status) => status)
						} // Disable card for free plan users or if any app is loading
					/>
					{/* AppCard component for Twitter integration settings */}
					<AppCard
						logo="/images/twitter-logo.png" // Twitter logo image path
						name="Twitter" // App name - Twitter
						description={
							// Description based on connection status
							isProfileConnected.Twitter
								? "Disconnect your Twitter profile"
								: "Connect your Twitter account to analyze your tweets and identify topics of interest."
						}
						onClick={
							// OnClick handler for Twitter card, toggles between connect and disconnect
							isProfileConnected.Twitter
								? () => handleDisconnectClick("Twitter")
								: () => handleConnectClick("Twitter")
						}
						action={
							// Action text based on connection status and pricing plan
							isProfileConnected.Twitter ? (
								"disconnect"
							) : pricing === "free" ? (
								<ProIcon /> // Show ProIcon for free plan users
							) : (
								"connect"
							)
						}
						loading={loading.Twitter} // Loading state for Twitter card
						disabled={
							pricing === "free" ||
							Object.values(loading).some((status) => status)
						} // Disable card for free plan users or if any app is loading
					/>
				</div>
				{/* New Model Selection Section */}
				<div className="w-4/5 px-4 py-4">
					<h2 className="font-Poppins text-white text-2xl py-2">
						Model Settings
					</h2>
					<label htmlFor="model-select" className="text-white mr-2">
						Select AI Model:
					</label>
					<select
						id="model-select"
						value={selectedModel}
						onChange={async (e) => {
							const newModel = e.target.value
							try {
								await window.electron.invoke("set-db-data", {
									data: { selectedModel: newModel }
								})
								setSelectedModel(newModel)
								toast.success("Model updated successfully.")
							} catch (error) {
								toast.error("Error updating model.")
							}
						}}
						className="bg-gray-800 text-white p-2 rounded"
					>
						{availableModels.length > 0 ? (
							availableModels.map((model) => (
								<option key={model} value={model}>
									{model}
								</option>
							))
						) : (
							<option value="" disabled>
								No models available
							</option>
						)}
					</select>
					{availableModels.length === 0 && (
						<p className="text-red-500 mt-2">
							No models found. Please download models using Ollama
							and restart the app.
						</p>
					)}
				</div>	
			</div>
			{/* ModalDialog component for Referral Program info */}
			{showReferralDialog && (
				<ModalDialog
					title="Refer Sentient" // Title of the referral dialog
					description={
						// Description based on referrer status
						referrerStatus
							? "You are already a referrer. Enjoy your free 10 Pro credits daily!"
							: "Refer Sentient to one person, ask them to enter your referral code when they install, and get 10 daily free Pro credits."
					}
					onCancel={() => setShowReferralDialog(false)} // Handler to close referral dialog
					onConfirm={() => setShowReferralDialog(false)} // Handler for confirm action (just closes dialog)
					confirmButtonText="Got it" // Text for confirm button
					showInput={false} // No input field in this dialog
					extraContent={
						// Extra content to display referral code
						<div className="flex flex-col mt-4">
							<p>Your Referral Code:</p>
							<p className="text-lg font-bold">{referralCode}</p>
						</div>
					}
					confirmButtonIcon={IconGift} // Icon for confirm button - Gift icon
					confirmButtonColor="bg-lightblue" // Background color for confirm button
					confirmButtonBorderColor="border-lightblue" // Border color for confirm button
					cancelButton={false} // No cancel button in this dialog
				/>
			)}
			{/* ModalDialog component for Beta Program info and toggle */}
			{showBetaDialog && (
				<ModalDialog
					title={
						betaUser ? "Exit Beta Program" : "Become a Beta User"
					} // Title based on beta user status
					description={
						// Description based on beta user status
						betaUser
							? "Exiting the beta program means you will no longer get early access to new features. Are you sure you want to continue?"
							: "Joining the beta program allows you to test new features before they are released to the public. You may encounter bugs and unstable features. Are you sure you want to continue?"
					}
					onCancel={() => setShowBetaDialog(false)} // Handler to close beta dialog
					onConfirm={handleBetaUserToggle} // Handler for confirm action - toggles beta user status
					confirmButtonText={betaUser ? "Exit Beta" : "Join Beta"} // Confirm button text based on beta user status
					confirmButtonIcon={IconBeta} // Icon for confirm button - Beta icon
					confirmButtonColor="bg-lightblue" // Background color for confirm button
					confirmButtonBorderColor="border-lightblue" // Border color for confirm button
				/>
			)}
			{/* Disclaimer component for displaying disclaimers before connect/disconnect actions */}
			{showDisclaimer && (
				<Disclaimer
					appName={selectedApp} // App name for disclaimer context
					profileUrl={
						// Profile URL prop based on selected app
						selectedApp === "LinkedIn"
							? linkedInProfileUrl
							: selectedApp === "Reddit"
								? redditProfileUrl
								: twitterProfileUrl
					}
					setProfileUrl={
						// Set profile URL function based on selected app
						selectedApp === "LinkedIn"
							? setLinkedInProfileUrl
							: selectedApp === "Reddit"
								? setRedditProfileUrl
								: setTwitterProfileUrl
					}
					onAccept={handleDisclaimerAccept} // Handler for disclaimer accept action
					onDecline={handleDisclaimerDecline} // Handler for disclaimer decline action
					action={action} // Action type for disclaimer - "connect" or "disconnect"
				/>
			)}
		</div>
	)
}

export default Settings
