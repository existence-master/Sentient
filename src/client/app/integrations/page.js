"use client"

import React from "react"
import { useState, useEffect } from "react" // Importing necessary React hooks
import { useRouter } from "next/navigation" // Hook for client-side navigation in Next.js
import Disclaimer from "@components/Disclaimer" // Component to display a disclaimer for data usage
import AppCard from "@components/AppCard" // Reusable component to display app integration cards
import AnimatedLogo from "@components/AnimatedLogo" // Component for displaying an animated logo
import { Tooltip } from "@node_modules/react-tooltip/dist/react-tooltip" // Component for creating tooltips
import ProIcon from "@components/ProIcon" // Component to indicate Pro features
import ShiningButton from "@components/ShiningButton" // Custom button component with a shining effect
import { IconQuestionMark } from "@node_modules/@tabler/icons-react/dist/esm/tabler-icons-react" // Icon for help/question mark
import { WavyBackground } from "@components/WavyBackground" // Component for a wavy background effect
import AnimatedBeam from "@components/AnimatedBeam" // Component for an animated beam effect during loading
import toast from "react-hot-toast" // Library for displaying toast notifications

/**
 * AppIntegration Component - Handles the app integration page where users can connect social media profiles.
 *
 * This component allows users to connect their LinkedIn, Reddit, and Twitter accounts to enhance their profile.
 * It includes features for scraping profile data, displaying disclaimers, and building a user profile based on connected apps.
 *
 * @returns {React.ReactNode} - The AppIntegration component UI.
 */
const AppIntegration = () => {
	const router = useRouter() // Hook to get the router object for navigation
	const [showDisclaimer, setShowDisclaimer] = useState(false) // State to control the visibility of the disclaimer modal - showDisclaimer: boolean
	const [linkedInProfileUrl, setLinkedInProfileUrl] = useState("") // State to store LinkedIn profile URL input - linkedInProfileUrl: string
	const [redditProfileUrl, setRedditProfileUrl] = useState("") // State to store Reddit profile URL input - redditProfileUrl: string
	const [twitterProfileUrl, setTwitterProfileUrl] = useState("") // State to store Twitter profile URL input - twitterProfileUrl: string
	const [isBuildingProfile, setIsBuildingProfile] = useState(false) // State to indicate if the profile building process is active - isBuildingProfile: boolean
	const [selectedApp, setSelectedApp] = useState("") // State to store the name of the app currently being connected - selectedApp: string ("LinkedIn" | "Reddit" | "Twitter" | "")
	const [isConnecting, setIsConnecting] = useState(false) // State to indicate if the connection process is active - isConnecting: boolean
	const [connectedApps, setConnectedApps] = useState({
		// State to track which apps are currently connected - connectedApps: { LinkedIn: boolean, Reddit: boolean, Twitter: boolean }
		LinkedIn: false,
		Reddit: false,
		Twitter: false
	})
	const [pricingPlan, setPricingPlan] = useState("free") // State to store the user's pricing plan, defaults to "free" - pricingPlan: string ("free" | "pro")

	/**
	 * useEffect hook to fetch the user's pricing plan on component mount.
	 * Calls the electron backend to retrieve the pricing plan and updates the state.
	 */
	useEffect(() => {
		/**
		 * Fetches the pricing plan from the backend using electron invoke.
		 * Sets the pricingPlan state with the fetched plan or defaults to "free" on error or if no plan is returned.
		 */
		const fetchPricingPlan = async () => {
			try {
				const pricing =
					await window.electron?.invoke("fetch-pricing-plan") // Invoke electron backend to get pricing plan
				setPricingPlan(pricing || "free") // Set state with fetched pricing or default to 'free'
			} catch (error) {
				toast.error(`Error fetching pricing plan: ${error}`) // Show error toast if fetching fails
			}
		}

		fetchPricingPlan() // Call fetchPricingPlan on component mount
	}, [])

	/**
	 * Handles setting or adding data to the database.
	 *
	 * Fetches existing data from the database and either sets new data for a key if it doesn't exist or is empty,
	 * or adds data under an existing key if it already contains data.
	 *
	 * @async
	 * @function handleSetOrAddData
	 * @param {string} key - The database key to set or add data to.
	 * @param {any} data - The data to be set or added.
	 * @throws {Error} - Throws an error if fetching existing user data fails.
	 * @returns {Promise<void>}
	 */
	const handleSetOrAddData = async (key, data) => {
		try {
			const response = await window.electron?.invoke("get-user-data") // Invoke electron to get data from database
			if (response.status !== 200) {
				throw new Error("Error fetching existing user data") // Throw error if response status is not 200
			}

			const existingData = response.data || {} // Get existing data from response, default to empty object if no data

			// Check if data for the key is empty or doesn't exist
			if (
				!existingData[key] ||
				Object.keys(existingData[key]).length === 0 ||
				existingData[key].length === 0
			) {
				await window.electron?.invoke("set-user-data", {
					// Invoke electron to set data in database
					data: { [key]: data } // Data to set for the given key
				})
			} else {
				await window.electron?.invoke("add-db-data", {
					// Invoke electron to add data to existing data in database
					data: { [key]: data } // Data to add for the given key
				})
			}
		} catch (error) {
			toast.error(`Error handling data for key ${key}: ${error}`) // Show error toast if any error occurs
		}
	}

	/**
	 * Handles the acceptance of the disclaimer and proceeds with profile scraping.
	 *
	 * This function is called when the user accepts the disclaimer in the Disclaimer component.
	 * It sets the `isConnecting` state to true, initiates profile scraping based on the `selectedApp`,
	 * and updates the `connectedApps` state upon successful scraping.
	 *
	 * @async
	 * @function handleDisclaimerAccept
	 * @returns {Promise<void>}
	 */
	const handleDisclaimerAccept = async () => {
		setShowDisclaimer(false) // Hide the disclaimer modal
		setIsConnecting(true) // Set connecting state to true to indicate connection process is starting

		try {
			// Handle LinkedIn profile scraping
			if (selectedApp === "LinkedIn" && linkedInProfileUrl) {
				let response = await window.electron?.invoke(
					"scrape-linkedin",
					{
						// Invoke electron to scrape LinkedIn profile
						linkedInProfileUrl // Pass LinkedIn profile URL to backend
					}
				)

				if (response.status === 200) {
					await handleSetOrAddData(
						"linkedInProfile",
						response.profile
					) // Store scraped LinkedIn profile data in database
					setConnectedApps((prev) => ({ ...prev, LinkedIn: true })) // Update connectedApps state to mark LinkedIn as connected
				}
				// Handle Reddit profile scraping
			} else if (selectedApp === "Reddit" && redditProfileUrl) {
				let response = await window.electron?.invoke("scrape-reddit", {
					// Invoke electron to scrape Reddit profile
					redditProfileUrl // Pass Reddit profile URL to backend
				})

				if (response.status === 200) {
					await handleSetOrAddData("redditProfile", response.topics) // Store scraped Reddit topics data in database
					setConnectedApps((prev) => ({ ...prev, Reddit: true })) // Update connectedApps state to mark Reddit as connected
				}
				// Handle Twitter profile scraping
			} else if (selectedApp === "Twitter" && twitterProfileUrl) {
				let response = await window.electron?.invoke("scrape-twitter", {
					// Invoke electron to scrape Twitter profile
					twitterProfileUrl // Pass Twitter profile URL to backend
				})

				if (response.status === 200) {
					await handleSetOrAddData("twitterProfile", response.topics) // Store scraped Twitter topics data in database
					setConnectedApps((prev) => ({ ...prev, Twitter: true })) // Update connectedApps state to mark Twitter as connected
				}
			}
		} catch (error) {
			toast.error(`Error processing ${selectedApp} profile: ${error}`) // Show error toast if any error occurs during scraping
		} finally {
			setIsConnecting(false) // Set connecting state to false regardless of success or failure
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
	 * Handles the action to build the user profile after app integrations.
	 *
	 * This function sets `isBuildingProfile` to true to display a loading animation,
	 * invokes the electron backend to create a document and graph based on user data,
	 * and then marks the first run as completed in the database.
	 * Finally, it redirects the user to the chat page.
	 *
	 * @async
	 * @function handleBuildProfile
	 * @returns {Promise<void>}
	 */
	const handleBuildProfile = async () => {
		setIsBuildingProfile(true) // Set building profile state to true to show loading animation

		try {
			const response = await window.electron?.invoke("build-personality") // Invoke electron to create document and graph in backend
			if (response.status !== 200) {
				throw new Error("Error creating graph") // Throw error if response status is not 200
			}

			await window.electron?.invoke("set-user-data", {
				// Invoke electron to set data in database
				data: { firstRunCompleted: true } // Mark first run as completed in database
			})
		} catch (error) {
			toast.error("Error building profile") // Show error toast if any error occurs during profile building
		} finally {
			router.push("/chat") // Redirect user to the chat page after profile building is complete or failed
		}
	}

	/**
	 * Conditional rendering for the profile building animation screen.
	 *
	 * If `isBuildingProfile` is true, this screen with animated logo and "Building profile" message is displayed.
	 *
	 * @returns {React.ReactNode | null} - Returns the animated beam component with loading animation if `isBuildingProfile` is true, otherwise null.
	 */
	if (isBuildingProfile) {
		return (
			<AnimatedBeam>
				{" "}
				{/* AnimatedBeam component for loading effect */}
				<div className="flex flex-col items-center justify-center min-h-screen">
					<div className="flex flex-col items-center justify-center h-full backdrop-blur-xs">
						<AnimatedLogo /> {/* Animated logo component */}
						<div className="flex items-center justify-center gap-3">
							<h1 className="text-white font-Poppins text-4xl mb-6 mt-8">
								Building profile
							</h1>{" "}
							{/* Text indicating profile building */}
							<div className="flex space-x-1 mt-4">
								<div className="dot dot1" />{" "}
								{/* Animated dots for loading indicator */}
								<div className="dot dot2" />
								<div className="dot dot3" />
							</div>
						</div>
					</div>
				</div>
			</AnimatedBeam>
		)
	}

	/**
	 * Main return statement for the AppIntegration component, rendering the app integration UI.
	 *
	 * Includes app cards for LinkedIn, Reddit, and Twitter, a "Build Profile" button,
	 * a tooltip for integrations info, and conditional rendering for the Disclaimer component.
	 *
	 * @returns {React.ReactNode} - The main UI for the AppIntegration component.
	 */
	return (
		<>
			<WavyBackground className="max-w-screen overflow-hidden">
				{" "}
				{/* Wavy background for the entire page */}
				<div className="flex flex-col items-center justify-between py-20 min-h-screen bg-transparent relative">
					<h1 className="text-white text-5xl font-Poppins">
						Connect Your Apps
					</h1>{" "}
					{/* Title of the page */}
					<div className="flex items-center justify-center h-full w-1/2 space-x-8">
						{" "}
						{/* Container for app cards */}
						<AppCard
							logo="/images/linkedin-logo.png" // LinkedIn logo image path
							name="LinkedIn" // App name - LinkedIn
							description="Connect your LinkedIn account to add your professional information to Sentient's context." // Description for LinkedIn card
							onClick={() => {
								// OnClick handler for LinkedIn card
								if (!isConnecting && !connectedApps.LinkedIn) {
									// Check if not already connecting and LinkedIn not already connected
									setShowDisclaimer(true) // Show disclaimer modal
									setSelectedApp("LinkedIn") // Set selected app to LinkedIn
								}
							}}
							action={
								connectedApps.LinkedIn ? "Connected" : "connect"
							} // Action text based on connection status
							loading={selectedApp === "LinkedIn" && isConnecting} // Show loading state when connecting LinkedIn
							disabled={connectedApps.LinkedIn || isConnecting} // Disable card if already connected or currently connecting
						/>
						<AppCard
							logo="/images/reddit-logo.png" // Reddit logo image path
							name="Reddit" // App name - Reddit
							description="Connect your Reddit account to let Sentient analyze your subreddit activity and identify topics of interest." // Description for Reddit card
							onClick={() => {
								// OnClick handler for Reddit card
								if (
									!isConnecting &&
									!connectedApps.Reddit &&
									pricingPlan !== "free"
								) {
									// Check if not connecting, not connected, and not free plan
									setShowDisclaimer(true) // Show disclaimer modal
									setSelectedApp("Reddit") // Set selected app to Reddit
								}
							}}
							action={
								connectedApps.Reddit ? (
									"Connected"
								) : pricingPlan === "free" ? (
									<ProIcon />
								) : (
									"connect"
								)
							} // Action text, shows ProIcon for free plan
							loading={selectedApp === "Reddit" && isConnecting} // Show loading state when connecting Reddit
							disabled={
								connectedApps.Reddit ||
								isConnecting ||
								pricingPlan === "free"
							} // Disable if connected, connecting, or free plan
						/>
						<AppCard
							logo="/images/twitter-logo.png" // Twitter logo image path
							name="Twitter" // App name - Twitter
							description="Connect your Twitter account to let Sentient analyze your tweets and identify topics of interest." // Description for Twitter card
							onClick={() => {
								// OnClick handler for Twitter card
								if (
									!isConnecting &&
									!connectedApps.Twitter &&
									pricingPlan !== "free"
								) {
									// Check if not connecting, not connected, and not free plan
									setShowDisclaimer(true) // Show disclaimer modal
									setSelectedApp("Twitter") // Set selected app to Twitter
								}
							}}
							action={
								connectedApps.Twitter ? (
									"Connected"
								) : pricingPlan === "free" ? (
									<ProIcon />
								) : (
									"connect"
								)
							} // Action text, shows ProIcon for free plan
							loading={selectedApp === "Twitter" && isConnecting} // Loading state when connecting Twitter
							disabled={
								connectedApps.Twitter ||
								isConnecting ||
								pricingPlan === "free"
							} // Disable if connected, connecting, or free plan
						/>
					</div>
					<ShiningButton // Shining button component for "Build Profile" action
						className="font-Poppins font-semibold" // Styling classes
						onClick={handleBuildProfile} // OnClick handler to build profile
						disabled={isConnecting} // Disable button while connecting
					>
						Build Profile
					</ShiningButton>
					<div
						data-tooltip-id="integrations" // Tooltip ID for integrations info
						data-tooltip-content="You can always connect or disconnect the apps from the Settings page later." // Tooltip content
						className="absolute top-4 right-4" // Positioning classes
					>
						<button className="text-gray-300 hover-button p-2 rounded-[50%] text-sm cursor-default">
							<IconQuestionMark />
						</button>{" "}
						{/* Question mark icon button for tooltip */}
					</div>
					<Tooltip
						id="integrations"
						place="right"
						type="dark"
						effect="float"
					/>{" "}
					{/* Tooltip component for integrations info */}
				</div>
				{showDisclaimer && ( // Conditional rendering for Disclaimer component
					<Disclaimer
						appName={selectedApp} // App name to display in disclaimer
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
						action="connect" // Action type for disclaimer - "connect"
					/>
				)}
			</WavyBackground>
		</>
	)
}

export default AppIntegration
