"use client"

import { useState, useEffect } from "react"
import Disclaimer from "@components/Disclaimer"
import AppCard from "@components/AppCard"
import Sidebar from "@components/Sidebar"
import ProIcon from "@components/ProIcon"
import toast from "react-hot-toast"
import ShiningButton from "@components/ShiningButton"
import ModalDialog from "@components/ModalDialog"
import { IconGift, IconBeta, IconRocket } from "@tabler/icons-react"
import React from "react"
import { Switch } from "@radix-ui/react-switch" // Make sure this path is correct

const Settings = () => {
	const [showDisclaimer, setShowDisclaimer] = useState(false)
	const [linkedInProfileUrl, setLinkedInProfileUrl] = useState("")
	const [redditProfileUrl, setRedditProfileUrl] = useState("")
	const [twitterProfileUrl, setTwitterProfileUrl] = useState("")
	const [isProfileConnected, setIsProfileConnected] = useState({
		LinkedIn: false,
		Reddit: false,
		Twitter: false
	})
	const [action, setAction] = useState("")
	const [selectedApp, setSelectedApp] = useState("")
	const [loading, setLoading] = useState({
		LinkedIn: false,
		Reddit: false,
		Twitter: false
	})
	const [userDetails, setUserDetails] = useState({})
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	const [pricing, setPricing] = useState("free")
	const [showReferralDialog, setShowReferralDialog] = useState(false)
	const [referralCode, setReferralCode] = useState("")
	const [referrerStatus, setReferrerStatus] = useState(false)
	const [betaUser, setBetaUser] = useState(false)
	const [showBetaDialog, setShowBetaDialog] = useState(false)
	const [dataSources, setDataSources] = useState([])

	useEffect(() => {
		fetchData()
		fetchUserDetails()
		fetchPricingPlan()
		fetchReferralDetails()
		fetchBetaUserStatus()
		fetchDataSources()
	}, [])

	const fetchDataSources = async () => {
		try {
			const response = await window.electron.invoke("get-data-sources")
			if (response.error) {
				console.error("Error fetching data sources:", response.error)
				toast.error("Error fetching data sources.")
			} else {
				// Ensure data_sources is an array
				setDataSources(
					Array.isArray(response.data_sources)
						? response.data_sources
						: []
				)
			}
		} catch (error) {
			console.error("Error fetching data sources:", error)
			toast.error("Error fetching data sources.")
		}
	}

	const handleToggle = async (source, enabled) => {
		// Add loading state for toggles if needed
		try {
			const response = await window.electron.invoke(
				"set-data-source-enabled",
				source,
				enabled
			)
			if (response.error) {
				console.error(
					`Error updating ${source} data source:`,
					response.error
				)
				toast.error(`Error updating ${source} data source.`)
			} else {
				toast.success(
					`${source} data source ${enabled ? "enabled" : "disabled"}. Restart may be needed.`
				)
				setDataSources((prev) =>
					prev.map((ds) =>
						ds.name === source ? { ...ds, enabled } : ds
					)
				)
			}
		} catch (error) {
			console.error(`Error updating ${source} data source:`, error)
			toast.error(`Error updating ${source} data source.`)
		}
	}

	// ... (keep all other functions: fetchUserDetails, fetchPricingPlan, etc.) ...
	const fetchUserDetails = async () => {
		try {
			const response = await window.electron?.invoke("get-profile")
			setUserDetails(response)
		} catch (error) {
			console.error("Error fetching user details:", error)
			toast.error("Error fetching user details.")
		}
	}

	const fetchPricingPlan = async () => {
		try {
			const response = await window.electron?.invoke("fetch-pricing-plan")
			setPricing(response || "free")
		} catch (error) {
			console.error("Error fetching pricing plan:", error)
			toast.error("Error fetching pricing plan.")
		}
	}

	const fetchBetaUserStatus = async () => {
		try {
			const response = await window.electron?.invoke(
				"get-beta-user-status"
			)
			setBetaUser(response === true)
		} catch (error) {
			console.error("Error fetching beta user status:", error)
			toast.error("Error fetching beta user status.")
		}
	}

	const fetchReferralDetails = async () => {
		try {
			const referral = await window.electron?.invoke(
				"fetch-referral-code"
			)
			const referrer = await window.electron?.invoke(
				"fetch-referrer-status"
			)
			setReferralCode(referral || "N/A")
			setReferrerStatus(referrer === true)
		} catch (error) {
			console.error("Error fetching referral details:", error)
			toast.error("Error fetching referral details.")
		}
	}

	const handleBetaUserToggle = async () => {
		try {
			await window.electron?.invoke("invert-beta-user-status")
			setBetaUser((prev) => !prev)
			toast.success(
				betaUser
					? "You have exited the Beta User Program."
					: "You are now a Beta User!"
			)
		} catch (error) {
			console.error("Error updating beta user status:", error)
			toast.error("Error updating beta user status.")
		}
		setShowBetaDialog(false)
	}

	const fetchData = async () => {
		try {
			const response = await window.electron?.invoke("get-user-data")
			if (response.status === 200 && response.data) {
				const { linkedInProfile, redditProfile, twitterProfile } =
					response.data
				setIsProfileConnected({
					LinkedIn:
						!!linkedInProfile && // Use !! for clearer boolean check
						Object.keys(linkedInProfile).length > 0,
					Reddit:
						!!redditProfile &&
						Object.keys(redditProfile).length > 0,
					Twitter:
						!!twitterProfile &&
						Object.keys(twitterProfile).length > 0
				})
			} else if (response.status !== 200) {
				console.error(
					"Error fetching DB data, status:",
					response.status,
					"response:",
					response
				)
				toast.error("Error fetching user data (status not 200).")
			}
		} catch (error) {
			console.error("Error fetching user data:", error)
			toast.error("Error fetching user data.")
		}
	}

	const handleConnectClick = (appName) => {
		if (
			pricing === "free" &&
			(appName === "Reddit" || appName === "Twitter")
		) {
			toast.error("This feature is only available for Pro users.")
			return
		}
		setShowDisclaimer(true)
		setSelectedApp(appName)
		setAction("connect")
	}

	const handleDisconnectClick = (appName) => {
		setShowDisclaimer(true)
		setSelectedApp(appName)
		setAction("disconnect")
	}

	const handleDisclaimerAccept = async () => {
		setShowDisclaimer(false)
		setLoading((prev) => ({ ...prev, [selectedApp]: true }))
		try {
			let successMessage = ""
			let response = null // Define response outside if/else
			if (action === "connect") {
				const profileKey = `${selectedApp.toLowerCase()}Profile`
				const urlKey = `${selectedApp.toLowerCase()}ProfileUrl`
				const profileUrl =
					selectedApp === "LinkedIn"
						? linkedInProfileUrl
						: selectedApp === "Reddit"
							? redditProfileUrl
							: twitterProfileUrl
				const scrapeMethod = `scrape-${selectedApp.toLowerCase()}`

				response = await window.electron?.invoke(scrapeMethod, {
					[urlKey]: profileUrl
				})

				if (response && response.status === 200) {
					const dataToSet =
						selectedApp === "LinkedIn"
							? response.profile
							: response.topics
					await window.electron?.invoke("set-user-data", {
						data: { [profileKey]: dataToSet }
					})
					successMessage = `${selectedApp} profile connected successfully.`
					await window.electron?.invoke("build-personality") // Call this after successful connect
				} else {
					console.error(
						`Error scraping ${selectedApp} profile:`,
						response
					)
					throw new Error(
						`Error scraping ${selectedApp} profile. Status: ${response?.status}`
					)
				}
			} else if (action === "disconnect") {
				const profileKey = `${selectedApp.toLowerCase()}Profile`
				await window.electron?.invoke("set-user-data", {
					data: { [profileKey]: {} }
				})
				await window.electron?.invoke("delete-subgraph", {
					source_name: selectedApp.toLowerCase()
				})
				successMessage = `${selectedApp} profile disconnected successfully.`
			}
			toast.success(successMessage)
			// Update state *after* successful operation
			setIsProfileConnected((prev) => ({
				...prev,
				[selectedApp]: action === "connect"
			}))
			// Reset URLs after connect/disconnect might be useful
			// setLinkedInProfileUrl(""); setRedditProfileUrl(""); setTwitterProfileUrl("");
		} catch (error) {
			console.error(`Error processing ${selectedApp} profile:`, error)
			toast.error(
				`Error processing ${selectedApp} profile. ${error.message || ""}`
			)
		} finally {
			setLoading((prev) => ({ ...prev, [selectedApp]: false }))
			// Reset action/selectedApp after operation
			setAction("")
			setSelectedApp("")
		}
	}

	const handleDisclaimerDecline = () => {
		setShowDisclaimer(false)
		setAction("")
		setSelectedApp("")
	}

	return (
		<div className="flex h-screen w-screen bg-matteblack text-white overflow-hidden">
			{" "}
			{/* Added overflow-hidden */}
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
				fromChat={false}
			/>
			{/* MAIN CONTENT AREA - Added overflow-y-auto */}
			<div className="flex-1 flex flex-col pb-9 items-start bg-matteblack overflow-y-auto px-8">
				{" "}
				{/* Adjusted width using flex-1, added padding, added overflow */}
				<div className="w-full flex justify-between items-center py-4 mt-4">
					{" "}
					{/* Use w-full, added items-center */}
					<h1 className="font-Poppins text-white text-4xl md:text-5xl lg:text-6xl py-4">
						{" "}
						{/* Responsive text size */}
						Settings
					</h1>
					<div className="flex flex-wrap gap-3">
						{" "}
						{/* Added flex-wrap */}
						{/* ... Buttons ... */}
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
				{/* Connections Section */}
				<h2 className="text-2xl font-Poppins mb-4 mt-6 text-gray-300">
					Connections
				</h2>
				<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 w-full">
					{" "}
					{/* Responsive Grid layout */}
					{/* AppCard component for LinkedIn integration settings */}
					<AppCard
						logo="/images/linkedin-logo.png"
						name="LinkedIn"
						description={
							isProfileConnected.LinkedIn
								? "Disconnect your LinkedIn profile and erase your professional info"
								: "Connect your LinkedIn account to pull in your professional profile and enhance your experience."
						}
						onClick={
							isProfileConnected.LinkedIn
								? () => handleDisconnectClick("LinkedIn")
								: () => handleConnectClick("LinkedIn")
						}
						action={
							isProfileConnected.LinkedIn
								? "disconnect"
								: "connect"
						}
						loading={loading.LinkedIn}
						disabled={Object.values(loading).some(
							(status) => status
						)}
					/>
					{/* ... Reddit and Twitter AppCards ... */}
					<AppCard
						logo="/images/reddit-logo.png"
						name="Reddit"
						description={
							isProfileConnected.Reddit
								? "Disconnect your Reddit profile"
								: "Connect your Reddit account to analyze your activity and identify topics of interest."
						}
						onClick={
							isProfileConnected.Reddit
								? () => handleDisconnectClick("Reddit")
								: () => handleConnectClick("Reddit")
						}
						action={
							isProfileConnected.Reddit ? (
								"disconnect"
							) : pricing === "free" ? (
								<ProIcon />
							) : (
								"connect"
							)
						}
						loading={loading.Reddit}
						disabled={
							pricing === "free" ||
							Object.values(loading).some((status) => status)
						}
					/>
					<AppCard
						logo="/images/twitter-logo.png"
						name="Twitter"
						description={
							isProfileConnected.Twitter
								? "Disconnect your Twitter profile"
								: "Connect your Twitter account to analyze your tweets and identify topics of interest."
						}
						onClick={
							isProfileConnected.Twitter
								? () => handleDisconnectClick("Twitter")
								: () => handleConnectClick("Twitter")
						}
						action={
							isProfileConnected.Twitter ? (
								"disconnect"
							) : pricing === "free" ? (
								<ProIcon />
							) : (
								"connect"
							)
						}
						loading={loading.Twitter}
						disabled={
							pricing === "free" ||
							Object.values(loading).some((status) => status)
						}
					/>
				</div>
				{/* --- DATA SOURCES SECTION - MOVED HERE --- */}
				<h2 className="text-2xl font-Poppins mb-4 mt-8 text-gray-300">
					Data Sources
				</h2>
				<div className="w-full bg-gray-800 p-4 md:p-6 rounded-lg shadow-md">
					{" "}
					{/* Added background, padding, rounded corners */}
					<div className="space-y-4">
						{dataSources.length > 0 ? (
							dataSources.map((source) => (
								<div
									key={source.name}
									className="flex items-center justify-between text-white py-2 border-b border-gray-700 last:border-b-0"
								>
									<span className="font-medium">
										{source.name}
									</span>
									{/* Radix Switch with basic styling */}
									<Switch
										checked={source.enabled}
										onCheckedChange={(enabled) =>
											handleToggle(source.name, enabled)
										}
										className="group relative inline-flex h-5 w-10 flex-shrink-0 cursor-pointer items-center justify-center rounded-full focus:outline-none focus:ring-2 focus:ring-lightblue focus:ring-offset-2 focus:ring-offset-gray-800"
									>
										<span className="sr-only">
											Use setting
										</span>
										{/* Background */}
										<span
											aria-hidden="true"
											className={`${
												source.enabled
													? "bg-lightblue"
													: "bg-gray-600"
											} pointer-events-none absolute h-full w-full rounded-md`}
										/>
										{/* Thumb */}
										<span
											aria-hidden="true"
											className={`${
												source.enabled
													? "translate-x-5"
													: "translate-x-0"
											} pointer-events-none absolute left-0 inline-block h-5 w-5 transform rounded-full border border-gray-200 bg-white shadow ring-0 transition-transform duration-200 ease-in-out`}
										/>
									</Switch>
								</div>
							))
						) : (
							<p className="text-gray-400 italic">
								No data sources found or loading...
							</p>
						)}
					</div>
				</div>
				{/* --- END OF DATA SOURCES SECTION --- */}
			</div>{" "}
			{/* END OF MAIN CONTENT AREA */}
			{/* Modals remain outside the main flow */}
			{showReferralDialog && <ModalDialog /* ...props... */ />}
			{showBetaDialog && <ModalDialog /* ...props... */ />}
			{showDisclaimer && <Disclaimer /* ...props... */ />}
		</div>
	)
}

export default Settings
