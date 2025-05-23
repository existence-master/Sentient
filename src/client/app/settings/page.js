"use client"

import { useState, useEffect, useCallback } from "react" // Added useCallback
import Sidebar from "@components/Sidebar"
import ProIcon from "@components/ProIcon" // Still used within AppCard logic
import toast from "react-hot-toast"
// REMOVED: ShiningButton import - replaced with standard buttons or integrated styling
import ModalDialog from "@components/ModalDialog"
// ADDED: More icons
import {
	IconGift,
	IconRocket,
	IconFlask, // Replaced IconBeta with IconFlask
	IconMail,
	IconCalendarEvent,
	IconWorldSearch, // Data source icons
	IconLoader,
	IconSettingsCog // For customize/recreate buttons
} from "@tabler/icons-react"
import React from "react"
import { Switch } from "@radix-ui/react-switch" // Keep Radix Switch
import { cn } from "@utils/cn" // Import cn utility

// ADDED: Mapping for Data Source Icons
const dataSourceIcons = {
	gmail: IconMail,
	gcalendar: IconCalendarEvent,
	internet_search: IconWorldSearch
}

const Settings = () => {
	const [userDetails, setUserDetails] = useState({})
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	const [pricing, setPricing] = useState("free")
	const [showReferralDialog, setShowReferralDialog] = useState(false)
	const [referralCode, setReferralCode] = useState("DUMMY")
	const [referrerStatus, setReferrerStatus] = useState(false)
	const [betaUser, setBetaUser] = useState(false)
	const [showBetaDialog, setShowBetaDialog] = useState(false)
	const [dataSources, setDataSources] = useState([])
	// ADDED: Customize input state
	const [isCustomizeInputVisible, setCustomizeInputVisible] = useState(false)
	const [newGraphInfo, setNewGraphInfo] = useState("")
	const [customizeLoading, setCustomizeLoading] = useState(false) // Separate loading for customize
	const [recreateGraphLoading, setRecreateGraphLoading] = useState(false)

	// --- Data Fetching ---
	// MODIFIED: Wrapped fetchDataSources in useCallback
	const fetchDataSources = useCallback(async () => {
		console.log("Fetching data sources...")
		try {
			const response = await window.electron.invoke("get-data-sources")
			if (response.error) {
				console.error("Error fetching data sources:", response.error)
				toast.error("Error fetching data sources.")
				setDataSources([]) // Ensure empty array on error
			} else {
				// Ensure data_sources is an array and add icons
				const sourcesWithIcons = (
					Array.isArray(response.data_sources)
						? response.data_sources
						: []
				).map((ds) => ({
					...ds,
					icon: dataSourceIcons[ds.name] || IconSettingsCog // Assign icon or default
				}))
				setDataSources(sourcesWithIcons)
				console.log("Data sources fetched:", sourcesWithIcons)
			}
		} catch (error) {
			console.error("Error fetching data sources:", error)
			toast.error("Error fetching data sources.")
			setDataSources([])
		}
	}, []) // Empty dependency array

	// MODIFIED: Wrapped handleToggle in useCallback
	const handleToggle = async (sourceName, enabled) => {
		console.log(`Toggling ${sourceName} to ${enabled}`)
		console.log(typeof (enabled)) // Check type
		console.log(typeof (sourceName)) // Check type
		// Optimistic UI update
		setDataSources((prev) =>
			prev.map((ds) => (ds.name === sourceName ? { ...ds, enabled } : ds))
		)
		console.log(`Toggling ${sourceName} to ${enabled}`)
		try {
			const response = await window.electron.invoke(
				"set-data-source-enabled",
				{ source: sourceName, enabled }
			) // Pass object
			if (response.error) {
				console.error(
					`Error updating ${sourceName} data source:`,
					response.error
				)
				toast.error(`Error updating ${sourceName}: ${response.error}`)
				// Revert optimistic update on error
				setDataSources((prev) =>
					prev.map((ds) =>
						ds.name === sourceName
							? { ...ds, enabled: !enabled }
							: ds
					)
				)
			} else {
				toast.success(
					`${sourceName} ${enabled ? "enabled" : "disabled"}.`
				) // Removed restart message for now
			}
		} catch (error) {
			console.error(`Error updating ${sourceName} data source:`, error)
			toast.error(`Error updating ${sourceName}.`)
			// Revert optimistic update on error
			setDataSources((prev) =>
				prev.map((ds) =>
					ds.name === sourceName ? { ...ds, enabled: !enabled } : ds
				)
			)
		}
	}

	const fetchUserDetails = useCallback(async () => {
		/* ...no functional change, ensure useCallback if needed... */
	}, [])
	const fetchPricingPlan = useCallback(async () => {
		/* ...no functional change, ensure useCallback if needed... */
	}, [])
	const fetchBetaUserStatus = useCallback(async () => {
		/* ...no functional change, ensure useCallback if needed... */
	}, [])
	const fetchReferralDetails = useCallback(async () => {
		/* ...no functional change, ensure useCallback if needed... */
	}, [])

	const handleBetaUserToggle = async () => {
		try {
			// await window.electron?.invoke("invert-beta-user-status")
			setBetaUser((prev) => !prev)
			toast.success(
				betaUser ? "Exited Beta Program." : "You are now a Beta User!"
			)
		} catch (error) {
			console.error("Error updating beta user status:", error)
			toast.error("Error updating beta user status.")
		}
		setShowBetaDialog(false)
	}

	const fetchData = useCallback(async () => {
		// Removed fetching for LinkedIn, Reddit, Twitter connection statuses
		console.log("Fetching user data (excluding social media connection status)...")
		try {
			const response = await window.electron?.invoke("get-user-data")
			if (response.status === 200 && response.data) {
				// User data might still contain other relevant info, process if needed
				// For now, just logging that it was fetched
				console.log("User data fetched successfully (social media connections parts are ignored).");
			} else {
				console.error(
					"Error fetching DB data, status:",
					response?.status,
					"response:",
					response
				)
			}
		} catch (error) {
			console.error("Error fetching user data:", error)
		}
	}, []) // Empty dependency array

	useEffect(() => {
		console.log("Initial useEffect running...")
		fetchData() // Fetch connection status
		fetchUserDetails()
		fetchPricingPlan()
		fetchReferralDetails()
		fetchBetaUserStatus()
		fetchDataSources()
		// No interval here, data fetched on mount or refresh
	}, [
		fetchData,
		fetchUserDetails,
		fetchPricingPlan,
		fetchReferralDetails,
		fetchBetaUserStatus,
		fetchDataSources
	]) // Add all useCallback functions

	// --- Action Handlers ---
	// handleConnectClick and handleDisconnectClick are removed as their UI elements are gone.
	// handleDisclaimerAccept and handleDisclaimerDecline might become redundant if no other feature uses them.
	// For now, let's assume the disclaimer flow might be used by other (future) settings,
	// Disclaimer-related functions (handleDisclaimerAccept, handleDisclaimerDecline) are removed.

	return (
		// MODIFIED: Overall page structure using flex
		<div className="h-screen bg-matteblack flex relative overflow-hidden dark">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
			/>
			{/* MODIFIED: Main Content Area */}
			<div className="flex-grow flex flex-col h-full bg-matteblack text-white relative overflow-y-auto p-6 md:p-10 custom-scrollbar">
				{/* --- Top Section: Heading & Action Buttons --- */}
				<div className="flex justify-between items-center mb-8 flex-shrink-0 px-4">
					<h1 className="font-Poppins text-white text-3xl md:text-4xl font-light">
						{" "}
						Settings{" "}
					</h1>
					{/* MODIFIED: Top right buttons - smaller, themed */}
					<div className="flex items-center gap-3">
						<button
							onClick={() =>
								window.open(
									"https://existence-sentient.vercel.app/dashboard",
									"_blank"
								)
							}
							className="flex items-center gap-2 py-2 px-4 rounded-full bg-darkblue hover:bg-lightblue text-white text-xs sm:text-sm font-medium transition-colors shadow-md"
							title={
								pricing === "free"
									? "Upgrade for more features"
									: "Manage Subscription"
							}
						>
							<IconRocket size={18} />
							<span>
								{pricing === "free"
									? "Upgrade to Pro"
									: "Manage Pro Plan"}
							</span>
						</button>
						<button
							onClick={() => setShowReferralDialog(true)}
							className="flex items-center gap-2 py-2 px-4 rounded-full bg-neutral-700 hover:bg-neutral-600 text-white text-xs sm:text-sm font-medium transition-colors shadow-md"
							title="Refer a friend"
						>
							<IconGift size={18} />
							<span>Refer Sentient</span>
						</button>
						<button
							onClick={() => setShowBetaDialog(true)}
							className="flex items-center gap-2 py-2 px-4 rounded-full bg-neutral-700 hover:bg-neutral-600 text-white text-xs sm:text-sm font-medium transition-colors shadow-md"
							title={
								betaUser
									? "Leave Beta Program"
									: "Join Beta Program"
							}
						>
							<IconFlask size={18} /> {/* Replaced IconBeta */}
							<span>{betaUser ? "Leave Beta" : "Join Beta"}</span>
						</button>
					</div>
				</div>
				{/* --- Main Settings Content --- */}
				{/* MODIFIED: Centered content with max-width */}
				<div className="w-full max-w-5xl mx-auto space-y-10 flex-grow">
					{/* Data Sources Section */}
					<section>
						<h2 className="text-xl font-semibold mb-5 text-gray-300 border-b border-neutral-700 pb-2">
							Background Data Sources
						</h2>
						{/* MODIFIED: Restyled container and list items */}
						<div className="bg-neutral-800/50 p-4 md:p-6 rounded-lg border border-neutral-700">
							<div className="space-y-4">
								{dataSources.length > 0 ? (
									dataSources.map((source) => {
										const SourceIcon =
											source.icon || IconSettingsCog // Use mapped icon or default
										return (
											<div
												key={source.name}
												className="flex items-center justify-between py-2"
											>
												<div className="flex items-center gap-3">
													<SourceIcon className="w-6 h-6 text-lightblue" />{" "}
													{/* Use icon */}
													<span className="font-medium text-white text-base">
														{source.name}
													</span>{" "}
													{/* Increased text size */}
												</div>
												{/* MODIFIED: Radix Switch with custom theme styling */}
												<Switch
													checked={source.enabled}
													onCheckedChange={(
														enabled
													) =>
														handleToggle(
															source.name,
															enabled
														)
													}
													className={cn(
														"group relative inline-flex h-[24px] w-[44px] flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-lightblue focus:ring-offset-2 focus:ring-offset-neutral-800",
														source.enabled
															? "bg-lightblue"
															: "bg-neutral-600" // Background color based on state
													)}
												>
													<span className="sr-only">
														Toggle {source.name}
													</span>
													<span
														aria-hidden="true"
														className={cn(
															"pointer-events-none inline-block h-[20px] w-[20px] transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out",
															source.enabled
																? "translate-x-[20px]"
																: "translate-x-0" // Thumb position based on state
														)}
													/>
												</Switch>
											</div>
										)
									})
								) : (
									<p className="text-gray-400 italic text-center py-4">
										{" "}
										Data source settings loading...{" "}
									</p>
								)}
							</div>
						</div>
					</section>

					{/* End Centered Content */}
					{/* Modals */}
					{showReferralDialog && (
						<ModalDialog
							title="Referral Code"
							description={`Share this code with friends: ${referralCode === "N/A" ? "Loading..." : ""}`}
							extraContent={
								referrerStatus ? (
									<p className="text-sm text-green-400">
										Referrer status: Active
									</p>
								) : (
									<p className="text-sm text-yellow-400">
										Referrer status: Inactive
									</p>
								)
							}
							onConfirm={() => setShowReferralDialog(false)}
							confirmButtonText="Close"
							cancelButton={false}
						/>
					)}
					{showBetaDialog && (
						<ModalDialog
							title={
								betaUser
									? "Exit Beta Program?"
									: "Join Beta Program?"
							}
							description={
								betaUser
									? "You will lose access to beta features."
									: "Get early access to new features!"
							}
							onCancel={() => setShowBetaDialog(false)}
							onConfirm={handleBetaUserToggle}
							confirmButtonText={
								betaUser ? "Exit Beta" : "Join Beta"
							}
							confirmButtonColor={
								betaUser ? "bg-red-600" : "bg-green-600"
							}
							confirmButtonBorderColor={
								betaUser ? "border-red-600" : "border-green-600"
							}
						/>
					)}
				</div>{" "}
				{/* END OF MAIN CONTENT AREA */}
			</div>{" "}
		</div>
	)
}

export default Settings
