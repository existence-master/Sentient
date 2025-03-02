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
import { IconTrash } from "@tabler/icons-react"

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
	const [availableModels, setAvailableModels] = useState([])
	const [selectedModel, setSelectedModel] = useState("llama3.2:3b")
	const [apiKeyDialog, setApiKeyDialog] = useState(false)
	const [apiKey, setApiKey] = useState("")
	const [selectedProvider, setSelectedProvider] = useState("")
	const [storedProviders, setStoredProviders] = useState({
		openai: false,
		gemini: false,
		claude: false
	})
	const [setModelOnSubmit, setSetModelOnSubmit] = useState(false)

	// Define cloud models to appear at the top of the dropdown
	const cloudModels = [
		{ name: "OpenAI", value: "openai" },
		{ name: "Gemini", value: "gemini" },
		{ name: "Claude", value: "claude" }
	]

	useEffect(() => {
		fetchData()
		fetchUserDetails()
		fetchPricingPlan()
		fetchReferralDetails()
		fetchBetaUserStatus()
		fetchAvailableModels()
		fetchStoredProviders() // Fetch stored providers on load
	}, [])

	const fetchUserDetails = async () => {
		try {
			const response = await window.electron?.invoke("get-profile")
			setUserDetails(response)
		} catch (error) {
			toast.error("Error fetching user details.")
		}
	}

	const fetchPricingPlan = async () => {
		try {
			const response = await window.electron?.invoke("fetch-pricing-plan")
			setPricing(response || "free")
		} catch (error) {
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
			toast.error("Error updating beta user status.")
		}
		setShowBetaDialog(false)
	}

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
				setSelectedModel(selectedModel || "llama3.2:3b")
			}
		} catch (error) {
			toast.error("Error fetching user data.")
		}
	}

	const fetchAvailableModels = async () => {
		try {
			const models = await window.electron.invoke("get-ollama-models")
			setAvailableModels(models)
			if (models.length > 0 && !models.includes(selectedModel)) {
				const newModel = models[0]
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

	const fetchStoredProviders = async () => {
		try {
			const response = await window.electron?.invoke(
				"get-stored-providers"
			)
			setStoredProviders(response)
		} catch (error) {
			toast.error("Error fetching stored API keys.")
		}
	}

	const handleDeleteKey = async (provider) => {
		try {
			await window.electron.invoke("delete-api-key", provider)
			setStoredProviders((prev) => ({ ...prev, [provider]: false }))
			toast.success(`API key for ${provider} deleted successfully.`)
		} catch (error) {
			toast.error(`Error deleting API key for ${provider}.`)
		}
	}

	// Handle model selection change
	const handleModelChange = async (model) => {
		if (cloudModels.some((m) => m.value === model)) {
			setSelectedProvider(model)
			const hasKey = await window.electron.invoke("check-api-key", model)
			if (!hasKey) {
				setSetModelOnSubmit(true) // Set model after saving key
				setApiKeyDialog(true)
			} else {
				setSelectedModel(model)
				await window.electron.invoke("set-db-data", {
					data: { selectedModel: model }
				})
				toast.success("Model updated successfully.")
			}
		} else {
			setSelectedModel(model)
			await window.electron.invoke("set-db-data", {
				data: { selectedModel: model }
			})
			toast.success("Model updated successfully.")
		}
	}

	// Handle API key submission
	const handleApiKeySubmit = async () => {
		try {
			await window.electron.invoke("set-api-key", {
				provider: selectedProvider,
				apiKey
			})
			if (setModelOnSubmit) {
				setSelectedModel(selectedProvider)
				await window.electron.invoke("set-db-data", {
					data: { selectedModel: selectedProvider }
				})
			}
			toast.success(
				setModelOnSubmit
					? "API key saved and model updated successfully."
					: "API key saved successfully."
			)
			setApiKeyDialog(false)
			setApiKey("")
			fetchStoredProviders() // Refresh stored providers
		} catch (error) {
			toast.error("Error saving API key.")
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
			if (action === "connect") {
				let response = null
				if (selectedApp === "LinkedIn") {
					response = await window.electron?.invoke(
						"scrape-linkedin",
						{ linkedInProfileUrl }
					)
					if (response.status === 200) {
						await window.electron?.invoke("set-db-data", {
							data: { linkedInProfile: response.profile }
						})
						successMessage =
							"LinkedIn profile connected successfully."
					} else {
						throw new Error("Error scraping LinkedIn profile")
					}
				} else if (selectedApp === "Reddit") {
					response = await window.electron?.invoke("scrape-reddit", {
						redditProfileUrl
					})
					if (response.status === 200) {
						await window.electron?.invoke("set-db-data", {
							data: { redditProfile: response.topics }
						})
						successMessage =
							"Reddit profile connected successfully."
					} else {
						throw new Error("Error scraping Reddit profile")
					}
				} else if (selectedApp === "Twitter") {
					response = await window.electron?.invoke("scrape-twitter", {
						twitterProfileUrl
					})
					if (response.status === 200) {
						await window.electron?.invoke("set-db-data", {
							data: { twitterProfile: response.topics }
						})
						successMessage =
							"Twitter profile connected successfully."
					} else {
						throw new Error("Error scraping Twitter profile")
					}
				}
				await window.electron?.invoke("create-document-and-graph")
			} else if (action === "disconnect") {
				await window.electron?.invoke("set-db-data", {
					data: { [`${selectedApp.toLowerCase()}Profile`]: {} }
				})
				await window.electron?.invoke("delete-subgraph", {
					source_name: selectedApp.toLowerCase()
				})
				successMessage = `${selectedApp} profile disconnected successfully.`
			}
			toast.success(successMessage)
			setIsProfileConnected((prev) => ({
				...prev,
				[selectedApp]: action === "connect"
			}))
		} catch (error) {
			toast.error(`Error processing ${selectedApp} profile.`)
		} finally {
			setLoading((prev) => ({ ...prev, [selectedApp]: false }))
		}
	}

	const handleDisclaimerDecline = () => {
		setShowDisclaimer(false)
	}

	const openAddKeyDialog = (provider) => {
		setSelectedProvider(provider)
		setApiKey("")
		setSetModelOnSubmit(false) // Do not set model after saving key
		setApiKeyDialog(true)
	}

	return (
		<div className="flex h-screen w-screen bg-matteblack text-white">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
				fromChat={false}
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
						}
						loading={loading.LinkedIn}
						disabled={Object.values(loading).some(
							(status) => status
						)}
					/>
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
						onChange={(e) => handleModelChange(e.target.value)}
						className="bg-matteblack text-white p-2 rounded focus:outline-none"
					>
						{cloudModels.map((model) => (
							<option key={model.value} value={model.value}>
								{model.name}
							</option>
						))}
						{availableModels.map((model) => (
							<option key={model} value={model}>
								{model}
							</option>
						))}
						{availableModels.length === 0 && (
							<option value="" disabled>
								No local models available
							</option>
						)}
					</select>
					{availableModels.length === 0 && (
						<p className="text-red-500 mt-2">
							No local models found. Please download models using
							Ollama and restart the app.
						</p>
					)}
				</div>
				<div className="w-4/5 px-4 py-4">
					<h2 className="font-Poppins text-white text-2xl py-2">
						Stored API Keys
					</h2>
					<div className="w-full min-h-fit overflow-x-auto rounded-xl border border-gray-400">
						<table className="min-w-full bg-matteblack text-white">
							<thead>
								<tr>
									<th className="py-2 px-4 border-b">
										Provider
									</th>
									<th className="py-2 px-4 border-b">
										Status
									</th>
									<th className="py-2 px-4 border-b">
										Action
									</th>
								</tr>
							</thead>
							<tbody>
								{cloudModels.map((model) => (
									<tr key={model.value}>
										<td className="py-2 px-4 border-b text-center">
											{model.name}
										</td>
										<td className="py-2 px-4 border-b text-center">
											{storedProviders[model.value] ? (
												<span className="text-green-500">
													Key stored
												</span>
											) : (
												<span className="text-red-500">
													No key stored
												</span>
											)}
										</td>
										<td className="py-2 px-4 border-b text-center">
											{storedProviders[model.value] ? (
												<button
													onClick={() =>
														handleDeleteKey(
															model.value
														)
													}
													className="bg-red-500 cursor-pointer text-white px-2 py-1 rounded hover:bg-red-600"
												>
													<IconTrash />
												</button>
											) : (
												<button
													onClick={() =>
														openAddKeyDialog(
															model.value
														)
													}
													className="bg-green-500 cursor-pointer text-white px-2 py-1 rounded hover:bg-green-600"
												>
													Add Key
												</button>
											)}
										</td>
									</tr>
								))}
							</tbody>
						</table>
					</div>
				</div>
			</div>
			{apiKeyDialog && (
				<ModalDialog
					title={`Enter API Key for ${selectedProvider}`}
					description="Please enter your API key to use this model."
					onCancel={() => setApiKeyDialog(false)}
					onConfirm={handleApiKeySubmit}
					confirmButtonText="Save"
					showInput={true}
					inputValue={apiKey}
					onInputChange={setApiKey}
				/>
			)}
			{showReferralDialog && (
				<ModalDialog
					title="Refer Sentient"
					description={
						referrerStatus
							? "You are already a referrer. Enjoy your free 10 Pro credits daily!"
							: "Refer Sentient to one person, ask them to enter your referral code when they install, and get 10 daily free Pro credits."
					}
					onCancel={() => setShowReferralDialog(false)}
					onConfirm={() => setShowReferralDialog(false)}
					confirmButtonText="Got it"
					showInput={false}
					extraContent={
						<div className="flex flex-col mt-4">
							<p>Your Referral Code:</p>
							<p className="text-lg font-bold">{referralCode}</p>
						</div>
					}
					confirmButtonIcon={IconGift}
					confirmButtonColor="bg-lightblue"
					confirmButtonBorderColor="border-lightblue"
					cancelButton={false}
				/>
			)}
			{showBetaDialog && (
				<ModalDialog
					title={
						betaUser ? "Exit Beta Program" : "Become a Beta User"
					}
					description={
						betaUser
							? "Exiting the beta program means you will no longer get early access to new features. Are you sure you want to continue?"
							: "Joining the beta program allows you to test new features before they are released to the public. You may encounter bugs and unstable features. Are you sure you want to continue?"
					}
					onCancel={() => setShowBetaDialog(false)}
					onConfirm={handleBetaUserToggle}
					confirmButtonText={betaUser ? "Exit Beta" : "Join Beta"}
					confirmButtonIcon={IconBeta}
					confirmButtonColor="bg-lightblue"
					confirmButtonBorderColor="border-lightblue"
				/>
			)}
			{showDisclaimer && (
				<Disclaimer
					appName={selectedApp}
					profileUrl={
						selectedApp === "LinkedIn"
							? linkedInProfileUrl
							: selectedApp === "Reddit"
								? redditProfileUrl
								: twitterProfileUrl
					}
					setProfileUrl={
						selectedApp === "LinkedIn"
							? setLinkedInProfileUrl
							: selectedApp === "Reddit"
								? setRedditProfileUrl
								: setTwitterProfileUrl
					}
					onAccept={handleDisclaimerAccept}
					onDecline={handleDisclaimerDecline}
					action={action}
				/>
			)}
		</div>
	)
}

export default Settings
