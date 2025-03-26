"use client"

import React from "react"
import { useEffect, useState } from "react" // Importing necessary React hooks
import GraphVisualization from "@components/GraphViz" // Component for visualizing the graph data
import Sidebar from "@components/Sidebar" // Sidebar component for navigation
import { fetchGraphData } from "@utils/neo4j" // Utility function to fetch graph data from Neo4j
import {
	IconInfoCircle,
	IconRefresh,
	IconDatabase,
	IconBrain
} from "@tabler/icons-react" // Icons from tabler-icons-react library
import toast from "react-hot-toast" // Library for displaying toast notifications
import ProIcon from "@components/ProIcon" // Component to indicate Pro features
import ShiningButton from "@components/ShiningButton" // Custom button component with a shining effect
import SQLiteMemoryDisplay from "@components/SQLiteMemoryDisplay" // Assuming this component exists and is for SQLite memory display

/**
 * Profile Component - Displays the user profile, including personality type and knowledge graph.
 *
 * This component fetches and displays user details, personality type, and a graph visualization of user knowledge.
 * It allows users to customize their graph by adding memories and recreate the graph.
 *
 * @returns {React.ReactNode} - The Profile component UI.
 */
const Profile = () => {
	// State to store user details fetched from the backend.
	const [userDetails, setUserDetails] = useState({}) // userDetails: any - Structure depends on backend response
	// State to store the user's personality type, fetched from the database.
	const [personalityType, setPersonalityType] = useState("") // personalityType: string
	// State to control the visibility of the personality type tooltip.
	const [showTooltip, setShowTooltip] = useState(false) // showTooltip: boolean
	// State to manage the visibility of the sidebar.
	const [isSidebarVisible, setSidebarVisible] = useState(false) // isSidebarVisible: boolean
	// State to control the visibility of the input area for adding graph information.
	const [isInputVisible, setInputVisible] = useState(false) // isInputVisible: boolean
	// State to store the new graph information input by the user.
	const [newGraphInfo, setNewGraphInfo] = useState("") // newGraphInfo: string
	// State to hold the data for graph visualization (nodes and edges).
	const [graphData, setGraphData] = useState({ nodes: [], edges: [] }) // graphData: { nodes: Node[], edges: Edge[] }
	// State to indicate if the process of adding memories to the graph is loading.
	const [addMemoriesLoading, setAddMemoriesLoading] = useState(false) // addMemoriesLoading: boolean
	// State to indicate if the process of recreating the graph is loading.
	const [recreateGraphLoading, setRecreateGraphLoading] = useState(false) // recreateGraphLoading: boolean
	// State to store the user's pricing plan, defaults to "free".
	const [pricing, setPricing] = useState("free") // pricing: string ("free" | "pro")
	// State to store the user's pro credits, initially null.
	const [credits, setCredits] = useState(null) // credits: number | null
	const [memoryDisplayType, setMemoryDisplayType] = useState("neo4j")

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
			toast.error(`Error fetching user details: ${error}`) // Show error toast if fetching fails
		}
	}

	/**
	 * Fetches the user's personality type from the database.
	 *
	 * Uses electron invoke to retrieve personality type data from the database and updates the `personalityType` state.
	 * Handles errors with a toast notification.
	 *
	 * @async
	 * @function fetchPersonalityType
	 * @returns {Promise<void>}
	 */
	const fetchPersonalityType = async () => {
		try {
			const response = await window.electron?.invoke("get-db-data") // Invoke electron to get data from database
			if (response?.data?.personalityType) {
				setPersonalityType(response.data.personalityType) // Set personalityType state if data is available
			}
		} catch (error) {
			toast.error(`Error fetching personality type: ${error}`) // Show error toast if fetching fails
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
			toast.error(`Error fetching pricing plan: ${error}`) // Show error toast if fetching fails
		}
	}

	/**
	 * Fetches the user's pro credits balance.
	 *
	 * Calls the electron backend to get the current pro credits balance and updates the `credits` state.
	 * Defaults to 0 if credits cannot be fetched or in case of an error.
	 *
	 * @async
	 * @function fetchProCredits
	 * @returns {Promise<void>}
	 */
	const fetchProCredits = async () => {
		try {
			const response = await window.electron?.invoke("fetch-pro-credits") // Invoke electron backend to get pro credits
			setCredits(response || 0) // Set credits state with fetched credits or default to 0
		} catch (error) {
			toast.error(`Error fecching pro credits: ${error}`) // Show error toast if fetching fails
		}
	}

	/**
	 * Loads graph data for visualization.
	 *
	 * Uses the `fetchGraphData` utility function to retrieve graph data and updates the `graphData` state.
	 * Handles errors during data fetching with a toast notification.
	 *
	 * @async
	 * @function loadGraphData
	 * @returns {Promise<void>}
	 */
	const loadGraphData = async () => {
		try {
			const data = await fetchGraphData() // Fetch graph data using utility function
			setGraphData(data) // Update graphData state with fetched data
		} catch (error) {
			toast.error(`Error loading graph data: ${error}`) // Show error toast if loading fails
		}
	}

	/**
	 * Handles the action to recreate the knowledge graph.
	 *
	 * Sets `recreateGraphLoading` to true to indicate loading state, invokes the electron backend to recreate the graph,
	 * reloads graph data on success, and displays success or error toast notifications.
	 * Resets `recreateGraphLoading` to false in the finally block.
	 *
	 * @async
	 * @function handleRecreateGraph
	 * @returns {Promise<void>}
	 */
	const handleRecreateGraph = async () => {
		setRecreateGraphLoading(true) // Set recreateGraphLoading state to true to indicate loading
		try {
			const response = await window.electron?.invoke("recreate-graph") // Invoke electron backend to recreate graph

			if (response?.status === 200) {
				await loadGraphData() // Reload graph data after successful recreation
				toast.success("Graph recreated successfully!") // Show success toast
			} else {
				toast.error(
					`Failed to recreate graph: ${
						response?.error || "Unknown error"
					}`
				) // Show error toast if recreation fails
			}
		} catch (error) {
			toast.error(`Error recreating graph: ${error.message}`) // Show error toast if any error occurs
		} finally {
			setRecreateGraphLoading(false) // Set recreateGraphLoading state to false regardless of success or failure
		}
	}

	/**
	 * Handles the action to customize the knowledge graph by adding new information.
	 *
	 * Sets `addMemoriesLoading` to true, validates input, invokes electron backend to customize graph with `newGraphInfo`,
	 * reloads graph data on success, clears input field, and displays success or error toast notifications.
	 * Resets loading states and input visibility in the finally block.
	 *
	 * @async
	 * @function handleCustomizeGraph
	 * @returns {Promise<void>}
	 */
	const handleCustomizeGraph = async () => {
		setAddMemoriesLoading(true) // Set addMemoriesLoading to true to indicate loading
		try {
			if (!newGraphInfo.trim()) {
				toast.error("Graph information cannot be empty.") // Show error toast if input is empty
				return // Exit function if input is empty
			}

			const result = await window.electron?.invoke("customize-graph", {
				// Invoke electron backend to customize graph
				newGraphInfo // Pass newGraphInfo to backend
			})

			if (result.status === 200) {
				await loadGraphData() // Reload graph data after successful customization
				setNewGraphInfo("") // Clear newGraphInfo input field
				toast.success("Graph customized successfully!") // Show success toast
			} else {
				toast.error(`Failed to customize graph: ${result.error}`) // Show error toast if customization fails
			}
		} catch (error) {
			toast.error(`Error customizing graph: ${error.message}`) // Show error toast if any error occurs
		} finally {
			setAddMemoriesLoading(false) // Set addMemoriesLoading to false regardless of success or failure
			setInputVisible(false) // Hide input area after submission or error
		}
	}

	/**
	 * Descriptions for personality traits, used in the tooltip.
	 *
	 * An object mapping each personality trait letter (E, I, S, N, T, F, J, P) to its description.
	 * Used to provide detailed information about each trait in the personality tooltip.
	 */
	const descriptions = {
		E: "Extroverts are outgoing and gain energy from being around others.",
		I: "Introverts are reserved and gain energy from spending time alone.",
		S: "Sensors focus on facts and details, preferring practical approaches.",
		N: "Intuitives look at the bigger picture and focus on abstract concepts.",
		T: "Thinkers base decisions on logic and objectivity.",
		F: "Feelers prioritize emotions and value empathy and harmony.",
		J: "Judgers prefer structure, organization, and planning.",
		P: "Perceivers are flexible and enjoy spontaneity and adaptability."
	}

	/**
	 * useEffect hook to fetch user details, personality type, pricing plan, pro credits, and graph data on component mount.
	 *
	 * Calls various data fetching functions when the component is first rendered to populate the profile page with user information and graph visualization.
	 */
	useEffect(() => {
		fetchUserDetails() // Fetch user details on component mount
		fetchPersonalityType() // Fetch personality type on component mount
		fetchPricingPlan() // Fetch pricing plan on component mount
		fetchProCredits() // Fetch pro credits on component mount
		loadGraphData() // Load graph data on component mount
	}, []) // Empty dependency array ensures this effect runs only once on mount

	console.log("userDetails", userDetails)	

	/**
	 * Main return statement for the Profile component, rendering the profile UI.
	 *
	 * Includes sidebar, profile header, personality type display with tooltip, graph visualization,
	 * input area for customizing graph (conditional based on pricing plan and credits), and buttons for actions like recreate graph.
	 *
	 * @returns {React.ReactNode} - The main UI for the Profile component.
	 */
	return (
		<div className="flex h-screen w-screen bg-matteblack text-white relative">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
				fromChat={false} // Indicates sidebar is not in chat view
			/>
			<div className="w-4/5 flex flex-col justify-center items-start h-full bg-matteblack relative overflow-hidden">
				<div className="w-4/5 flex justify-between">
					<div className="w-full px-4 py-4">
						<h1 className="font-Poppins text-white text-6xl py-4">
							Profile
						</h1>{" "}
						{/* Profile page title */}
					</div>
					{personalityType && ( // Conditional rendering for personality type display
						<div className="mt-4 flex flex-col items-end space-y-4 z-20">
							<div className="flex space-x-4">
								{/* Map over each trait letter to display personality traits */}
								{personalityType
									.split("")
									.map((trait, index) => (
										<div
											key={index}
											className="flex flex-col items-center hover-button p-3 rounded-lg  shadow-lg w-12"
										>
											<h3 className="text-3xl font-extrabold text-white mb-1">
												{trait}
											</h3>{" "}
											{/* Display individual personality trait letters */}
										</div>
									))}
							</div>

							<div
								className="relative"
								onMouseEnter={() => setShowTooltip(true)} // Show tooltip on mouse enter
								onMouseLeave={() => setShowTooltip(false)} // Hide tooltip on mouse leave
							>
								<IconInfoCircle
									className="w-6 h-6 text-white cursor-pointer"
									aria-label="More info"
								/>
								{showTooltip && ( // Conditional rendering for personality tooltip
									<div className="absolute top-10 right-0 bg-matteblack border border-white text-white text-sm p-4 rounded-xl shadow-lg w-72">
										<h2 className="font-bold mb-2 text-xl text-center">
											What does this mean?
										</h2>{" "}
										{/* Tooltip title */}
										<div className="space-y-2">
											{/* Map over each trait letter to display trait descriptions in tooltip */}
											{personalityType
												.split("")
												.map((trait, index) => (
													<div
														key={index}
														className="flex flex-row gap-x-2 items-center"
													>
														<div className="hover-button rounded-xl w-8 h-8 p-4 flex items-center justify-center text-white text-xl font-bold">
															{trait}
														</div>
														<p className="text-gray-300">
															{
																descriptions[
																	trait
																]
															}
														</p>{" "}
														{/* Display personality trait descriptions */}
													</div>
												))}
										</div>
									</div>
								)}
							</div>
						</div>
					)}
				</div>

				<div className="flex items-center space-x-4">
					<button
						onClick={() => setMemoryDisplayType("neo4j")}
						className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
							memoryDisplayType === "neo4j"
								? "bg-blue-600 text-white"
								: "bg-gray-700 text-gray-300 hover:bg-gray-600"
						}`}
					>
						<IconDatabase className="w-5 h-5" />
						<span>Neo4j Memories</span>
					</button>
					<button
						onClick={() => setMemoryDisplayType("sqlite")}
						className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
							memoryDisplayType === "sqlite"
								? "bg-blue-600 text-white"
								: "bg-gray-700 text-gray-300 hover:bg-gray-600"
						}`}
					>
						<IconBrain className="w-5 h-5" />
						<span>SQLite Memories</span>
					</button>
				</div>

				<div className="flex items-start justify-center h-full w-[80%]">
					{memoryDisplayType === "neo4j" ? (
						<GraphVisualization
							nodes={graphData.nodes}
							edges={graphData.edges}
						/>
					) : (
						<SQLiteMemoryDisplay userDetails={userDetails} />
					)}
				</div>

				<div className="fixed bottom-5 right-5 flex flex-col items-end space-y-4">
					<div>
						{/* Conditional rendering for "Add your own memories" button based on pricing plan and credits */}
						{pricing !== "free" || credits > 0 ? (
							<ShiningButton
								onClick={() => setInputVisible((prev) => !prev)}
							>
								{isInputVisible
									? "Cancel"
									: "Add your own memories"}
							</ShiningButton>
						) : (
							<button
								disabled // Disable button for free plan users without credits
								className="relative flex items-center justify-center py-2 px-6 rounded-lg font-bold text-white text-lg bg-matteblack border"
							>
								<span className="mr-2">
									Add your own memories
								</span>
								{pricing === "free" && <ProIcon />}
								{/* Show ProIcon for free plan users */}
							</button>
						)}
					</div>
					{/* Conditional rendering for input area to customize graph, shown when isInputVisible is true and for non-free plans or users with credits */}
					{isInputVisible && (pricing !== "free" || credits > 0) && (
						<div className="bg-gray-800 p-4 rounded-lg shadow-lg w-96 space-y-4">
							<textarea
								className="w-full p-2 border rounded-sm bg-gray-900 text-white resize-none h-24 focus:outline-hidden"
								placeholder="Enter new information to add, update or delete your own memories..."
								value={newGraphInfo}
								onChange={(e) =>
									setNewGraphInfo(e.target.value)
								}
							/>
							<button
								className="bg-green-600 text-white py-2 px-4 rounded-sm hover:bg-green-700 transition"
								onClick={handleCustomizeGraph} // Handler for submitting graph customization
								disabled={addMemoriesLoading} // Disable button during loading
							>
								{addMemoriesLoading ? "Updating..." : "Submit"}
							</button>
						</div>
					)}

					<button
						className="bg-red-600 text-white py-2 px-6 rounded-lg hover:bg-red-700 transition flex items-center gap-2"
						onClick={handleRecreateGraph} // Handler for recreating graph
						disabled={recreateGraphLoading} // Disable button during loading
					>
						{recreateGraphLoading
							? "Recreating..."
							: "Recreate Graph"}
						<IconRefresh className="w-5 h-5" />
					</button>
				</div>
			</div>
		</div>
	)
}

export default Profile