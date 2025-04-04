"use client"

import React, { useEffect, useState, useCallback } from "react"
import GraphVisualization from "@components/GraphViz"
import Sidebar from "@components/Sidebar"
// ADDED: Icons needed for this page
import {
	IconInfoCircle,
	IconRefresh,
	IconDatabase, // Used in switcher and actions
	IconBrain, // Used in switcher and actions
	IconTrash,
	IconPlus, // For add memory button
	IconSettingsCog, // Placeholder for Customize/Recreate actions
	IconLoader,
} from "@tabler/icons-react"
import toast from "react-hot-toast"
import ProIcon from "@components/ProIcon"
// REMOVED: ShiningButton import (can be re-added if desired for action buttons)
import SQLiteMemoryDisplay from "@components/SQLiteMemoryDisplay"
// ADDED: Import the new switcher component
import MemoryTypeSwitcher from "@components/MemoryTypeSwitcher"
import { Tooltip } from "react-tooltip" // Keep Tooltip
import "react-tooltip/dist/react-tooltip.css" // Keep Tooltip CSS

const Memories = () => {
	const [userDetails, setUserDetails] = useState({})
	const [personalityType, setPersonalityType] = useState("")
	const [showTooltip, setShowTooltip] = useState(false)
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	// State for controlling customize input visibility
	const [isCustomizeInputVisible, setCustomizeInputVisible] = useState(false)
	const [newGraphInfo, setNewGraphInfo] = useState("")
	const [graphData, setGraphData] = useState({ nodes: [], edges: [] })
	const [addMemoriesLoading, setAddMemoriesLoading] = useState(false)
	const [recreateGraphLoading, setRecreateGraphLoading] = useState(false)
	const [pricing, setPricing] = useState("free")
	const [credits, setCredits] = useState(null)
	// State to control which memory view is active: 'neo4j' (graph) or 'sqlite' (list)
	const [memoryDisplayType, setMemoryDisplayType] = useState("neo4j") // Default to graph view
	const [clearMemoriesLoading, setClearMemoriesLoading] = useState(false)
	// State for graph loading indicator
	const [graphLoading, setGraphLoading] = useState(false)
	// ADDED: State for triggering refresh in SQLite view
	const [refreshSqlite, setRefreshSqlite] = useState(0)

	// --- Fetching Data ---
	const fetchUserDetails = async () => {
		try {
			const response = await window.electron?.invoke("get-profile")
			setUserDetails(response)
		} catch (error) {
			toast.error(`Error fetching user details: ${error}`)
		}
	}

	const fetchPersonalityType = async () => {
		try {
			const response = await window.electron?.invoke("get-user-data")
			if (response?.data?.personalityType) {
				setPersonalityType(response.data.personalityType)
			}
		} catch (error) {
			toast.error(`Error fetching personality type: ${error}`)
		}
	}

	const fetchPricingPlan = async () => {
		try {
			const response = await window.electron?.invoke("fetch-pricing-plan")
			setPricing(response || "free")
		} catch (error) {
			toast.error(`Error fetching pricing plan: ${error}`)
		}
	}
	const fetchProCredits = async () => {
		try {
			const response = await window.electron?.invoke("fetch-pro-credits")
			setCredits(response || 0)
		} catch (error) {
			toast.error(`Error fetching pro credits: ${error}`)
		}
	}

	const loadGraphData = useCallback(async () => {
		console.log("Attempting to load graph data via IPC...")
		setGraphLoading(true)
		setGraphData({ nodes: [], edges: [] })
		try {
			const response = await window.electron?.invoke(
				"fetch-long-term-memories"
			)
			if (response?.error) {
				console.error(
					"Error received from fetch-long-term-memories IPC:",
					response.error
				)
				toast.error(`Error loading graph data: ${response.error}`)
				setGraphData({ nodes: [], edges: [] })
			} else if (response?.nodes && response?.edges) {
				if (
					Array.isArray(response.nodes) &&
					Array.isArray(response.edges)
				) {
					console.log("Graph data loaded successfully via IPC.")
					setGraphData({
						nodes: response.nodes,
						edges: response.edges
					})
				} else {
					console.error(
						"Invalid data format received from fetch-long-term-memories IPC:",
						response
					)
					toast.error("Received invalid graph data format.")
					setGraphData({ nodes: [], edges: [] })
				}
			} else {
				console.error(
					"Unexpected response structure from fetch-long-term-memories IPC:",
					response
				)
				toast.error("Failed to load graph data: Unexpected response.")
				setGraphData({ nodes: [], edges: [] })
			}
		} catch (error) {
			console.error("Error invoking fetch-long-term-memories IPC:", error)
			toast.error(`Error loading graph data: ${error.message}`)
			setGraphData({ nodes: [], edges: [] })
		} finally {
			setGraphLoading(false)
		}
	}, []) // Empty dependency array for useCallback

	// --- Action Handlers ---
	const handleRecreateGraph = async () => {
		setRecreateGraphLoading(true)
		try {
			const response = await window.electron?.invoke(
				"reset-long-term-memories"
			)
			if (response?.status === 200) {
				await loadGraphData()
				toast.success("Graph recreated successfully!")
			} else {
				toast.error(
					`Failed to recreate graph: ${response?.error || "Unknown error"}`
				)
			}
		} catch (error) {
			toast.error(`Error recreating graph: ${error.message}`)
		} finally {
			setRecreateGraphLoading(false)
		}
	}

	const handleCustomizeGraph = async () => {
		if (!newGraphInfo.trim()) {
			toast.error("Information cannot be empty.")
			return
		}
		setAddMemoriesLoading(true)
		try {
			const result = await window.electron?.invoke(
				"customize-long-term-memories",
				{ newGraphInfo }
			) // Pass object
			if (result.status === 200) {
				await loadGraphData()
				setNewGraphInfo("")
				toast.success("Graph customized successfully!")
			} else {
				toast.error(
					`Failed to customize graph: ${result.error || "Unknown error"}`
				)
			}
		} catch (error) {
			toast.error(`Error customizing graph: ${error.message}`)
		} finally {
			setAddMemoriesLoading(false)
			setCustomizeInputVisible(false)
		} // Close input on finish
	}

	const handleClearAllMemories = async () => {
		setClearMemoriesLoading(true)
		try {
			const response = await window.electron.invoke(
				"clear-all-short-term-memories",
				{ user_id: userDetails?.sub || "default_user" }
			) // Pass user_id if available
			if (response.error) {
				toast.error(`Failed to clear memories: ${response.error}`)
			} else {
				toast.success("All short-term memories cleared successfully")
				setRefreshSqlite((prev) => prev + 1)
			} // Trigger refresh
		} catch (error) {
			toast.error("Failed to clear memories")
		} finally {
			setClearMemoriesLoading(false)
		}
	}

	// Descriptions for personality traits
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

	useEffect(() => {
		fetchUserDetails()
		fetchPersonalityType()
		fetchPricingPlan()
		fetchProCredits()
		// Load graph data only if graph view is default
		if (memoryDisplayType === "neo4j") {
			loadGraphData()
		}
	}, [loadGraphData, memoryDisplayType]) // Rerun if loadGraphData changes (it shouldn't) or if default type changes

	return (
		// MODIFIED: Overall page structure uses flex
		<div className="h-screen bg-matteblack flex relative overflow-hidden dark">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
			/>
			{/* MODIFIED: Main content area with flex-grow */}
			<div className="flex-grow flex flex-col h-full bg-matteblack text-white relative overflow-hidden p-6 gap-4">
				{" "}
				{/* Added gap */}
				{/* --- Top Section: Heading and Switcher --- */}
				<div className="flex justify-between items-center px-4 pt-2 flex-shrink-0">
					<h1 className="font-Poppins text-white text-4xl font-light">
						{" "}
						{/* Adjusted size */}
						Memories
					</h1>
					{/* ADDED: Memory Type Switcher Component */}
					<MemoryTypeSwitcher
						currentType={memoryDisplayType}
						onTypeChange={setMemoryDisplayType}
					/>
					{/* Personality display remains similar, maybe adjust position if needed */}
					{personalityType && (
						<div className="flex flex-col items-end space-y-1">
							{" "}
							{/* Reduced spacing */}
							<div className="flex space-x-2">
								{" "}
								{/* Reduced spacing */}
								{personalityType
									.split("")
									.map((trait, index) => (
										<div
											key={index}
											className="flex flex-col items-center bg-neutral-700/50 p-2 rounded-md shadow w-10"
										>
											{" "}
											{/* Adjusted style */}
											<h3 className="text-xl font-semibold text-white">
												{" "}
												{trait}{" "}
											</h3>
										</div>
									))}
							</div>
							<div
								className="relative"
								onMouseEnter={() => setShowTooltip(true)}
								onMouseLeave={() => setShowTooltip(false)}
							>
								<IconInfoCircle
									className="w-5 h-5 text-gray-400 cursor-pointer hover:text-white"
									aria-label="More info"
								/>
								{showTooltip && (
									<div className="absolute top-full right-0 mt-2 bg-neutral-800 border border-neutral-700 text-white text-xs p-3 rounded-lg shadow-lg w-64 z-50">
										{" "}
										{/* Adjusted style/size */}{" "}
										<h2 className="font-bold mb-2 text-sm text-center">
											{" "}
											Personality Type{" "}
										</h2>{" "}
										<div className="space-y-1.5">
											{" "}
											{personalityType
												.split("")
												.map((trait, index) => (
													<div
														key={index}
														className="flex items-center gap-2"
													>
														{" "}
														<span className="font-bold text-lightblue w-4">
															{trait}
														</span>{" "}
														<p className="text-gray-300">
															{
																descriptions[
																	trait
																]
															}
														</p>{" "}
													</div>
												))}{" "}
										</div>{" "}
									</div>
								)}
							</div>
						</div>
					)}
				</div>
				{/* --- Memory View Area --- */}
				{/* MODIFIED: Takes remaining space, handles overflow */}
				<div className="flex-grow w-full relative overflow-hidden rounded-lg bg-neutral-900/30 border border-neutral-800 shadow-inner">
					{/* Conditional Rendering based on memoryDisplayType */}
					{memoryDisplayType === "neo4j" ? (
						// --- Graph View ---
						<>
							{graphLoading && ( // Loading overlay for graph
								<div className="absolute inset-0 flex items-center justify-center bg-matteblack/80 z-10">
									<IconLoader className="w-8 h-8 animate-spin text-lightblue" />
									<span className="ml-3 text-lg">
										Loading Knowledge Graph...
									</span>
								</div>
							)}
							{/* Ensure GraphVisualization fills container */}
							<GraphVisualization
								nodes={graphData.nodes}
								edges={graphData.edges}
							/>
						</>
					) : (
						// --- Short Term Memory List View ---
						<SQLiteMemoryDisplay
							userDetails={userDetails}
							refreshTrigger={refreshSqlite} // Pass trigger to re-fetch on clear
						/>
					)}
				</div>
				{/* --- Action Buttons Area (Bottom Right) --- */}
				<div className="absolute bottom-6 right-6 flex flex-col items-end space-y-3 z-40">
					{memoryDisplayType === "neo4j" && (
						<>
							{/* Input section for adding/customizing graph */}
							{isCustomizeInputVisible &&
								(pricing !== "free" || credits > 0) && (
									<div className="bg-neutral-800 p-4 rounded-lg shadow-lg w-96 space-y-3 border border-neutral-700">
										<textarea
											className="w-full p-2.5 border border-neutral-600 rounded-md bg-neutral-700 text-white resize-none h-24 focus:outline-none focus:border-lightblue text-sm"
											placeholder="Enter information to add, update, or delete from your long-term memory..."
											value={newGraphInfo}
											onChange={(e) =>
												setNewGraphInfo(e.target.value)
											}
										/>
										<button
											className="w-full bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 transition text-sm font-medium disabled:opacity-50"
											onClick={handleCustomizeGraph}
											disabled={
												addMemoriesLoading ||
												!newGraphInfo.trim()
											}
										>
											{addMemoriesLoading ? (
												<IconLoader className="w-5 h-5 animate-spin mx-auto" />
											) : (
												"Submit Information"
											)}
										</button>
									</div>
								)}

							{/* Button to toggle customize input */}
							<div>
								{" "}
								{/* Wrap button for better layout control if needed */}
								{pricing !== "free" || credits > 0 ? (
									<button
										onClick={() =>
											setCustomizeInputVisible(
												(prev) => !prev
											)
										}
										className="flex items-center gap-2 py-2 px-4 rounded-full bg-darkblue hover:bg-lightblue text-white text-sm font-medium transition-colors shadow-md"
									>
										<IconSettingsCog className="w-5 h-5" />{" "}
										{/* Example Icon */}
										{isCustomizeInputVisible
											? "Cancel"
											: "Customize Memories"}
									</button>
								) : (
									// Disabled state for free users without credits
									<button
										disabled
										className="relative flex items-center justify-center py-2 px-4 rounded-full font-medium text-white text-sm bg-neutral-700 cursor-not-allowed opacity-60 shadow-md"
									>
										<IconSettingsCog className="w-5 h-5 mr-2" />
										Customize Memories
										{pricing === "free" && (
											<ProIcon className="ml-1.5" />
										)}
									</button>
								)}
							</div>

							{/* Recreate Graph Button */}
							<button
								className="flex items-center gap-2 py-2 px-4 rounded-full bg-red-600 hover:bg-red-500 text-white text-sm font-medium transition-colors shadow-md disabled:opacity-50"
								onClick={handleRecreateGraph}
								disabled={recreateGraphLoading || graphLoading}
							>
								{recreateGraphLoading ? (
									<IconLoader className="w-5 h-5 animate-spin" />
								) : (
									<IconRefresh className="w-5 h-5" />
								)}
								{recreateGraphLoading
									? "Recreating..."
									: "Recreate Graph"}
							</button>
						</>
					)}
					{memoryDisplayType === "sqlite" && (
						// Button to clear short-term memories
						<button
							className="flex items-center gap-2 py-2 px-4 rounded-full bg-red-600 hover:bg-red-500 text-white text-sm font-medium transition-colors shadow-md disabled:opacity-50"
							onClick={handleClearAllMemories}
							disabled={clearMemoriesLoading}
						>
							{clearMemoriesLoading ? (
								<IconLoader className="w-5 h-5 animate-spin" />
							) : (
								<IconTrash className="w-5 h-5" />
							)}
							{clearMemoriesLoading
								? "Clearing..."
								: "Clear Short-Term"}
						</button>
						// Add Memory button could also go here, triggering a modal in SQLiteMemoryDisplay
					)}
				</div>
			</div>{" "}
			{/* End Main Content Area */}
		</div> // End Page Container
	)
}

export default Memories
