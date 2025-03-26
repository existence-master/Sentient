"use client"

import React, { useEffect, useState } from "react"
import GraphVisualization from "@components/GraphViz"
import Sidebar from "@components/Sidebar"
import { fetchGraphData } from "@utils/neo4j"
import {
	IconInfoCircle,
	IconRefresh,
	IconDatabase,
	IconBrain,
	IconTrash // Added for Clear All Memories button
} from "@tabler/icons-react"
import toast from "react-hot-toast"
import ProIcon from "@components/ProIcon"
import ShiningButton from "@components/ShiningButton"
import SQLiteMemoryDisplay from "@components/SQLiteMemoryDisplay"

const Memories = () => {
	const [userDetails, setUserDetails] = useState({})
	const [personalityType, setPersonalityType] = useState("")
	const [showTooltip, setShowTooltip] = useState(false)
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	const [isInputVisible, setInputVisible] = useState(false)
	const [newGraphInfo, setNewGraphInfo] = useState("")
	const [graphData, setGraphData] = useState({ nodes: [], edges: [] })
	const [addMemoriesLoading, setAddMemoriesLoading] = useState(false)
	const [recreateGraphLoading, setRecreateGraphLoading] = useState(false)
	const [pricing, setPricing] = useState("free")
	const [credits, setCredits] = useState(null)
	const [memoryDisplayType, setMemoryDisplayType] = useState("neo4j")
	const [clearMemoriesLoading, setClearMemoriesLoading] = useState(false) // New state for clearing memories

	// Existing fetch functions remain unchanged
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
			const response = await window.electron?.invoke("get-db-data")
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

	const loadGraphData = async () => {
		try {
			const data = await fetchGraphData()
			setGraphData(data)
		} catch (error) {
			toast.error(`Error loading graph data: ${error}`)
		}
	}

	const handleRecreateGraph = async () => {
		setRecreateGraphLoading(true)
		try {
			const response = await window.electron?.invoke("recreate-graph")
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
		setAddMemoriesLoading(true)
		try {
			if (!newGraphInfo.trim()) {
				toast.error("Graph information cannot be empty.")
				return
			}
			const result = await window.electron?.invoke("customize-graph", {
				newGraphInfo
			})
			if (result.status === 200) {
				await loadGraphData()
				setNewGraphInfo("")
				toast.success("Graph customized successfully!")
			} else {
				toast.error(`Failed to customize graph: ${result.error}`)
			}
		} catch (error) {
			toast.error(`Error customizing graph: ${error.message}`)
		} finally {
			setAddMemoriesLoading(false)
			setInputVisible(false)
		}
	}

	// New handler for clearing all memories
	const handleClearAllMemories = async () => {
		setClearMemoriesLoading(true)
		try {
			const response = await window.electron.invoke("clear-all-memories")
			if (response.error) {
				toast.error(response.error)
			} else {
				toast.success("All memories cleared successfully")
				// Note: You may need to trigger a refresh in SQLiteMemoryDisplay
			}
		} catch (error) {
			toast.error("Failed to clear memories")
		} finally {
			setClearMemoriesLoading(false)
		}
	}

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
		loadGraphData()
	}, [])

	return (
		<div className="flex h-screen w-screen bg-matteblack text-white relative">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
				fromChat={false}
			/>
			<div className="w-4/5 flex flex-col justify-center items-start h-full bg-matteblack relative overflow-hidden">
				<div className="w-4/5 flex justify-between">
					<div className="w-full px-4 py-4">
						<h1 className="font-Poppins text-white text-6xl py-4">
							Memories
						</h1>
					</div>
					{personalityType && (
						<div className="mt-4 flex flex-col items-end space-y-4 z-20">
							<div className="flex space-x-4">
								{personalityType
									.split("")
									.map((trait, index) => (
										<div
											key={index}
											className="flex flex-col items-center hover-button p-3 rounded-lg shadow-lg w-12"
										>
											<h3 className="text-3xl font-extrabold text-white mb-1">
												{trait}
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
									className="w-6 h-6 text-white cursor-pointer"
									aria-label="More info"
								/>
								{showTooltip && (
									<div className="absolute top-10 right-0 bg-matteblack border border-white text-white text-sm p-4 rounded-xl shadow-lg w-72">
										<h2 className="font-bold mb-2 text-xl text-center">
											What does this mean?
										</h2>
										<div className="space-y-2">
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
														</p>
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
						<span>Long Term Memories</span>
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
						<span>Short Term Memories</span>
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

				{/* Updated button section with conditional rendering */}
				<div className="fixed bottom-5 right-5 flex flex-col items-end space-y-4">
					{memoryDisplayType === "neo4j" && (
						<>
							<div>
								{pricing !== "free" || credits > 0 ? (
									<ShiningButton
										onClick={() =>
											setInputVisible((prev) => !prev)
										}
									>
										{isInputVisible
											? "Cancel"
											: "Add your own memories"}
									</ShiningButton>
								) : (
									<button
										disabled
										className="relative flex items-center justify-center py-2 px-6 rounded-lg font-bold text-white text-lg bg-matteblack border"
									>
										<span className="mr-2">
											Add your own memories
										</span>
										{pricing === "free" && <ProIcon />}
									</button>
								)}
							</div>
							{isInputVisible &&
								(pricing !== "free" || credits > 0) && (
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
											onClick={handleCustomizeGraph}
											disabled={addMemoriesLoading}
										>
											{addMemoriesLoading
												? "Updating..."
												: "Submit"}
										</button>
									</div>
								)}
							<button
								className="bg-red-600 text-white py-2 px-6 rounded-lg hover:bg-red-700 transition flex items-center gap-2"
								onClick={handleRecreateGraph}
								disabled={recreateGraphLoading}
							>
								{recreateGraphLoading
									? "Recreating..."
									: "Recreate Graph"}
								<IconRefresh className="w-5 h-5" />
							</button>
						</>
					)}
					{memoryDisplayType === "sqlite" && (
						<button
							className="bg-red-600 text-white py-2 px-6 rounded-lg hover:bg-red-700 transition flex items-center gap-2"
							onClick={handleClearAllMemories}
							disabled={clearMemoriesLoading}
						>
							{clearMemoriesLoading
								? "Clearing..."
								: "Clear All Memories"}
							<IconTrash className="w-5 h-5" />
						</button>
					)}
				</div>
			</div>
		</div>
	)
}

export default Memories