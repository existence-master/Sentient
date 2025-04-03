"use client"

// ADDED: Import necessary icons for status etc.
import React, { useState, useEffect, useCallback } from "react"
import {
	IconLoader,
	IconPencil,
	IconTrash,
	IconPlus,
	IconX,
	IconHelpCircle,
	IconSearch,
	IconRefresh,
	IconClock,
	IconPlayerPlay,
	IconCircleCheck,
	IconMailQuestion,
	IconAlertCircle,
	IconFilter,
	IconChevronUp, // For potential drop-up visual cue if needed
	IconSend // Using send icon for add task button
} from "@tabler/icons-react"
import Sidebar from "@components/Sidebar"
import toast from "react-hot-toast"
import { Tooltip } from "react-tooltip" // Keep tooltip if used elsewhere
import "react-tooltip/dist/react-tooltip.css" // Keep tooltip CSS
import { cn } from "@utils/cn" // Assuming cn utility is available

// --- Task Status Mapping ---
// ADDED: Mapping status strings to icons and colors for visual feedback
const statusMap = {
	pending: { icon: IconClock, color: "text-yellow-500", label: "Pending" },
	processing: {
		icon: IconPlayerPlay,
		color: "text-blue-500",
		label: "Processing"
	},
	completed: {
		icon: IconCircleCheck,
		color: "text-green-500",
		label: "Completed"
	},
	error: { icon: IconAlertCircle, color: "text-red-500", label: "Error" },
	approval_pending: {
		icon: IconMailQuestion,
		color: "text-purple-500",
		label: "Approval Pending"
	},
	cancelled: { icon: IconX, color: "text-gray-500", label: "Cancelled" }, // Assuming IconX exists
	default: { icon: IconHelpCircle, color: "text-gray-400", label: "Unknown" } // Assuming IconHelpCircle exists
}

// ADDED: Mapping for priority levels
const priorityMap = {
	0: { label: "High", color: "text-red-400" },
	1: { label: "Medium", color: "text-yellow-400" },
	2: { label: "Low", color: "text-green-400" },
	default: { label: "Unknown", color: "text-gray-400" }
}

const Tasks = () => {
	const [tasks, setTasks] = useState([])
	const [loading, setLoading] = useState(true)
	const [error, setError] = useState(null)
	const [userDetails, setUserDetails] = useState({})
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	const [newTaskDescription, setNewTaskDescription] = useState("")
	// MODIFIED: State for new task priority uses numbers (0, 1, 2), default to Medium
	const [newTaskPriorityLevel, setNewTaskPriorityLevel] = useState(1)
	const [editingTask, setEditingTask] = useState(null) // State to hold the task being edited
	const [filterStatus, setFilterStatus] = useState("all") // 'all', 'pending', 'processing', etc.
	const [searchTerm, setSearchTerm] = useState("")
	const [selectedTask, setSelectedTask] = useState(null) // For approval modal data

	// --- Fetching Data ---
	// MODIFIED: Wrapped fetch logic in useCallback
	const fetchTasksData = useCallback(async () => {
		console.log("Fetching tasks data...")
		setLoading(true) // Set loading true at the start of fetch
		setError(null)
		try {
			const response = await window.electron.invoke("fetch-tasks")
			console.log("Raw tasks response:", response)
			if (response.error) {
				console.error("Error fetching tasks:", response.error)
				setError(response.error)
				setTasks([]) // Clear tasks on error
			} else if (Array.isArray(response.tasks)) {
				// Sort tasks primarily by status (pending/processing first), then by priority, then by date
				const sortedTasks = response.tasks.sort((a, b) => {
					const statusOrder = {
						processing: 0,
						approval_pending: 1,
						pending: 2,
						error: 3,
						cancelled: 4,
						completed: 5
					}
					const statusA = statusOrder[a.status] ?? 99
					const statusB = statusOrder[b.status] ?? 99

					if (statusA !== statusB) return statusA - statusB // Sort by status first
					if (a.priority !== b.priority)
						return a.priority - b.priority // Then by priority (lower number = higher priority)
					// Then by creation date (newest first) - ensure created_at is available and comparable
					try {
						const dateA = a.created_at
							? new Date(a.created_at).getTime()
							: 0
						const dateB = b.created_at
							? new Date(b.created_at).getTime()
							: 0
						return dateB - dateA // Descending order for date
					} catch (dateError) {
						console.warn("Error comparing task dates:", dateError)
						return 0 // Keep original order if date parsing fails
					}
				})
				setTasks(sortedTasks)
				console.log("Tasks fetched and sorted:", sortedTasks.length)
			} else {
				console.error("Invalid tasks response format:", response)
				setError("Failed to fetch tasks: Invalid response format")
				setTasks([])
			}
		} catch (err) {
			console.error("Exception fetching tasks:", err)
			setError("Failed to fetch tasks: " + err.message)
			setTasks([])
		} finally {
			console.log("Finished fetching tasks, setting loading false.")
			setLoading(false) // Always set loading false at the end
		}
	}, []) // useCallback dependency array is empty

	const fetchUserDetails = async () => {
		try {
			const response = await window.electron?.invoke("get-profile")
			setUserDetails(response || {}) // Ensure userDetails is an object
		} catch (error) {
			toast.error("Error fetching user details for sidebar.")
			console.error("Error fetching user details for sidebar:", error)
		}
	}

	// --- Effects ---
	useEffect(() => {
		fetchUserDetails()
		fetchTasksData() // Initial fetch
		// Setup interval for refetching tasks data
		const intervalId = setInterval(fetchTasksData, 60000) // Refresh every 60 seconds
		// Cleanup interval on component unmount
		return () => clearInterval(intervalId)
	}, [fetchTasksData]) // Include fetchTasksData in dependency array due to useCallback

	// --- Task Actions ---
	const handleAddTask = async () => {
		if (!newTaskDescription.trim()) {
			// Check trimmed description
			toast.error("Please enter a task description.")
			return
		}
		// Priority is now guaranteed by the dropdown state

		console.log("Adding task:", {
			description: newTaskDescription,
			priority: newTaskPriorityLevel
		})
		try {
			const taskData = {
				description: newTaskDescription,
				priority: newTaskPriorityLevel // Use the numeric state
			}
			const response = await window.electron.invoke("add-task", taskData)
			if (response.error) {
				console.error("Error adding task via IPC:", response.error)
				toast.error(`Failed to add task: ${response.error}`)
			} else {
				toast.success("Task added successfully!")
				setNewTaskDescription("") // Clear input
				setNewTaskPriorityLevel(1) // Reset priority dropdown to Medium
				await fetchTasksData() // Refresh list immediately
			}
		} catch (error) {
			console.error("Exception adding task:", error)
			toast.error("Failed to add task: An unexpected error occurred.")
		}
	}

	const handleEditTask = (task) => {
		console.log("Editing task:", task)
		// Set the editing task state, including the numeric priority
		setEditingTask({
			...task
			// Priority is already a number in the task object, no conversion needed
			// priority: priorityNumberToString(task.priority) // OLD logic removed
		})
	}

	const handleUpdateTask = async () => {
		// Validate editing task data
		if (!editingTask || !editingTask.description?.trim()) {
			toast.error("Task description cannot be empty.")
			return
		}
		if (
			editingTask.priority === undefined ||
			editingTask.priority === null ||
			![0, 1, 2].includes(editingTask.priority)
		) {
			toast.error("Invalid priority selected.")
			return
		}

		console.log("Updating task:", editingTask.task_id, {
			description: editingTask.description,
			priority: editingTask.priority
		})
		try {
			// Call IPC invoke with the task ID and updated data (using numeric priority)
			const response = await window.electron.invoke("update-task", {
				taskId: editingTask.task_id,
				description: editingTask.description,
				priority: editingTask.priority // Use the numeric priority from state
			})
			if (response.error) {
				console.error("Error updating task via IPC:", response.error)
				toast.error(`Failed to update task: ${response.error}`)
			} else {
				toast.success("Task updated successfully!")
				setEditingTask(null) // Close the modal
				await fetchTasksData() // Refresh list
			}
		} catch (error) {
			console.error("Exception updating task:", error)
			toast.error("Failed to update task: An unexpected error occurred.")
		}
	}

	const handleDeleteTask = async (taskId) => {
		if (!taskId) return
		console.log("Deleting task:", taskId)
		// Optional: Add confirmation dialog here
		// if (!confirm("Are you sure you want to delete this task?")) { return; }
		try {
			const response = await window.electron.invoke("delete-task", taskId) // Pass only taskId
			if (response.error) {
				console.error("Error deleting task via IPC:", response.error)
				toast.error(`Failed to delete task: ${response.error}`)
			} else {
				toast.success("Task deleted successfully!")
				await fetchTasksData() // Refresh list
			}
		} catch (error) {
			console.error("Exception deleting task:", error)
			toast.error("Failed to delete task: An unexpected error occurred.")
		}
	}

	const handleViewApprovalData = async (taskId) => {
		if (!taskId) return
		console.log("Fetching approval data for task:", taskId)
		try {
			const response = await window.electron.invoke(
				"get-task-approval-data",
				taskId
			)
			console.log("Approval data response:", response)
			if (response.status === 200 && response.approval_data) {
				setSelectedTask({
					// Store task ID and the approval data
					taskId,
					approvalData: response.approval_data
				})
			} else {
				console.error(
					"Error fetching approval data:",
					response.error || "Invalid response"
				)
				toast.error(
					`Error fetching approval data: ${response.error || "No data found"}`
				)
			}
		} catch (error) {
			console.error("Exception fetching approval data:", error)
			toast.error(
				"Error fetching approval data: An unexpected error occurred."
			)
		}
	}

	const handleApproveTask = async (taskId) => {
		if (!taskId) return
		console.log("Approving task:", taskId)
		try {
			const response = await window.electron.invoke(
				"approve-task",
				taskId
			)
			console.log("Approve task response:", response)
			if (response.status === 200) {
				toast.success("Task approved and completed!")
				setSelectedTask(null) // Close modal
				await fetchTasksData() // Refresh list
			} else {
				console.error(
					"Error approving task:",
					response.error || "Invalid status"
				)
				toast.error(
					`Error approving task: ${response.error || "Approval failed"}`
				)
			}
		} catch (error) {
			console.error("Exception approving task:", error)
			toast.error("Error approving task: An unexpected error occurred.")
		}
	}

	// --- Filtering Logic ---
	const filteredTasks = tasks.filter((task) => {
		// Filter by status
		if (filterStatus !== "all" && task.status !== filterStatus) {
			return false
		}
		// Filter by search term (case-insensitive)
		if (
			searchTerm &&
			!task.description?.toLowerCase().includes(searchTerm.toLowerCase())
		) {
			return false
		}
		return true // Include task if it passes filters
	})

	// --- Render Loading/Error States ---
	if (loading && tasks.length === 0) {
		// Show full screen loader only on initial load
		return (
			<div className="flex justify-center items-center h-screen bg-matteblack">
				<IconLoader className="w-10 h-10 animate-spin text-white" />
				<span className="ml-2 text-white">Loading tasks...</span>
			</div>
		)
	}

	if (error) {
		return (
			<div className="flex justify-center items-center h-screen bg-matteblack text-red-500">
				Error loading tasks: {error}
				<button
					onClick={fetchTasksData}
					className="ml-4 p-2 bg-lightblue text-white rounded"
				>
					Retry
				</button>
			</div>
		)
	}

	console.log(filteredTasks)

	// --- Main Render ---
	return (
		// MODIFIED: Use flex for overall page structure
		<div className="h-screen bg-matteblack flex relative overflow-hidden dark">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
			/>
			{/* MODIFIED: Main content area takes remaining space */}
			<div className="flex-grow flex flex-col h-full bg-matteblack text-white relative overflow-hidden">
				{/* --- MODIFIED: Top Bar for Search/Filter --- */}
				<div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-30 w-full max-w-2xl px-4">
					<div className="flex items-center space-x-2 bg-neutral-800/80 backdrop-blur-sm rounded-full p-2 shadow-lg">
						{/* Search Input */}
						<IconSearch className="h-5 w-5 text-gray-400 ml-2 flex-shrink-0" />
						<input
							type="text"
							placeholder="Search tasks..."
							value={searchTerm}
							onChange={(e) => setSearchTerm(e.target.value)}
							className="bg-transparent text-white focus:outline-none w-full flex-grow px-2 placeholder-gray-500 text-sm"
						/>
						{/* Filter Dropdown */}
						<div className="relative flex-shrink-0">
							<IconFilter className="absolute left-2 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" />
							<select
								value={filterStatus}
								onChange={(e) =>
									setFilterStatus(e.target.value)
								}
								className="appearance-none bg-neutral-700 border border-neutral-600 text-white text-xs rounded-full pl-8 pr-3 py-1.5 focus:outline-none focus:border-lightblue cursor-pointer"
								title="Filter tasks by status"
							>
								<option value="all">All</option>
								<option value="pending">Pending</option>
								<option value="processing">Processing</option>
								<option value="approval_pending">
									Approval
								</option>
								<option value="completed">Completed</option>
								<option value="error">Error</option>
								<option value="cancelled">Cancelled</option>
							</select>
						</div>
					</div>
				</div>

				{/* --- MODIFIED: Refresh Button (Top Right) --- */}
				<div className="absolute top-5 right-5 z-30">
					<button
						onClick={fetchTasksData}
						className="p-2 rounded-full hover:bg-neutral-700/60 transition-colors text-gray-300"
						data-tooltip-id="refresh-tooltip"
						disabled={loading} // Disable while loading
					>
						{/* Show loader when loading, otherwise refresh icon */}
						{loading ? (
							<IconLoader className="h-5 w-5 animate-spin" />
						) : (
							<IconRefresh className="h-5 w-5" />
						)}
					</button>
					<Tooltip
						id="refresh-tooltip"
						content={
							loading
								? "Refreshing..."
								: "Refresh Tasks (Auto refreshes every minute)"
						}
						place="bottom"
					/>
				</div>

				{/* --- MODIFIED: Task List Container --- */}
				{/* Takes remaining vertical space, centered horizontally, scrolls internally */}
				{/* Added padding top/bottom to avoid overlap */}
				<div className="flex-grow w-full max-w-5xl mx-auto px-6 pt-24 pb-28 flex flex-col overflow-hidden">
					{/* Centered Heading */}
					{/* <h2 className="text-2xl font-semibold text-center mb-6 text-gray-300">Tasks</h2> */}
					{/* Task List Area */}
					<div className="flex-grow overflow-y-auto space-y-3 pr-2">
						{" "}
						{/* Added pr-2 for scrollbar */}
						{filteredTasks.length === 0 ? (
							<p className="text-gray-500 text-center py-10">
								No tasks found matching your criteria.
							</p>
						) : (
							filteredTasks.map((task) => {
								const StatusIcon =
									statusMap[task.status]?.icon ||
									statusMap.default.icon
								const statusColor =
									statusMap[task.status]?.color ||
									statusMap.default.color
								const priorityInfo =
									priorityMap[task.priority] ||
									priorityMap.default
								return (
									// MODIFIED: Task item using flexbox divs
									<div
										key={task.task_id}
										className="flex items-center gap-4 bg-neutral-800 p-3 rounded-lg shadow hover:bg-neutral-700/60 transition-colors duration-150"
									>
										{/* Status Icon & Priority */}
										<div className="flex flex-col items-center w-16 flex-shrink-0">
											<StatusIcon
												className={`h-6 w-6 ${statusColor}`}
											/>
											<span
												className={`text-xs mt-1 font-medium ${priorityInfo.color}`}
											>
												{priorityInfo.label}
											</span>
										</div>
										{/* Task Details */}
										<div className="flex-grow min-w-0">
											<p
												className="text-sm font-medium text-white truncate"
												title={task.description}
											>
												{task.status ===
												"approval_pending" ? (
													<button
														onClick={async () =>
															await handleViewApprovalData(
																task.task_id
															)
														}
														className="hover:underline text-blue-400"
													>
														{task.description}
													</button>
												) : (
													task.description
												)}
											</p>
											<p className="text-xs text-gray-400 mt-1">
												ID: {task.task_id} | Added:{" "}
												{task.created_at
													? new Date(
															task.created_at
														).toLocaleString()
													: "N/A"}
											</p>
											{task.result && (
												<p
													className="text-xs text-gray-500 mt-1 truncate"
													title={task.result}
												>
													Result: {task.result}
												</p>
											)}
											{task.error && (
												<p
													className="text-xs text-red-500 mt-1 truncate"
													title={task.error}
												>
													Error: {task.error}
												</p>
											)}
										</div>
										{/* Actions */}
										<div className="flex items-center gap-2 flex-shrink-0">
											<button
												onClick={() =>
													handleEditTask(task)
												}
												disabled={
													task.status === "processing"
												}
												className={`p-1.5 rounded-md transition-colors ${task.status === "processing" ? "text-gray-600 cursor-not-allowed" : "text-yellow-400 hover:bg-neutral-700"}`}
												title="Edit Task"
											>
												<IconPencil className="h-4 w-4" />
											</button>
											<button
												onClick={() =>
													handleDeleteTask(
														task.task_id
													)
												}
												className="p-1.5 rounded-md text-red-400 hover:bg-neutral-700 transition-colors"
												title="Delete Task"
											>
												<IconTrash className="h-4 w-4" />
											</button>
										</div>
									</div>
								)
							})
						)}
					</div>
				</div>

				{/* --- MODIFIED: Add Task Bar (Bottom) --- */}
				<div className="absolute bottom-0 left-0 right-0 z-30 p-4">
					<div className="max-w-5xl mx-auto flex items-center gap-2 bg-neutral-800/80 backdrop-blur-sm rounded-lg p-2 shadow-lg">
						{/* Text Area */}
						<textarea
							placeholder="Enter new task description..."
							value={newTaskDescription}
							onChange={(e) =>
								setNewTaskDescription(e.target.value)
							}
							rows={1} // Start with 1 row, auto-expands slightly
							className="flex-grow p-2 bg-transparent text-white text-sm focus:outline-none resize-none placeholder-gray-500 min-h-[40px] max-h-[100px]" // Added min/max height
						/>
						{/* Priority Dropdown */}
						<div className="relative flex-shrink-0">
							<select
								value={newTaskPriorityLevel}
								onChange={(e) =>
									setNewTaskPriorityLevel(
										Number(e.target.value)
									)
								}
								className="appearance-none bg-neutral-700 border border-neutral-600 text-white text-xs rounded-full px-3 py-1.5 focus:outline-none focus:border-lightblue cursor-pointer"
								title="Set task priority"
							>
								<option value={0}>High</option>
								<option value={1}>Medium</option>
								<option value={2}>Low</option>
							</select>
							<IconChevronUp className="absolute right-2 top-1/2 transform -translate-y-1/2 h-3 w-3 text-gray-400 pointer-events-none" />{" "}
							{/* Visual cue */}
						</div>
						{/* Add Button */}
						<button
							onClick={handleAddTask}
							className="p-2 rounded-full bg-lightblue text-white hover:bg-blue-700 focus:outline-none transition-colors flex-shrink-0"
							title="Add Task"
						>
							<IconSend className="h-5 w-5" />{" "}
							{/* Changed to Send icon */}
						</button>
					</div>
				</div>

				{/* --- Edit Task Modal --- */}
				{/* MODIFIED: Styling and Priority input */}
				{editingTask && (
					<div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex justify-center items-center z-50">
						<div className="bg-neutral-800 p-6 rounded-lg shadow-xl w-full max-w-md mx-4">
							<h3 className="text-lg font-semibold mb-4 text-white">
								Edit Task
							</h3>
							<div className="mb-4">
								<label
									htmlFor="edit-description"
									className="block text-gray-400 text-sm font-medium mb-1"
								>
									{" "}
									Description{" "}
								</label>
								<input
									type="text"
									id="edit-description"
									value={editingTask.description}
									onChange={(e) =>
										setEditingTask({
											...editingTask,
											description: e.target.value
										})
									}
									className="p-2 rounded-md bg-neutral-700 border border-neutral-600 text-white focus:outline-none w-full focus:border-lightblue"
								/>
							</div>
							<div className="mb-6">
								<label
									htmlFor="edit-priority"
									className="block text-gray-400 text-sm font-medium mb-1"
								>
									{" "}
									Priority{" "}
								</label>
								{/* MODIFIED: Use select dropdown for priority */}
								<select
									id="edit-priority"
									value={editingTask.priority} // Use the numeric priority
									onChange={(e) =>
										setEditingTask({
											...editingTask,
											priority: Number(e.target.value)
										})
									}
									className="p-2 rounded-md bg-neutral-700 border border-neutral-600 text-white focus:outline-none w-full focus:border-lightblue appearance-none"
								>
									<option value={0}>High</option>
									<option value={1}>Medium</option>
									<option value={2}>Low</option>
								</select>
							</div>
							<div className="flex justify-end gap-3">
								<button
									onClick={() => setEditingTask(null)}
									className="py-2 px-4 rounded bg-neutral-600 hover:bg-neutral-500 text-white text-sm font-medium transition-colors"
								>
									{" "}
									Cancel{" "}
								</button>
								<button
									onClick={handleUpdateTask}
									className="py-2 px-4 rounded bg-green-600 hover:bg-green-500 text-white text-sm font-medium transition-colors"
								>
									{" "}
									Save Changes{" "}
								</button>
							</div>
						</div>
					</div>
				)}

				{/* --- Approval Modal --- */}
				{/* MODIFIED: Styling */}
				{selectedTask && (
					<div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex justify-center items-center z-50">
						<div className="bg-neutral-800 p-6 rounded-lg shadow-xl w-full max-w-lg mx-4 text-gray-300">
							<h3 className="text-lg font-semibold mb-4 text-white">
								Approve Task Action
							</h3>
							<div className="space-y-2 text-sm mb-4">
								<p>
									<strong>Task ID:</strong>{" "}
									<span className="text-white font-mono">
										{selectedTask.taskId}
									</span>
								</p>
								<p>
									<strong>Tool:</strong>{" "}
									<span className="text-white">
										{selectedTask.approvalData?.tool_name ||
											"N/A"}
									</span>
								</p>
								{/* Display parameters dynamically */}
								{selectedTask.approvalData?.parameters &&
									Object.entries(
										selectedTask.approvalData.parameters
									).map(
										([key, value]) =>
											key !== "body" && ( // Don't display body here
												<p key={key}>
													<strong>
														{key
															.charAt(0)
															.toUpperCase() +
															key.slice(1)}
														:
													</strong>{" "}
													<span className="text-white">
														{value}
													</span>
												</p>
											)
									)}
								{selectedTask.approvalData?.parameters
									?.body && (
									<>
										<p className="mt-2">
											<strong>Body:</strong>
										</p>
										<textarea
											readOnly
											className="w-full h-32 p-2 mt-1 bg-neutral-700 border border-neutral-600 text-white rounded text-xs font-mono focus:outline-none"
											value={
												selectedTask.approvalData
													.parameters.body
											}
										/>
									</>
								)}
							</div>
							<div className="flex justify-end gap-3">
								<button
									onClick={() => setSelectedTask(null)}
									className="py-2 px-4 rounded bg-neutral-600 hover:bg-neutral-500 text-white text-sm font-medium transition-colors"
								>
									{" "}
									Cancel{" "}
								</button>
								<button
									onClick={() =>
										handleApproveTask(selectedTask.taskId)
									}
									className="py-2 px-4 rounded bg-green-600 hover:bg-green-500 text-white text-sm font-medium transition-colors"
								>
									{" "}
									Approve{" "}
								</button>
							</div>
						</div>
					</div>
				)}
			</div>{" "}
			{/* End Main Content Area */}
		</div> // End Page Container
	)
}

export default Tasks
