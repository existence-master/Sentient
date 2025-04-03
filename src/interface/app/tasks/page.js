"use client"

// ADDED: Import necessary icons for status etc.
import React, { useState, useEffect, useCallback } from "react"
import {
	IconLoader,
	IconPencil,
	IconTrash,
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
	IconPlus // Using Plus icon for add task button again
} from "@tabler/icons-react"
import Sidebar from "@components/Sidebar"
import toast from "react-hot-toast"
import { Tooltip } from "react-tooltip"
import "react-tooltip/dist/react-tooltip.css"
import { cn } from "@utils/cn"

// --- Task Status Mapping ---
const statusMap = {
	pending: {
		icon: IconClock,
		color: "text-yellow-500",
		borderColor: "border-yellow-500",
		label: "Pending"
	},
	processing: {
		icon: IconPlayerPlay,
		color: "text-blue-500",
		borderColor: "border-blue-500",
		label: "Processing"
	},
	completed: {
		icon: IconCircleCheck,
		color: "text-green-500",
		borderColor: "border-green-500",
		label: "Completed"
	},
	error: {
		icon: IconAlertCircle,
		color: "text-red-500",
		borderColor: "border-red-500",
		label: "Error"
	},
	approval_pending: {
		icon: IconMailQuestion,
		color: "text-purple-500",
		borderColor: "border-purple-500",
		label: "Approval Pending"
	},
	cancelled: {
		icon: IconX,
		color: "text-gray-500",
		borderColor: "border-gray-500",
		label: "Cancelled"
	},
	default: {
		icon: IconHelpCircle,
		color: "text-gray-400",
		borderColor: "border-gray-400",
		label: "Unknown"
	}
}

// --- Priority Mapping ---
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
	const [newTaskPriorityLevel, setNewTaskPriorityLevel] = useState(1)
	const [editingTask, setEditingTask] = useState(null)
	const [filterStatus, setFilterStatus] = useState("all")
	const [searchTerm, setSearchTerm] = useState("")
	const [selectedTask, setSelectedTask] = useState(null)

	// --- Fetching Data ---
	const fetchTasksData = useCallback(async () => {
		console.log("Fetching tasks data...")
		if (tasks.length === 0) {
			setLoading(true)
		}
		setError(null)
		try {
			const response = await window.electron.invoke("fetch-tasks")
			console.log("Raw tasks response:", response)
			if (response.error) {
				console.error("Error fetching tasks:", response.error)
				setError(response.error)
				setTasks([])
			} else if (Array.isArray(response.tasks)) {
				// Sort tasks
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
					if (statusA !== statusB) return statusA - statusB
					if (a.priority !== b.priority)
						return a.priority - b.priority
					try {
						const dateA = a.created_at
							? new Date(a.created_at).getTime()
							: 0
						const dateB = b.created_at
							? new Date(b.created_at).getTime()
							: 0
						return dateB - dateA
					} catch (dateError) {
						console.warn("Error comparing task dates:", dateError)
						return 0
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
			setLoading(false)
		}
	}, [tasks.length])

	const fetchUserDetails = async () => {
		try {
			const response = await window.electron?.invoke("get-profile")
			setUserDetails(response || {})
		} catch (error) {
			toast.error("Error fetching user details for sidebar.")
			console.error("Error fetching user details for sidebar:", error)
		}
	}

	// --- Effects ---
	useEffect(() => {
		fetchUserDetails()
		fetchTasksData()
		const intervalId = setInterval(fetchTasksData, 60000)
		return () => clearInterval(intervalId)
	}, [fetchTasksData])

	// --- Task Actions ---
	const handleAddTask = async () => {
		if (!newTaskDescription.trim()) {
			toast.error("Please enter a task description.")
			return
		}
		console.log("Adding task:", {
			description: newTaskDescription,
			priority: newTaskPriorityLevel
		})
		try {
			const taskData = {
				description: newTaskDescription,
				priority: newTaskPriorityLevel
			}
			const response = await window.electron.invoke("add-task", taskData)
			if (response.error) {
				console.error("Error adding task via IPC:", response.error)
				toast.error(`Failed to add task: ${response.error}`)
			} else {
				toast.success("Task added successfully!")
				setNewTaskDescription("")
				setNewTaskPriorityLevel(1)
				await fetchTasksData()
			}
		} catch (error) {
			console.error("Exception adding task:", error)
			toast.error("Failed to add task: An unexpected error occurred.")
		}
	}

	const handleEditTask = (task) => {
		console.log("Editing task:", task)
		setEditingTask({ ...task })
	}

	const handleUpdateTask = async () => {
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
			const response = await window.electron.invoke("update-task", {
				taskId: editingTask.task_id,
				description: editingTask.description,
				priority: editingTask.priority
			})
			if (response.error) {
				console.error("Error updating task via IPC:", response.error)
				toast.error(`Failed to update task: ${response.error}`)
			} else {
				toast.success("Task updated successfully!")
				setEditingTask(null)
				await fetchTasksData()
			}
		} catch (error) {
			console.error("Exception updating task:", error)
			toast.error("Failed to update task: An unexpected error occurred.")
		}
	}

	const handleDeleteTask = async (taskId) => {
		if (!taskId) return
		console.log("Deleting task:", taskId)
		try {
			const response = await window.electron.invoke("delete-task", taskId)
			if (response.error) {
				console.error("Error deleting task via IPC:", response.error)
				toast.error(`Failed to delete task: ${response.error}`)
			} else {
				toast.success("Task deleted successfully!")
				await fetchTasksData()
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
				setSelectedTask(null)
				await fetchTasksData()
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
		if (filterStatus !== "all" && task.status !== filterStatus) return false
		if (
			searchTerm &&
			!task.description?.toLowerCase().includes(searchTerm.toLowerCase())
		)
			return false
		return true
	})

	// --- Render Loading/Error States ---
	if (loading && tasks.length === 0) {
		// Show loader only on initial load
		return (
			<div className="flex justify-center items-center h-screen bg-matteblack">
				{" "}
				<IconLoader className="w-10 h-10 animate-spin text-white" />{" "}
				<span className="ml-2 text-white">Loading tasks...</span>{" "}
			</div>
		)
	}
	if (error && tasks.length === 0) {
		// Show error only if there are no tasks to display
		return (
			<div className="flex flex-col justify-center items-center h-screen bg-matteblack text-red-500">
				{" "}
				<p>Error loading tasks: {error}</p>{" "}
				<button
					onClick={fetchTasksData}
					className="mt-4 py-2 px-4 bg-lightblue text-white rounded hover:bg-blue-700"
				>
					Retry
				</button>{" "}
			</div>
		)
	}

	console.log(filteredTasks)

	// --- Main Render ---
	return (
		<div className="h-screen bg-matteblack flex relative overflow-hidden dark">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
			/>
			<div className="flex-grow flex flex-col h-full bg-matteblack text-white relative overflow-hidden p-6">
				{/* --- Top Bar for Search/Filter --- */}
				<div className="absolute top-6 left-1/2 transform -translate-x-1/2 z-30 w-full max-w-3xl px-4">
					<div className="flex items-center space-x-3 bg-neutral-800/80 backdrop-blur-sm rounded-full p-3 shadow-lg border border-neutral-700">
						<IconSearch className="h-6 w-6 text-gray-400 ml-2 flex-shrink-0" />
						<input
							type="text"
							placeholder="Search tasks..."
							value={searchTerm}
							onChange={(e) => setSearchTerm(e.target.value)}
							className="bg-transparent text-white focus:outline-none w-full flex-grow px-2 placeholder-gray-500 text-base rounded-md py-1" // Removed redundant border
						/>
						<div className="relative flex-shrink-0">
							<IconFilter className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" />
							<select
								value={filterStatus}
								onChange={(e) =>
									setFilterStatus(e.target.value)
								}
								className="appearance-none bg-neutral-700 border border-neutral-600 text-white text-sm rounded-full pl-9 pr-4 py-2 focus:outline-none focus:border-lightblue cursor-pointer"
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

				{/* --- Refresh Button (Top Right) --- */}
				<div className="absolute top-6 right-6 z-30">
					<button
						onClick={fetchTasksData}
						className="p-2.5 rounded-full hover:bg-neutral-700/60 transition-colors text-gray-300"
						data-tooltip-id="refresh-tooltip"
						disabled={loading && tasks.length > 0}
					>
						{loading && tasks.length > 0 ? (
							<IconLoader className="h-6 w-6 animate-spin" />
						) : (
							<IconRefresh className="h-6 w-6" />
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

				{/* --- Task List Container --- */}
				<div className="flex-grow w-full max-w-4xl mx-auto px-0 pt-24 pb-28 flex flex-col overflow-hidden">
					<div className="flex-grow overflow-y-auto space-y-4 pr-2 custom-scrollbar">
						{filteredTasks.length === 0 && !loading ? (
							<p className="text-gray-500 text-center py-16">
								No tasks found matching your criteria.
							</p>
						) : (
							filteredTasks.map((task) => {
								const statusInfo =
									statusMap[task.status] || statusMap.default
								const priorityInfo =
									priorityMap[task.priority] ||
									priorityMap.default
								return (
									<div
										key={task.task_id}
										className={cn(
											"flex items-center gap-4 bg-neutral-800 p-4 rounded-lg shadow hover:bg-neutral-700/60 transition-colors duration-150",
											"border-l-4",
											statusInfo.borderColor
										)}
									>
										{/* Status Icon & Priority */}
										<div className="flex flex-col items-center w-20 flex-shrink-0">
											<statusInfo.icon
												className={cn(
													"h-7 w-7",
													statusInfo.color
												)}
											/>
											<span
												className={cn(
													"text-xs mt-1.5 font-semibold",
													priorityInfo.color
												)}
											>
												{priorityInfo.label}
											</span>
										</div>
										{/* Task Details */}
										<div className="flex-grow min-w-0">
											<p
												className="text-base font-medium text-white truncate"
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
														{" "}
														{task.description}{" "}
													</button>
												) : (
													task.description
												)}
											</p>
											<p className="text-sm text-gray-400 mt-1">
												{" "}
												ID: {task.task_id} | Added:{" "}
												{task.created_at
													? new Date(
															task.created_at
														).toLocaleString()
													: "N/A"}{" "}
											</p>
											{/* --- MODIFIED: Safely render result/error --- */}
											{task.result != null &&
												task.result !== "" && (
													<p
														className="text-xs text-gray-500 mt-1 truncate"
														title={
															typeof task.result ===
															"string"
																? task.result
																: JSON.stringify(
																		task.result
																	)
														}
													>
														Result:{" "}
														{typeof task.result ===
														"string"
															? task.result
															: "[Details in Modal/Log]"}
													</p>
												)}
											{task.error != null &&
												task.error !== "" && (
													<p
														className="text-xs text-red-500 mt-1 truncate"
														title={
															typeof task.error ===
															"string"
																? task.error
																: JSON.stringify(
																		task.error
																	)
														}
													>
														Error:{" "}
														{typeof task.error ===
														"string"
															? task.error
															: "[Error Details]"}
													</p>
												)}
											{/* --- End Fix --- */}
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
												className={cn(
													"p-2 rounded-md transition-colors",
													task.status === "processing"
														? "text-gray-600 cursor-not-allowed"
														: "text-yellow-400 hover:bg-neutral-700"
												)}
												title="Edit Task"
											>
												{" "}
												<IconPencil className="h-5 w-5" />{" "}
											</button>
											<button
												onClick={() =>
													handleDeleteTask(
														task.task_id
													)
												}
												className="p-2 rounded-md text-red-400 hover:bg-neutral-700 transition-colors"
												title="Delete Task"
											>
												{" "}
												<IconTrash className="h-5 w-5" />{" "}
											</button>
										</div>
									</div>
								)
							})
						)}
					</div>
				</div>

				{/* --- Add Task Bar (Bottom) --- */}
				<div className="absolute bottom-0 left-0 right-0 z-30 p-5">
					<div className="max-w-5xl mx-auto flex items-center gap-3 border border-neutral-600 bg-neutral-800/80 backdrop-blur-sm rounded-full p-3 shadow-lg">
						{/* Text Area */}
						<textarea
							placeholder="Enter new task description..."
							value={newTaskDescription}
							onChange={(e) =>
								setNewTaskDescription(e.target.value)
							}
							rows={1}
							className="flex-grow p-2.5 bg-transparent text-white text-base focus:outline-none resize-none placeholder-gray-500 min-h-[48px] max-h-[120px] rounded-md" // Removed border here as parent has one
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
								className="appearance-none bg-neutral-700 border border-neutral-600 text-white text-sm rounded-full px-4 py-2.5 focus:outline-none focus:border-lightblue cursor-pointer"
								title="Set task priority"
							>
								<option value={0}>High</option>
								<option value={1}>Medium</option>
								<option value={2}>Low</option>
							</select>
							<IconChevronUp className="absolute right-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" />
						</div>
						{/* Add Button */}
						<button
							onClick={handleAddTask}
							className="p-3 rounded-full bg-lightblue text-white hover:bg-blue-700 focus:outline-none transition-colors flex-shrink-0"
							title="Add Task"
						>
							<IconPlus className="h-6 w-6" />
						</button>
					</div>
				</div>

				{/* --- Edit Task Modal --- */}
				{editingTask && (
					<div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex justify-center items-center z-50 p-4">
						<div className="bg-neutral-800 p-8 rounded-lg shadow-xl w-full max-w-lg mx-auto">
							<h3 className="text-xl font-semibold mb-6 text-white">
								Edit Task
							</h3>
							<div className="mb-5">
								<label
									htmlFor="edit-description"
									className="block text-gray-300 text-sm font-medium mb-2"
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
									className="p-3 rounded-md bg-neutral-700 border border-neutral-600 text-white focus:outline-none w-full focus:border-lightblue text-base"
								/>
							</div>
							<div className="mb-8">
								<label
									htmlFor="edit-priority"
									className="block text-gray-300 text-sm font-medium mb-2"
								>
									{" "}
									Priority{" "}
								</label>
								<select
									id="edit-priority"
									value={editingTask.priority}
									onChange={(e) =>
										setEditingTask({
											...editingTask,
											priority: Number(e.target.value)
										})
									}
									className="p-3 rounded-md bg-neutral-700 border border-neutral-600 text-white focus:outline-none w-full focus:border-lightblue appearance-none text-base"
								>
									<option value={0}>High</option>
									<option value={1}>Medium</option>
									<option value={2}>Low</option>
								</select>
							</div>
							<div className="flex justify-end gap-4">
								<button
									onClick={() => setEditingTask(null)}
									className="py-2.5 px-5 rounded bg-neutral-600 hover:bg-neutral-500 text-white text-sm font-medium transition-colors"
								>
									{" "}
									Cancel{" "}
								</button>
								<button
									onClick={handleUpdateTask}
									className="py-2.5 px-5 rounded bg-green-600 hover:bg-green-500 text-white text-sm font-medium transition-colors"
								>
									{" "}
									Save Changes{" "}
								</button>
							</div>
						</div>
					</div>
				)}

				{/* --- Approval Modal --- */}
				{selectedTask && (
					<div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex justify-center items-center z-50 p-4">
						<div className="bg-neutral-800 p-8 rounded-lg shadow-xl w-full max-w-xl mx-auto text-gray-300">
							<h3 className="text-xl font-semibold mb-6 text-white">
								Approve Task Action
							</h3>
							<div className="space-y-3 text-base mb-6">
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
								{selectedTask.approvalData?.parameters &&
									Object.entries(
										selectedTask.approvalData.parameters
									).map(
										([key, value]) =>
											key !== "body" && (
												<p key={key}>
													<strong>
														{key
															.charAt(0)
															.toUpperCase() +
															key.slice(1)}
														:
													</strong>{" "}
													<span className="text-white">
														{String(value)}
													</span>
												</p>
											)
									)}
								{selectedTask.approvalData?.parameters
									?.body && (
									<>
										<p className="mt-3">
											<strong>Body:</strong>
										</p>
										<textarea
											readOnly
											className="w-full h-40 p-3 mt-1 bg-neutral-700 border border-neutral-600 text-white rounded text-sm font-mono focus:outline-none"
											value={
												selectedTask.approvalData
													.parameters.body
											}
										/>
									</>
								)}
							</div>
							<div className="flex justify-end gap-4">
								<button
									onClick={() => setSelectedTask(null)}
									className="py-2.5 px-5 rounded bg-neutral-600 hover:bg-neutral-500 text-white text-sm font-medium transition-colors"
								>
									{" "}
									Cancel{" "}
								</button>
								<button
									onClick={() =>
										handleApproveTask(selectedTask.taskId)
									}
									className="py-2.5 px-5 rounded bg-green-600 hover:bg-green-500 text-white text-sm font-medium transition-colors"
								>
									{" "}
									Approve{" "}
								</button>
							</div>
						</div>
					</div>
				)}

				{/* --- Status Legend --- */}
				<div className="absolute bottom-24 right-6 z-30 bg-neutral-800/80 backdrop-blur-sm rounded-lg p-3 shadow-lg text-xs">
					<h4 className="font-semibold text-gray-300 mb-2 text-center">
						Status Legend
					</h4>
					<div className="space-y-1.5">
						{Object.entries(statusMap).map(
							([key, { label, color, borderColor }]) => {
								if (key === "default") return null
								const borderClass =
									borderColor || "border-transparent"
								const textClass = color || "text-gray-400"
								return (
									<div
										key={key}
										className="flex items-center gap-2"
									>
										<span
											className={cn(
												"block w-3 h-3 rounded-sm border-l-4",
												borderClass
											)}
										></span>
										<span
											className={cn(
												"capitalize",
												textClass
											)}
										>
											{label}
										</span>
									</div>
								)
							}
						)}
					</div>
				</div>
			</div>{" "}
			{/* End Main Content Area */}
		</div> // End Page Container
	)
}

export default Tasks
