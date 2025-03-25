"use client"

import React, { useState, useEffect } from "react"
import {
	IconLoader,
	IconPencil,
	IconTrash,
	IconPlus,
	IconSearch,
	IconRefresh
} from "@tabler/icons-react"
import Sidebar from "@components/Sidebar"
import toast from "react-hot-toast"
import { Tooltip } from "react-tooltip"
import "react-tooltip/dist/react-tooltip.css"

const Tasks = () => {
	const [tasks, setTasks] = useState([])
	const [loading, setLoading] = useState(true)
	const [error, setError] = useState(null)
	const [userDetails, setUserDetails] = useState({})
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	const [newTaskDescription, setNewTaskDescription] = useState("")
	const [newTaskPriority, setNewTaskPriority] = useState("") // String based priority now
	const [editingTask, setEditingTask] = useState(null)
	const [filterStatus, setFilterStatus] = useState("all")
	const [searchTerm, setSearchTerm] = useState("")

	useEffect(() => {
		fetchTasksData()
		const intervalId = setInterval(fetchTasksData, 60000)
		fetchUserDetails()
		return () => clearInterval(intervalId)
	}, [])

	const fetchTasksData = async () => {
		setLoading(true)
		setError(null)
		try {
			const response = await window.electron.invoke("fetch-tasks")
			if (response.error) {
				setError(response.error)
			} else if (response.tasks) {
				setTasks(response.tasks)
			} else {
				setError("Failed to fetch tasks: Invalid response format")
			}
		} catch (err) {
			setError("Failed to fetch tasks: " + err.message)
		} finally {
			setLoading(false)
		}
	}

	const fetchUserDetails = async () => {
		try {
			const response = await window.electron?.invoke("get-profile")
			setUserDetails(response)
		} catch (error) {
			toast.error("Error fetching user details for sidebar.")
			console.error("Error fetching user details for sidebar:", error)
		}
	}

	const priorityStringToNumber = (priorityString) => {
		const priorityMap = { High: 0, Medium: 1, Low: 2 }
		return priorityMap[priorityString] !== undefined
			? priorityMap[priorityString]
			: null
	}

	const priorityNumberToString = (priorityNumber) => {
		const priorityMap = { 0: "High", 1: "Medium", 2: "Low" }
		return priorityMap[priorityNumber] || "Unknown"
	}

	const handleAddTask = async () => {
		if (!newTaskDescription || !newTaskPriority) {
			toast.error("Please fill in all fields")
			return
		}

		const priorityNumber = priorityStringToNumber(newTaskPriority)
		if (priorityNumber === null) {
			toast.error(
				"Invalid priority value. Please use 'High', 'Medium', or 'Low'."
			)
			return
		}

		try {
			const taskData = {
				description: newTaskDescription,
				priority: priorityNumber // Send number to backend
			}
			const response = await window.electron.invoke("add-task", taskData)
			if (response.error) {
				toast.error(response.error)
			} else {
				toast.success("Task added successfully")
				setNewTaskDescription("")
				setNewTaskPriority("")
				fetchTasksData()
			}
		} catch (error) {
			toast.error("Failed to add task")
		}
	}

	const handleEditTask = (task) => {
		setEditingTask({
			...task,
			priority: priorityNumberToString(task.priority) // Convert to string for editing
		})
	}

	const handleUpdateTask = async () => {
		if (!editingTask.description || !editingTask.priority) {
			toast.error("Please fill in all fields")
			return
		}

		const priorityNumber = priorityStringToNumber(editingTask.priority)
		if (priorityNumber === null) {
			toast.error(
				"Invalid priority value. Please use 'High', 'Medium', or 'Low'."
			)
			return
		}

		try {
			const response = await window.electron.invoke("update-task", {
				taskId: editingTask.task_id,
				description: editingTask.description,
				priority: priorityNumber // Send number to backend
			})
			if (response.error) {
				toast.error(response.error)
			} else {
				toast.success("Task updated successfully")
				setEditingTask(null)
				fetchTasksData()
			}
		} catch (error) {
			toast.error("Failed to update task")
		}
	}

	const handleDeleteTask = async (taskId) => {
		try {
			const response = await window.electron.invoke("delete-task", taskId)
			if (response.error) {
				toast.error(response.error)
			} else {
				toast.success("Task deleted successfully")
				fetchTasksData()
			}
		} catch (error) {
			toast.error("Failed to delete task")
		}
	}

	const filteredTasks = tasks
		.filter((task) => {
			if (filterStatus === "all") return true
			return task.status === filterStatus
		})
		.filter((task) => {
			if (!searchTerm) return true
			return task.description
				.toLowerCase()
				.includes(searchTerm.toLowerCase())
		})

	if (loading) {
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
			</div>
		)
	}

	return (
		<div className="h-screen bg-matteblack flex relative">
			<Sidebar
				userDetails={userDetails}
				isSidebarVisible={isSidebarVisible}
				setSidebarVisible={setSidebarVisible}
				fromChat={false}
			/>
			<div className="w-4/5 flex flex-col items-start h-full bg-matteblack ml-5 py-8">
				<div className="w-full max-w-5xl p-6 bg-gray-900 rounded-lg shadow-xl text-white flex flex-col gap-6">
					<div className="flex justify-between items-center">
						<h2 className="text-3xl font-bold text-center">
							Task Management
						</h2>
						<div>
							<button
								onClick={fetchTasksData}
								className="p-2 rounded-md hover:bg-gray-800 transition-colors text-gray-300"
								data-tooltip-id="refresh-tooltip"
							>
								<IconRefresh className="h-5 w-5" />
							</button>
							<Tooltip
								id="refresh-tooltip"
								content="Auto refreshes every minute"
								place="bottom"
							/>
						</div>
					</div>

					{/* Search and Filter Bar */}
					<div className="flex flex-col md:flex-row items-center justify-between gap-4">
						<div className="flex bg-gray-800 rounded-md p-2 w-full md:w-auto flex-grow">
							<IconSearch className="h-5 w-5 text-gray-500 mr-2" />
							<input
								type="text"
								placeholder="Search tasks..."
								value={searchTerm}
								onChange={(e) => setSearchTerm(e.target.value)}
								className="bg-transparent text-white focus:outline-none w-full"
							/>
						</div>
						<div className="flex space-x-2">
							<button
								onClick={() => setFilterStatus("all")}
								className={`rounded-md px-3 py-2 text-sm font-medium ${
									filterStatus === "all"
										? "bg-lightblue text-white"
										: "bg-gray-800 hover:bg-gray-700 text-gray-300"
								}`}
							>
								All
							</button>
							<button
								onClick={() => setFilterStatus("to do")}
								className={`rounded-md px-3 py-2 text-sm font-medium ${
									filterStatus === "to do"
										? "bg-lightblue text-white"
										: "bg-gray-800 hover:bg-gray-700 text-gray-300"
								}`}
							>
								To Do
							</button>
							<button
								onClick={() => setFilterStatus("in progress")}
								className={`rounded-md px-3 py-2 text-sm font-medium ${
									filterStatus === "in progress"
										? "bg-lightblue text-white"
										: "bg-gray-800 hover:bg-gray-700 text-gray-300"
								}`}
							>
								In Progress
							</button>
							<button
								onClick={() => setFilterStatus("done")}
								className={`rounded-md px-3 py-2 text-sm font-medium ${
									filterStatus === "done"
										? "bg-lightblue text-white"
										: "bg-gray-800 hover:bg-gray-700 text-gray-300"
								}`}
							>
								Done
							</button>
							<button
								onClick={() => setFilterStatus("error")}
								className={`rounded-md px-3 py-2 text-sm font-medium ${
									filterStatus === "error"
										? "bg-lightblue text-white"
										: "bg-gray-800 hover:bg-gray-700 text-gray-300"
								}`}
							>
								Error
							</button>
						</div>
					</div>

					{/* Add Task Section */}
					<div className="flex flex-col md:flex-row items-start md:items-center gap-4">
						<div className="flex flex-col flex-grow">
							<input
								type="text"
								placeholder="Task Description"
								value={newTaskDescription}
								onChange={(e) =>
									setNewTaskDescription(e.target.value)
								}
								className="p-3 rounded-md bg-gray-800 text-white focus:outline-none w-full"
							/>
						</div>
						<div className="flex flex-col md:flex-row items-center gap-2">
							<input
								type="text" // Changed to text input
								placeholder="Priority (High, Medium, Low)"
								value={newTaskPriority}
								onChange={(e) =>
									setNewTaskPriority(e.target.value)
								}
								className="p-3 rounded-md bg-gray-800 text-white focus:outline-none w-24 md:w-auto"
							/>
							<button
								onClick={handleAddTask}
								className="flex items-center justify-center p-3 rounded-md bg-lightblue text-white hover:bg-blue-700 focus:outline-none"
							>
								<IconPlus className="h-4 w-4 mr-2" /> Add Task
							</button>
						</div>
					</div>

					{/* Task Table */}
					{filteredTasks.length === 0 ? (
						<p className="text-gray-400 text-center py-8">
							No tasks available for the current filter.
						</p>
					) : (
						<div
							className="overflow-x-auto"
							style={{ maxHeight: "70vh", overflowY: "auto" }}
						>
							<table className="min-w-full table-auto border-collapse rounded-lg overflow-hidden">
								<thead className="bg-gray-800 text-gray-300">
									<tr>
										<th className="py-3.5 px-4 font-semibold text-left">
											Description
										</th>
										<th className="py-3.5 px-4 font-semibold text-left">
											Timestamp
										</th>
										<th className="py-3.5 px-4 font-semibold text-center">
											Priority
										</th>
										<th className="py-3.5 px-4 font-semibold text-left">
											Status
										</th>
										<th className="py-3.5 px-4 font-semibold text-left">
											Result
										</th>
										<th className="py-3.5 px-4 font-semibold text-left">
											Error
										</th>
										<th className="py-3.5 px-4 font-semibold text-center">
											Actions
										</th>
									</tr>
								</thead>
								<tbody className="bg-gray-700 text-gray-100 divide-y divide-gray-800">
									{filteredTasks.map((task) => (
										<tr
											key={task.task_id}
											className="hover:bg-gray-600 transition-colors"
										>
											<td className="py-3 px-4">
												{task.description}
											</td>
											<td className="py-3 px-4">
												{task.created_at}
											</td>
											<td className="py-3 px-4 text-center">
												{priorityNumberToString(
													task.priority
												)}{" "}
												{/* Display string priority */}
											</td>
											<td className="py-3 px-4">
												{task.status}
											</td>
											<td className="py-3 px-4">
												{task.result || "N/A"}
											</td>
											<td className="py-3 px-4">
												{task.error || "N/A"}
											</td>
											<td className="py-3 px-4 text-center whitespace-nowrap">
												<button
													onClick={() =>
														handleEditTask(task)
													}
													disabled={
														task.status ===
														"in progress"
													}
													className={`p-2 rounded-md hover:bg-yellow-600 transition-colors ${
														task.status ===
														"in progress"
															? "text-gray-400 cursor-not-allowed"
															: "text-yellow-300"
													}`}
												>
													<IconPencil className="h-4 w-4" />
												</button>
												<button
													onClick={() =>
														handleDeleteTask(
															task.task_id
														)
													}
													className="p-2 rounded-md hover:bg-red-600 transition-colors text-red-300"
												>
													<IconTrash className="h-4 w-4" />
												</button>
											</td>
										</tr>
									))}
								</tbody>
							</table>
						</div>
					)}

					{/* Edit Task Modal */}
					{editingTask && (
						<div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center">
							<div className="bg-gray-800 p-8 rounded-lg shadow-lg">
								<h3 className="text-xl font-bold mb-4">
									Edit Task
								</h3>
								<div className="mb-4">
									<label
										htmlFor="edit-description"
										className="block text-gray-200 text-sm font-bold mb-2"
									>
										Description
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
										className="shadow appearance-none border rounded w-full p-3 bg-gray-700 text-white leading-tight focus:outline-none focus:shadow-outline"
									/>
								</div>
								<div className="mb-6">
									<label
										htmlFor="edit-priority"
										className="block text-gray-200 text-sm font-bold mb-2"
									>
										Priority (High, Medium, Low)
									</label>
									<input
										type="text" // Changed to text input
										id="edit-priority"
										value={editingTask.priority}
										onChange={(e) =>
											setEditingTask({
												...editingTask,
												priority: e.target.value
											})
										}
										className="shadow appearance-none border rounded w-full p-3 bg-gray-700 text-white leading-tight focus:outline-none focus:shadow-outline"
									/>
								</div>
								<div className="flex justify-end">
									<button
										onClick={handleUpdateTask}
										className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline mr-2"
									>
										Save
									</button>
									<button
										onClick={() => setEditingTask(null)}
										className="bg-gray-500 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
									>
										Cancel
									</button>
								</div>
							</div>
						</div>
					)}
				</div>
			</div>
		</div>
	)
}

export default Tasks