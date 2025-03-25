"use client"

import React, { useState, useEffect } from "react"
import { IconLoader } from "@tabler/icons-react"
import Sidebar from "@components/Sidebar"
import toast from "react-hot-toast"

const Tasks = () => {
	const [tasks, setTasks] = useState([])
	const [loading, setLoading] = useState(true)
	const [error, setError] = useState(null)
	const [userDetails, setUserDetails] = useState({})
	const [isSidebarVisible, setSidebarVisible] = useState(false)
	const [newTaskDescription, setNewTaskDescription] = useState("")
	const [newTaskPriority, setNewTaskPriority] = useState("")
	const [editingTask, setEditingTask] = useState(null)

	useEffect(() => {
		fetchTasksData()
		const intervalId = setInterval(fetchTasksData, 300000)
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

	const handleAddTask = async () => {
		if (!newTaskDescription || !newTaskPriority) {
			toast.error("Please fill in all fields")
			return
		}
		try {
			const taskData = {
				description: newTaskDescription,
				priority: parseInt(newTaskPriority),
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
		setEditingTask({ ...task })
	}

	const handleUpdateTask = async () => {
		if (!editingTask.description || !editingTask.priority) {
			toast.error("Please fill in all fields")
			return
		}
		try {
			const response = await window.electron.invoke("update-task", {
				taskId: editingTask.task_id,
				description: editingTask.description,
				priority: parseInt(editingTask.priority)
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
			<div className="w-4/5 flex flex-col justify-center items-start h-full bg-matteblack ml-5">
				<div className="w-full p-4">
					<h2 className="text-2xl font-bold mb-4 text-white">
						Task List
					</h2>
					<div className="mb-4">
						<input
							type="text"
							placeholder="Task Description"
							value={newTaskDescription}
							onChange={(e) =>
								setNewTaskDescription(e.target.value)
							}
							className="p-2 rounded bg-gray-800 text-white mr-2"
						/>
						<input
							type="number"
							placeholder="Priority"
							value={newTaskPriority}
							onChange={(e) => setNewTaskPriority(e.target.value)}
							className="p-2 rounded bg-gray-800 text-white mr-2"
						/>
						<button
							onClick={handleAddTask}
							className="p-2 rounded bg-blue-500 text-white"
						>
							Add Task
						</button>
					</div>
					{tasks.length === 0 ? (
						<p className="text-white">No tasks available.</p>
					) : (
						<div className="w-full min-h-fit overflow-x-auto rounded-xl border border-gray-400">
							<table className="min-w-full bg-matteblack text-white">
								<thead>
									<tr>
										<th className="py-2 px-4 border-b">
											Description
										</th>
										<th className="py-2 px-4 border-b">
											Timestamp
										</th>
										<th className="py-2 px-4 border-b">
											Priority
										</th>
										<th className="py-2 px-4 border-b">
											Status
										</th>
										<th className="py-2 px-4 border-b">
											Result
										</th>
										<th className="py-2 px-4 border-b">
											Error
										</th>
										<th className="py-2 px-4 border-b">
											Actions
										</th>
									</tr>
								</thead>
								<tbody>
									{tasks.map((task) => (
										<tr
											key={task.task_id}
											className="hover:bg-gray-800"
										>
											<td className="py-2 px-4 border-b">
												{task.description}
											</td>
											<td className="py-2 px-4 border-b">
												{task.timestamp}
											</td>
											<td className="py-2 px-4 border-b text-center">
												{task.priority}
											</td>
											<td className="py-2 px-4 border-b">
												{task.status}
											</td>
											<td className="py-2 px-4 border-b">
												{task.result || "N/A"}
											</td>
											<td className="py-2 px-4 border-b">
												{task.error || "N/A"}
											</td>
											<td className="py-2 px-4 border-b">
												<button
													onClick={() =>
														handleEditTask(task)
													}
													disabled={
														task.status ===
														"in progress"
													}
													className={`p-1 rounded mr-2 ${
														task.status ===
														"in progress"
															? "bg-gray-500 cursor-not-allowed"
															: "bg-yellow-500"
													} text-white`}
												>
													Edit
												</button>
												<button
													onClick={() =>
														handleDeleteTask(
															task.task_id
														)
													}
													className="p-1 rounded bg-red-500 text-white"
												>
													Delete
												</button>
											</td>
										</tr>
									))}
								</tbody>
							</table>
						</div>
					)}
					{editingTask && (
						<div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center">
							<div className="bg-gray-800 p-4 rounded text-white">
								<h3 className="text-lg mb-2">Edit Task</h3>
								<input
									type="text"
									value={editingTask.description}
									onChange={(e) =>
										setEditingTask({
											...editingTask,
											description: e.target.value
										})
									}
									className="p-2 rounded bg-gray-700 text-white w-full mb-2"
								/>
								<input
									type="number"
									value={editingTask.priority}
									onChange={(e) =>
										setEditingTask({
											...editingTask,
											priority: e.target.value
										})
									}
									className="p-2 rounded bg-gray-700 text-white w-full mb-2"
								/>
								<button
									onClick={handleUpdateTask}
									className="p-2 rounded bg-green-500 text-white mr-2"
								>
									Save
								</button>
								<button
									onClick={() => setEditingTask(null)}
									className="p-2 rounded bg-gray-500 text-white"
								>
									Cancel
								</button>
							</div>
						</div>
					)}
				</div>
			</div>
		</div>
	)
}

export default Tasks
