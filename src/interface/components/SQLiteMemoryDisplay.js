import React, { useEffect, useState } from "react"
import toast from "react-hot-toast"
import {
	IconPencil,
	IconTrash,
	IconPlus,
	IconRefresh
} from "@tabler/icons-react"

const SQLiteMemoryDisplay = ({ userDetails }) => {
	const [memories, setMemories] = useState([])
	const [selectedCategory, setSelectedCategory] = useState("PERSONAL")
	const [loading, setLoading] = useState(true)
	const [error, setError] = useState(null)
	const [newMemoryText, setNewMemoryText] = useState("")
	const [newMemoryCategory, setNewMemoryCategory] = useState("PERSONAL")
	const [newMemoryRetentionDays, setNewMemoryRetentionDays] = useState(7)
	const [editingMemory, setEditingMemory] = useState(null)
	const [updatedText, setUpdatedText] = useState("")
	const [updatedRetentionDays, setUpdatedRetentionDays] = useState(7)

	const categories = [
		"PERSONAL",
		"WORK",
		"SOCIAL",
		"RELATIONSHIP",
		"FINANCE",
		"SPIRITUAL",
		"CAREER",
		"TECHNOLOGY",
		"HEALTH",
		"EDUCATION",
		"TRANSPORTATION",
		"ENTERTAINMENT",
		"TASKS"
	]

	const fetchMemories = async () => {
		setLoading(true)
		setError(null)
		try {
			const memories = await window.electron?.invoke(
				"fetch-short-term-memories",
				{ category: selectedCategory.toLowerCase() }
			)
			if (memories.error) {
				setError(memories.error)
				setMemories([])
			} else {
				setMemories(memories || [])
			}
		} catch (error) {
			setError(`Error fetching memories: ${error.message}`)
			setMemories([])
		} finally {
			setLoading(false)
		}
	}

	useEffect(() => {
		fetchMemories()
	}, [selectedCategory, userDetails])

	const handleAddMemory = async () => {
		if (!newMemoryText || !newMemoryCategory || !newMemoryRetentionDays) {
			toast.error("Please fill in all fields")
			return
		}
		const retentionDays = parseInt(newMemoryRetentionDays)
		if (isNaN(retentionDays) || retentionDays < 1 || retentionDays > 90) {
			toast.error("Retention days must be between 1 and 90")
			return
		}

		try {
			const memoryData = {
				text: newMemoryText,
				category: newMemoryCategory.toLowerCase(),
				retention_days: retentionDays
			}
			const response = await window.electron.invoke(
				"add-memory",
				memoryData
			)
			if (response.error) {
				toast.error(response.error)
			} else {
				toast.success("Memory added successfully")
				setNewMemoryText("")
				setNewMemoryCategory("PERSONAL")
				setNewMemoryRetentionDays(7)
				fetchMemories()
			}
		} catch (error) {
			toast.error("Failed to add memory")
		}
	}

	const handleEditMemory = (memory) => {
		setEditingMemory(memory)
		setUpdatedText(memory.original_text)
		setUpdatedRetentionDays(7) // Default to 7, user sets new value
	}

	const handleUpdateMemory = async () => {
		if (!updatedText || !updatedRetentionDays) {
			toast.error("Please fill in all fields")
			return
		}
		const retentionDays = parseInt(updatedRetentionDays)
		if (isNaN(retentionDays) || retentionDays < 1 || retentionDays > 90) {
			toast.error("Retention days must be between 1 and 90")
			return
		}

		try {
			const memoryData = {
				category: editingMemory.category,
				id: editingMemory.id,
				text: updatedText,
				retention_days: retentionDays
			}
			const response = await window.electron.invoke(
				"update-memory",
				memoryData
			)
			if (response.error) {
				toast.error(response.error)
			} else {
				toast.success("Memory updated successfully")
				setEditingMemory(null)
				fetchMemories()
			}
		} catch (error) {
			toast.error("Failed to update memory")
		}
	}

	const handleDeleteMemory = async (memory) => {
		if (!window.confirm("Are you sure you want to delete this memory?"))
			return

		try {
			const memoryData = {
				category: memory.category,
				id: memory.id
			}
			const response = await window.electron.invoke(
				"delete-memory",
				memoryData
			)
			if (response.error) {
				toast.error(response.error)
			} else {
				toast.success("Memory deleted successfully")
				fetchMemories()
			}
		} catch (error) {
			toast.error("Failed to delete memory")
		}
	}
	
	if (loading) {
		return <div className="text-white p-6">Loading memories...</div>
	}

	if (error) {
		return <div className="text-red-500 p-6">{error}</div>
	}

	return (
		<div className="w-full h-full flex flex-col p-6 bg-gray-900 rounded-lg text-white">
			{/* Category Buttons */}
			<div className="flex space-x-2 mb-4 overflow-x-auto">
				{categories.map((category) => (
					<button
						key={category}
						onClick={() => setSelectedCategory(category)}
						className={`px-4 py-2 rounded-lg transition-all ${
							selectedCategory === category
								? "bg-blue-600 text-white"
								: "bg-gray-700 text-gray-300 hover:bg-gray-600"
						}`}
					>
						{category}
					</button>
				))}
				<button
					onClick={fetchMemories}
					className="p-2 rounded-md hover:bg-gray-800 text-gray-300"
				>
					<IconRefresh className="h-5 w-5" />
				</button>
			</div>

			{/* Add Memory Form */}
			<div className="mb-6 bg-gray-800 p-4 rounded-lg">
				<h3 className="text-lg font-bold mb-2">Add New Memory</h3>
				<div className="flex flex-col gap-4">
					<textarea
						value={newMemoryText}
						onChange={(e) => setNewMemoryText(e.target.value)}
						placeholder="Memory text"
						className="p-2 bg-gray-700 rounded text-white focus:outline-none"
					/>
					<select
						value={newMemoryCategory}
						onChange={(e) => setNewMemoryCategory(e.target.value)}
						className="p-2 bg-gray-700 rounded text-white"
					>
						{categories.map((cat) => (
							<option key={cat} value={cat}>
								{cat}
							</option>
						))}
					</select>
					<input
						type="number"
						min="1"
						max="90"
						value={newMemoryRetentionDays}
						onChange={(e) =>
							setNewMemoryRetentionDays(e.target.value)
						}
						placeholder="Retention days (1-90)"
						className="p-2 bg-gray-700 rounded text-white focus:outline-none"
					/>
					<button
						onClick={handleAddMemory}
						className="flex items-center p-2 bg-blue-600 hover:bg-blue-700 rounded"
					>
						<IconPlus className="h-4 w-4 mr-2" /> Add Memory
					</button>
				</div>
			</div>

			{/* Memory List */}
			<div className="flex-grow overflow-y-auto">
				{memories.length === 0 ? (
					<div className="text-center text-gray-500 py-10">
						No memories found in {selectedCategory} category
					</div>
				) : (
					<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
						{memories.map((memory) => (
							<div
								key={memory.id}
								className="bg-gray-800 p-4 rounded-lg shadow-md hover:bg-gray-700 transition-colors"
							>
								<p className="text-white mb-2">
									{memory.original_text}
								</p>
								<div className="flex justify-between text-sm text-gray-400 mb-2">
									<span>
										Created:{" "}
										{new Date(
											memory.created_at
										).toLocaleDateString()}
									</span>
									<span>
										Expires:{" "}
										{new Date(
											memory.expiry_at
										).toLocaleDateString()}
									</span>
								</div>
								<div className="flex justify-end gap-2">
									<button
										onClick={() => handleEditMemory(memory)}
										className="p-2 text-yellow-300 hover:bg-yellow-600 rounded"
									>
										<IconPencil className="h-4 w-4" />
									</button>
									<button
										onClick={() =>
											handleDeleteMemory(memory)
										}
										className="p-2 text-red-300 hover:bg-red-600 rounded"
									>
										<IconTrash className="h-4 w-4" />
									</button>
								</div>
							</div>
						))}
					</div>
				)}
			</div>

			{/* Edit Memory Modal */}
			{editingMemory && (
				<div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center">
					<div className="bg-gray-800 p-6 rounded-lg shadow-lg">
						<h3 className="text-xl font-bold mb-4">Edit Memory</h3>
						<div className="mb-4">
							<label className="block text-gray-200 mb-2">
								Text
							</label>
							<textarea
								value={updatedText}
								onChange={(e) => setUpdatedText(e.target.value)}
								className="w-full p-2 bg-gray-700 rounded text-white focus:outline-none"
							/>
						</div>
						<div className="mb-4">
							<label className="block text-gray-200 mb-2">
								New Retention Days (1-90, days from now)
							</label>
							<input
								type="number"
								min="1"
								max="90"
								value={updatedRetentionDays}
								onChange={(e) =>
									setUpdatedRetentionDays(e.target.value)
								}
								className="w-full p-2 bg-gray-700 rounded text-white focus:outline-none"
							/>
						</div>
						<div className="flex justify-end gap-2">
							<button
								onClick={handleUpdateMemory}
								className="p-2 bg-green-600 hover:bg-green-700 rounded"
							>
								Save
							</button>
							<button
								onClick={() => setEditingMemory(null)}
								className="p-2 bg-gray-500 hover:bg-gray-600 rounded"
							>
								Cancel
							</button>
						</div>
					</div>
				</div>
			)}
		</div>
	)
}

export default SQLiteMemoryDisplay