import React, { useEffect, useState } from "react"
import toast from "react-hot-toast" // Library for displaying toast notifications

const SQLiteMemoryDisplay = ({ userDetails }) => {
	const [memories, setMemories] = useState([])
	const [selectedCategory, setSelectedCategory] = useState("PERSONAL")

	useEffect(() => {
		const fetchMemories = async () => {
			try {
				const memories = await window.electron?.invoke(
					"fetch-short-term-memories",
					{
						category: selectedCategory.toLowerCase()
					}
				)
				setMemories(memories || [])
			} catch (error) {
				toast.error(`Error fetching memories: ${error}`)
			}
		}

		fetchMemories()
	}, [userDetails, selectedCategory])

	// Categories based on the MemoryManager in the SQLite code
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

	return (
		<div className="w-full h-full flex flex-col p-6 bg-gray-900 rounded-lg">
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
			</div>

			<div className="flex-grow overflow-y-auto">
				{memories.length === 0 ? (
					<div className="text-center text-gray-500 py-10">
						No memories found in {selectedCategory} category
					</div>
				) : (
					<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
						{memories.map((memory, index) => (
							<div
								key={memory.id || index}
								className="bg-gray-800 p-4 rounded-lg shadow-md hover:bg-gray-700 transition-colors"
							>
								<p className="text-white mb-2">
									{memory.original_text}
								</p>
								<div className="flex justify-between text-sm text-gray-400">
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
							</div>
						))}
					</div>
				)}
			</div>
		</div>
	)
}

export default SQLiteMemoryDisplay
