"use client"
import React from "react"
import { IconDatabase, IconBrain } from "@tabler/icons-react"
import { cn } from "@utils/cn" // Assuming cn utility is available

// ADDED: New component for switching between memory view types
const MemoryTypeSwitcher = ({ currentType, onTypeChange }) => {
	const buttonStyle = (type) =>
		cn(
			"flex-1 px-4 py-2.5 rounded-full text-sm font-semibold transition-all duration-200 ease-in-out flex items-center justify-center gap-2", // Increased padding/font-size
			"focus:outline-none focus:ring-2 focus:ring-lightblue focus:ring-opacity-50", // Added focus state
			currentType === type
				? "bg-lightblue text-white shadow-md" // Active state
				: "text-gray-400 hover:text-white hover:bg-neutral-700/60" // Inactive state
		)

	return (
		<div className="flex items-center space-x-1 bg-neutral-800/80 backdrop-blur-sm rounded-full p-1.5 shadow-lg border border-neutral-700 w-1/4">
			{" "}
			{/* Increased padding */}
			{/* Graph (Long-Term) Button */}
			<button
				onClick={() => onTypeChange("neo4j")}
				className={buttonStyle("neo4j")}
				title="View Long-Term Knowledge Graph"
			>
				<IconDatabase className="w-5 h-5" /> {/* Increased size */}
				<span>Long-Term</span>
			</button>
			{/* List (Short-Term) Button */}
			<button
				onClick={() => onTypeChange("sqlite")}
				className={buttonStyle("sqlite")}
				title="View Short-Term Memories List"
			>
				<IconBrain className="w-5 h-5" /> {/* Increased size */}
				<span>Short-Term</span>
			</button>
		</div>
	)
}

export default MemoryTypeSwitcher
