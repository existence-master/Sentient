import {
	IconClock,
	IconRefresh,
	IconCircleCheck,
	IconAlertCircle,
	IconArchive,
	IconMailQuestion,
	IconMessageQuestion,
	IconPlayerPlay,
	IconX,
	IconHelpCircle
} from "@tabler/icons-react"

export const taskStatusColors = {
	planning: {
		icon: IconRefresh,
		label: "Planning",
		textColor: "text-blue-300",
		bgColor: "bg-blue-500/20",
		border: "border-blue-400"
	},
	pending: {
		icon: IconClock,
		label: "Pending",
		textColor: "text-yellow-300",
		bgColor: "bg-yellow-500/20",
		border: "border-yellow-400"
	},
	approval_pending: {
		icon: IconMailQuestion,
		label: "Pending Approval",
		textColor: "text-purple-300",
		bgColor: "bg-purple-500/20",
		border: "border-purple-400"
	},
	clarification_pending: {
		icon: IconMessageQuestion,
		label: "Needs Input",
		textColor: "text-yellow-300",
		bgColor: "bg-yellow-500/20",
		border: "border-yellow-400"
	},
	waiting: {
		icon: IconClock,
		label: "Waiting",
		textColor: "text-yellow-300",
		bgColor: "bg-yellow-500/20",
		border: "border-yellow-400"
	},
	processing: {
		icon: IconPlayerPlay,
		label: "Processing",
		textColor: "text-blue-300",
		bgColor: "bg-blue-500/20",
		border: "border-blue-400"
	},
	active: {
		icon: IconRefresh,
		label: "Active",
		textColor: "text-green-300",
		bgColor: "bg-green-500/20",
		border: "border-green-400"
	},
	completed: {
		icon: IconCircleCheck,
		label: "Completed",
		textColor: "text-green-300",
		bgColor: "bg-green-500/20",
		border: "border-green-400"
	},
	error: {
		icon: IconAlertCircle,
		label: "Error",
		textColor: "text-red-300",
		bgColor: "bg-red-500/20",
		border: "border-red-400"
	},
	cancelled: {
		icon: IconX,
		label: "Cancelled",
		textColor: "text-gray-400",
		bgColor: "bg-gray-500/20",
		border: "border-gray-500"
	},
	archived: {
		icon: IconArchive,
		label: "Archived",
		textColor: "text-gray-400",
		bgColor: "bg-gray-500/20",
		border: "border-gray-500"
	},
	default: {
		icon: IconHelpCircle,
		label: "Unknown",
		textColor: "text-gray-400",
		bgColor: "bg-gray-500/20",
		border: "border-gray-400"
	}
}

export const priorityMap = {
	0: { label: "High", color: "text-red-400" },
	1: { label: "Medium", color: "text-yellow-400" },
	2: { label: "Low", color: "text-gray-400" },
	default: { label: "Medium", color: "text-yellow-400" }
}
