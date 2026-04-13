"use client"

import React, { useState, useMemo, useRef, useCallback } from "react"
import toast from "react-hot-toast"
import {
	IconLoader,
	IconBrain,
	IconTag,
	IconClock,
	IconFileText,
	IconX,
	IconLayoutGrid,
	IconShare3,
	IconInfoCircle,
	IconHeart,
	IconPlus,
	IconPencil,
	IconTrash,
	IconDeviceFloppy,
	IconCheck,
	IconSparkles
} from "@tabler/icons-react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { motion, AnimatePresence } from "framer-motion"
import { formatDistanceToNow, parseISO } from "date-fns"
import { cn } from "@utils/cn"
import dynamic from "next/dynamic"
import InteractiveNetworkBackground from "@components/ui/InteractiveNetworkBackground"
import {
	ModalDialog,
	ModalHeader,
	ModalTitle,
	ModalBody,
	ModalFooter,
	ModalCloseButton
} from "@components/ui/ModalDialog"
import useClickOutside from "@hooks/useClickOutside"
import { useRouter, useSearchParams } from "next/navigation"
import { Button } from "@components/ui/button"
import { Card, CardContent, CardFooter } from "@components/ui/card"
import { Drawer } from "@components/ui/drawer"
import { Textarea } from "@components/ui/textarea"
import apiClient, { ApiError } from "@lib/apiClient"
import { useMemoryStore } from "@stores/app-stores"

const InfoPanel = ({ onClose, title, children }) => (
	<motion.div
		initial={{ opacity: 0, backdropFilter: "blur(0px)" }}
		animate={{ opacity: 1, backdropFilter: "blur(12px)" }}
		exit={{ opacity: 0, backdropFilter: "blur(0px)" }}
		className="fixed inset-0 bg-black/70 z-[60] flex p-4 md:p-6"
		onClick={onClose}
	>
		<motion.div
			initial={{ opacity: 0, y: 20 }}
			animate={{ opacity: 1, y: 0 }}
			exit={{ opacity: 0, y: 20 }}
			transition={{ duration: 0.2, ease: "easeInOut" }}
			onClick={(e) => e.stopPropagation()}
			className="relative bg-neutral-900/80 backdrop-blur-2xl p-6 rounded-2xl shadow-2xl w-full h-full border border-neutral-700 flex flex-col"
		>
			<header className="flex justify-between items-center mb-6 flex-shrink-0">
				<h2 className="text-lg font-semibold text-white flex items-center gap-2">
					{title}
				</h2>
				<button
					onClick={onClose}
					className="p-1.5 rounded-full hover:bg-neutral-700"
				>
					<IconX size={18} />
				</button>
			</header>
			<main className="flex-1 overflow-y-auto custom-scrollbar pr-2 text-left space-y-6">
				{children}
			</main>
		</motion.div>
	</motion.div>
)

const MemoryDetailPanel = ({ memory, onClose, onUpdate, onDelete }) => {
	if (!memory) return null

	const [isEditing, setIsEditing] = useState(false)
	const [editedContent, setEditedContent] = useState(memory.content)
	const [isSaving, setIsSaving] = useState(false)
	const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
	const [isDeleting, setIsDeleting] = useState(false)

	const timeAgo = formatDistanceToNow(parseISO(memory.created_at), {
		addSuffix: true
	})

	const handleSave = async () => {
		if (editedContent.trim() === memory.content) {
			setIsEditing(false)
			return
		}
		setIsSaving(true)
		await onUpdate(memory.id, editedContent)
		setIsSaving(false)
		setIsEditing(false)
	}

	const handleDeleteConfirm = async () => {
		setIsDeleting(true)
		await onDelete(memory.id)
		// The parent component will handle closing the panel
	}

	return (
		<>
			<motion.div
				key={memory.id}
				initial={{ x: "100%" }}
				animate={{ x: 0 }}
				exit={{ x: "100%" }}
				transition={{ type: "spring", stiffness: 300, damping: 30 }}
				className="fixed inset-0 md:absolute md:top-0 md:right-0 md:inset-auto h-full w-full md:w-[400px] lg:w-[450px] bg-black/50 backdrop-blur-lg md:border-l border-neutral-800 flex flex-col z-50"
			>
				<header className="flex items-start justify-between p-4 border-b border-neutral-800 flex-shrink-0">
					<div className="flex items-center gap-3">
						<IconBrain className="w-6 h-6 text-brand-orange" />
						<h2 className="text-lg font-semibold text-white">
							Memory
						</h2>
					</div>
					<button
						onClick={onClose}
						className="p-1.5 rounded-full text-neutral-400 hover:bg-neutral-700 hover:text-white"
					>
						<IconX size={18} />
					</button>
				</header>
				<main className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
					<div>
						<h3 className="text-sm font-semibold text-neutral-400 mb-2">
							CONTENT
						</h3>
						{isEditing ? (
							<Textarea
								value={editedContent}
								onChange={(e) =>
									setEditedContent(e.target.value)
								}
								className="w-full h-40 p-3 text-base"
								autoFocus
							/>
						) : (
							<p className="text-neutral-200 text-base font-sans leading-relaxed whitespace-pre-wrap">
								{memory.content}
							</p>
						)}
					</div>
					<div className="space-y-4">
						<div className="flex items-start gap-3">
							<IconClock
								size={16}
								className="text-neutral-500 mt-0.5"
							/>
							<div>
								<h4 className="text-sm font-semibold text-neutral-400">
									CREATED
								</h4>
								<p className="text-xs text-neutral-300">
									{timeAgo}
								</p>
							</div>
						</div>
						{memory.source && (
							<div className="flex items-start gap-3">
								<IconFileText
									size={16}
									className="text-neutral-500 mt-0.5"
								/>
								<div>
									<h4 className="text-sm font-semibold text-neutral-400">
										SOURCE
									</h4>
									<p className="text-xs text-neutral-300">
										{memory.source}
									</p>
								</div>
							</div>
						)}
						{memory.topics && memory.topics.length > 0 && (
							<div className="flex items-start gap-3">
								<IconTag
									size={16}
									className="text-neutral-500 mt-0.5"
								/>
								<div>
									<h4 className="text-sm font-semibold text-neutral-400">
										TOPICS
									</h4>
									<div className="flex flex-wrap gap-2 mt-1">
										{memory.topics.map((topic) => (
											<span
												key={topic}
												className="bg-neutral-800 px-2 py-0.5 rounded-full text-xs text-neutral-300"
											>
												{topic}
											</span>
										))}
									</div>
								</div>
							</div>
						)}
					</div>
				</main>
				<footer className="p-4 border-t border-neutral-800 flex-shrink-0 flex justify-end gap-2">
					{isEditing ? (
						<>
							<button
								onClick={() => {
									setIsEditing(false)
									setEditedContent(memory.content)
								}}
								className="py-2 px-4 rounded-lg bg-neutral-700 hover:bg-neutral-600 text-sm font-medium"
							>
								Cancel
							</button>
							<button
								onClick={handleSave}
								disabled={isSaving}
								className="py-2 px-4 rounded-lg bg-brand-orange hover:bg-brand-orange/90 text-brand-black font-semibold text-sm flex items-center gap-2"
							>
								{isSaving ? (
									<IconLoader
										size={16}
										className="animate-spin"
									/>
								) : (
									<IconDeviceFloppy size={16} />
								)}
								{isSaving ? "Saving..." : "Save"}
							</button>
						</>
					) : (
						<>
							<button
								onClick={() => setShowDeleteConfirm(true)}
								className="py-2 px-4 rounded-lg bg-red-600/20 hover:bg-red-600/40 text-red-300 text-sm font-medium flex items-center gap-2"
							>
								<IconTrash size={16} /> Delete
							</button>
							<button
								onClick={() => setIsEditing(true)}
								className="py-2 px-4 rounded-lg bg-neutral-700 hover:bg-neutral-600 text-sm font-medium flex items-center gap-2"
							>
								<IconPencil size={16} /> Edit
							</button>
						</>
					)}
				</footer>
			</motion.div>
			<AnimatePresence>
				{showDeleteConfirm && (
					<ModalDialog
						isOpen={showDeleteConfirm}
						onClose={() => setShowDeleteConfirm(false)}
					>
						<ModalHeader>
							<ModalTitle>Delete Memory</ModalTitle>
							<ModalCloseButton
								onClose={() => setShowDeleteConfirm(false)}
							/>
						</ModalHeader>
						<ModalBody>
							<p>
								Are you sure you want to permanently delete this
								memory? This action cannot be undone.
							</p>
						</ModalBody>
						<ModalFooter>
							<Button
								variant="secondary"
								onClick={() => setShowDeleteConfirm(false)}
							>
								Cancel
							</Button>
							<Button
								variant="destructive"
								onClick={handleDeleteConfirm}
								disabled={isDeleting}
							>
								{isDeleting && (
									<IconLoader
										size={16}
										className="animate-spin mr-2"
									/>
								)}
								Delete
							</Button>
						</ModalFooter>
					</ModalDialog>
				)}
			</AnimatePresence>
		</>
	)
}

const CreateMemoryModal = ({ onClose, onCreate, userDetails }) => {
	const [content, setContent] = useState("")
	const [isSaving, setIsSaving] = useState(false)

	const handleCreate = async () => {
		if (!content.trim()) {
			toast.error("Memory content cannot be empty.")
			return
		}
		setIsSaving(true)
		await onCreate(content)
		setIsSaving(false)
		setContent("") // Reset for next time
	}

	return (
		<ModalDialog
			isOpen={true}
			onClose={onClose}
			className="max-w-lg bg-neutral-900/90 backdrop-blur-xl p-0"
		>
			<ModalHeader className="p-6">
				<ModalTitle>Add a New Memory</ModalTitle>
				<ModalCloseButton onClose={onClose} />
			</ModalHeader>
			<ModalBody className="p-6 pt-0">
				<Textarea
					value={content}
					onChange={(e) => setContent(e.target.value)}
					placeholder="Enter a fact or piece of information to remember..."
					className="w-full h-40 p-3 text-base"
					autoFocus
				/>
				<p className="text-xs text-neutral-500 mt-2 px-1">
					Note: All memories should be in the third person. e.g., "
					<span className="font-semibold text-neutral-400">
						{userDetails?.given_name || "User"} likes football
					</span>
					" instead of "I like football".
				</p>
			</ModalBody>
			<ModalFooter className="p-6">
				<Button onClick={onClose} variant="secondary">
					Cancel
				</Button>
				<Button
					onClick={handleCreate}
					disabled={isSaving}
					className="gap-2 bg-brand-orange hover:bg-brand-orange/90 text-brand-black font-semibold"
				>
					{isSaving ? (
						<IconLoader size={16} className="animate-spin" />
					) : (
						<IconPlus size={16} />
					)}
					{isSaving ? "Saving..." : "Add Memory"}
				</Button>
			</ModalFooter>
		</ModalDialog>
	)
}

const MotionCard = motion(Card)

const MemoryCard = ({ memory, onSelect }) => {
	const timeAgo = formatDistanceToNow(parseISO(memory.created_at), {
		addSuffix: true
	})

	return (
		<MotionCard
			layout
			initial={{ opacity: 0, scale: 0.9 }}
			animate={{ opacity: 1, scale: 1 }}
			exit={{ opacity: 0, scale: 0.9 }}
			transition={{ duration: 0.3 }}
			onClick={() => onSelect(memory)}
			className="flex flex-col justify-between text-left h-full shadow-lg hover:border-brand-orange/50 transition-colors cursor-pointer rounded-2xl"
		>
			<CardContent className="p-6">
				<p className="text-neutral-200 text-base font-sans leading-relaxed line-clamp-6">
					{memory.content}
				</p>
			</CardContent>
			<CardFooter className="p-6 pt-4 border-t border-neutral-800/50 text-xs text-neutral-500 flex-col items-start space-y-2">
				<div className="flex items-center gap-2">
					<IconClock size={14} />
					<span>{timeAgo}</span>
				</div>
				{memory.source && (
					<div className="flex items-center gap-2">
						<IconFileText size={14} />
						<span>Source: {memory.source}</span>
					</div>
				)}
				{memory.topics && memory.topics.length > 0 && (
					<div className="flex items-center gap-2 flex-wrap pt-1">
						<IconTag size={14} />
						{memory.topics.map((topic) => (
							<span
								key={topic}
								className="bg-neutral-800 px-2 py-0.5 rounded-full text-neutral-300"
							>
								{topic}
							</span>
						))}
					</div>
				)}
			</CardFooter>
		</MotionCard>
	)
}

const ForceGraph3D = dynamic(() => import("react-force-graph-3d"), {
	ssr: false,
	loading: () => (
		<p className="flex-1 flex justify-center items-center text-neutral-500">
			Loading 3D Graph...
		</p>
	)
})

const MemoryGraph = ({ data, onSelectNode, onClearSelection }) => {
	const fgRef = useRef()

	const graphData = useMemo(() => {
		if (!data) return { nodes: [], links: [] }
		const links = (data.links || data.edges || []).map((edge) => ({
			source: edge.source || edge.from,
			target: edge.target || edge.to,
			...edge
		}))
		const nodes = data.nodes.map((node) => ({
			...node,
			color: node.id % 5 === 0 ? "#22c55e" : "#f59e0b",
			val: 1
		}))
		return { nodes, links }
	}, [data])

	const handleNodeClick = useCallback(
		(node) => {
			const originalNode = data.nodes.find((n) => n.id === node.id)
			if (originalNode) onSelectNode(originalNode)
			const distance = 120
			const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z)
			if (fgRef.current) {
				fgRef.current.cameraPosition(
					{
						x: node.x * distRatio,
						y: node.y * distRatio,
						z: node.z * distRatio
					},
					node,
					1000
				)
			}
		},
		[data.nodes, onSelectNode]
	)

	return (
		<div className="w-full h-full bg-transparent absolute inset-0">
			<ForceGraph3D
				ref={fgRef}
				graphData={graphData}
				backgroundColor="rgba(0,0,0,0)"
				nodeLabel="title"
				nodeVal={4}
				nodeColor="color"
				nodeOpacity={0.8}
				nodeResolution={8}
				linkColor={() => "rgba(74, 158, 255, 1)"}
				linkWidth={2}
				onNodeClick={handleNodeClick}
				onBackgroundClick={onClearSelection}
			/>
		</div>
	)
}

// --- Tab Component for View Switcher (from tasks page) ---
const buttonVariants = {
	initial: {
		gap: 0,
		paddingLeft: ".5rem",
		paddingRight: ".5rem"
	},
	animate: (selected) => ({
		gap: selected ? ".5rem" : 0,
		paddingLeft: selected ? "1rem" : ".5rem",
		paddingRight: selected ? "1rem" : ".5rem"
	})
}

const spanVariants = {
	initial: { width: 0, opacity: 0 },
	animate: { width: "auto", opacity: 1 },
	exit: { width: 0, opacity: 0 }
}

const transition = { delay: 0.1, type: "spring", bounce: 0, duration: 0.35 }

const Tab = ({ text, selected, setSelected, value, children, ...props }) => {
	return (
		<motion.button
			variants={buttonVariants}
			initial="initial"
			animate="animate"
			{...props}
			custom={selected}
			onClick={() => setSelected(value)}
			transition={transition}
			className={`${
				selected ? "bg-white/10 text-white " : " hover:text-white"
			} relative flex items-center rounded-full px-4 py-2 text-sm font-medium text-neutral-400 transition-colors duration-300`}
		>
			{children}
			<AnimatePresence>
				{selected && (
					<motion.span
						variants={spanVariants}
						initial="initial"
						animate="animate"
						exit="exit"
						transition={transition}
						className="overflow-hidden"
					>
						{text}
					</motion.span>
				)}
			</AnimatePresence>
		</motion.button>
	)
}

const memoryTabs = [
	{ title: "Graph", value: "graph", icon: <IconShare3 size={16} /> },
	{ title: "List", value: "list", icon: <IconLayoutGrid size={16} /> }
]

export default function MemoriesPage() {
	const {
		view,
		activeTopic,
		selectedMemory,
		isInfoPanelOpen,
		isCreateModalOpen,
		setView,
		setActiveTopic,
		setSelectedMemory,
		openInfoPanel,
		closeInfoPanel,
		openCreateModal,
		closeCreateModal
	} = useMemoryStore()
	const router = useRouter()
	const searchParams = useSearchParams()

	const queryClient = useQueryClient()

	const { data: userDetails } = useQuery({
		queryKey: ["userProfile"],
		queryFn: async () => {
			const res = await fetch("/api/user/profile")
			if (!res.ok) {
				throw new Error("Failed to fetch user details")
			}
			return res.json()
		},
		staleTime: Infinity // User profile is unlikely to change during a session
	})

	const { data, isLoading } = useQuery({
		queryKey: ["memories", view],
		queryFn: async () => {
			const endpoint =
				view === "list" ? "/api/memories" : "/api/memories/graph"
			const response = await fetch(endpoint, { cache: "no-store" })
			if (!response.ok) {
				throw new Error(
					`Failed to fetch memories data for ${view} view.`
				)
			}
			return response.json()
		}
	})

	const memories = useMemo(() => {
		if (view === "list") {
			return data?.memories || []
		}
		return data?.nodes || []
	}, [data, view])

	const graphData = useMemo(() => {
		if (view === "graph") {
			return data || { nodes: [], links: [] }
		}
		return { nodes: [], links: [] }
	}, [data, view])

	const topics = useMemo(() => {
		const allTopics = new Set()
		memories.forEach((memory) => {
			;(memory.topics || []).forEach((topic) => allTopics.add(topic))
		})
		return ["All", ...Array.from(allTopics).sort()]
	}, [memories])

	const filteredMemories = useMemo(() => {
		if (activeTopic === "All") return memories
		return memories.filter((memory) =>
			(memory.topics || []).includes(activeTopic)
		)
	}, [memories, activeTopic])

	React.useEffect(() => {
		const memoryId = searchParams.get("memoryId")
		if (memoryId && memories.length > 0) {
			const memoryToSelect = memories.find(
				(m) => String(m.id) === memoryId
			)
			if (memoryToSelect) {
				setSelectedMemory(memoryToSelect)
				router.replace("/memories", { scroll: false })
			}
		}
	}, [searchParams, memories, router, setSelectedMemory])

	const createMemoryMutation = useMutation({
		mutationFn: (content) =>
			fetch("/api/memories", {
				method: "POST",
				body: JSON.stringify({ content, source: "manual_entry" })
			}).then(async (res) => {
				if (!res.ok) {
					const errorData = await res.json().catch(() => ({}))
					const error = new Error(
						errorData.error || "Failed to add memory"
					)
					error.status = res.status
					throw error
				}
				return res.json()
			}),
		onSuccess: () => {
			toast.success("Memory added successfully!")
			closeCreateModal()
			queryClient.invalidateQueries({ queryKey: ["memories"] })
		},
		onError: (error) => {
			toast.error(error.message)
		}
	})

	const updateMemoryMutation = useMutation({
		mutationFn: ({ memoryId, newContent }) =>
			fetch(`/api/memories/${memoryId}`, {
				method: "PUT",
				body: JSON.stringify({ content: newContent })
			}).then((res) => {
				if (!res.ok) throw new Error("Failed to update memory")
				return res.json()
			}),
		onSuccess: () => {
			toast.success("Memory updated!")
			setSelectedMemory(null) // This will be handled by the store action
			queryClient.invalidateQueries({ queryKey: ["memories"] })
		},
		onError: (error) => {
			toast.error(error.message)
		}
	})

	const deleteMemoryMutation = useMutation({
		mutationFn: (memoryId) =>
			fetch(`/api/memories/${memoryId}`, { method: "DELETE" }).then(
				(res) => {
					if (!res.ok) throw new Error("Failed to delete memory")
					return res.json()
				}
			),
		onSuccess: () => {
			toast.success("Memory deleted.")
			setSelectedMemory(null)
			queryClient.invalidateQueries({ queryKey: ["memories"] })
		},
		onError: (error) => {
			toast.error(error.message)
		}
	})

	const ViewSwitcher = () => (
		<div className="flex flex-wrap items-center gap-2">
			{memoryTabs.map((tab) => (
				<Tab
					key={tab.value}
					text={tab.title}
					selected={view === tab.value}
					setSelected={setView}
					value={tab.value}
				>
					{tab.icon}
				</Tab>
			))}
		</div>
	)

	return (
		<div className="flex-1 flex h-screen text-white overflow-hidden">
			<AnimatePresence>
				{isInfoPanelOpen && (
					<InfoPanel
						onClose={closeInfoPanel}
						title={
							<div className="flex items-center gap-2">
								<IconSparkles /> About Memories
							</div>
						}
					>
						<p className="text-neutral-300">
							This is your memory, a collection of facts and
							information I've learned about you to provide a
							truly personalized experience.
						</p>
						<div className="space-y-4">
							<div className="flex items-start gap-4">
								<IconBrain
									size={20}
									className="text-brand-orange flex-shrink-0 mt-1"
								/>
								<div>
									<h3 className="font-semibold text-white">
										How it Works
									</h3>
									<p className="text-neutral-400 text-sm mt-1">
										As we interact, I identify key pieces of
										information—like your preferences,
										goals, relationships, and interests—and
										store them as individual memories. This
										can happen during our chats or when I
										process information from your connected
										apps.
									</p>
								</div>
							</div>
							<div className="flex items-start gap-4">
								<IconHeart
									size={20}
									className="text-brand-orange flex-shrink-0 mt-1"
								/>
								<div>
									<h3 className="font-semibold text-white">
										How it's Used
									</h3>
									<p className="text-neutral-400 text-sm mt-1">
										I use these memories to provide more
										context-aware assistance. For example,
										if you ask me to 'email my manager', I
										can look up who your manager is from my
										memory instead of asking you every time.
										This makes our interactions faster and
										more intelligent.
									</p>
								</div>
							</div>
						</div>
					</InfoPanel>
				)}
			</AnimatePresence>
			<div className="flex-1 flex flex-col overflow-hidden relative w-full">
				<div className="absolute inset-0 z-[-1] network-grid-background">
					{view === "list" && <InteractiveNetworkBackground />}
				</div>
				<div className="absolute -top-[250px] left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-brand-orange/10 rounded-full blur-3xl -z-10" />

				<header className="flex flex-wrap items-center justify-between gap-4 p-6 pt-20 md:pt-6 bg-transparent shrink-0 z-10 border-b border-neutral-800/80">
					<h1 className="text-2xl md:text-3xl font-bold text-white">
						Memories
					</h1>
					<div className="flex items-center gap-2 sm:gap-4">
						{!isLoading && (
							<div className="text-center">
								<p className="text-xl sm:text-2xl font-bold text-brand-orange">
									{memories.length}
								</p>
								<p className="text-[10px] sm:text-xs text-neutral-400 -mt-1">
									Memories
								</p>
							</div>
						)}
						<div className="w-px h-8 bg-neutral-700 hidden sm:block"></div>
						<ViewSwitcher />
						<button
							onClick={openInfoPanel}
							className="p-2 rounded-full bg-neutral-800/50 hover:bg-neutral-700/80 text-white"
							aria-label="About memories"
						>
							<IconInfoCircle size={20} />
						</button>
					</div>
				</header>
				<main className="flex-1 relative overflow-hidden">
					{isLoading ? (
						<div className="w-full h-full flex justify-center items-center">
							<IconLoader className="w-12 h-12 animate-spin text-brand-orange" />
						</div>
					) : view === "graph" ? (
						<div className="w-full h-full">
							<MemoryGraph
								data={graphData}
								onSelectNode={setSelectedMemory}
								onClearSelection={() => setSelectedMemory(null)}
							/>
						</div>
					) : (
						<div className="overflow-y-auto h-full p-6 md:p-10 custom-scrollbar">
							<div className="w-full max-w-7xl mx-auto">
								<div className="flex flex-wrap gap-2 mb-8">
									{topics.map((topic) => (
										<button
											key={topic}
											onClick={() =>
												setActiveTopic(topic)
											}
											className={cn(
												"px-4 py-2 rounded-full text-sm font-medium transition-colors",
												activeTopic === topic
													? "bg-brand-orange text-black"
													: "bg-neutral-800 text-neutral-300 hover:bg-neutral-700 hover:text-white"
											)}
										>
											{topic}
										</button>
									))}
								</div>
								{filteredMemories.length > 0 ? (
									<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
										{filteredMemories.map((memory) => (
											<MemoryCard
												key={memory.id}
												memory={memory}
												onSelect={setSelectedMemory}
											/>
										))}
									</div>
								) : (
									<div className="text-center py-20 text-neutral-500">
										<IconBrain
											size={48}
											className="mx-auto mb-4"
										/>
										<h3 className="text-xl font-semibold text-neutral-300">
											No memories found
										</h3>
										<p>
											{activeTopic === "All"
												? "I haven't learned anything about you yet. Try telling me something in the chat!"
												: `No memories match the topic '${activeTopic}'.`}
										</p>
									</div>
								)}
							</div>
						</div>
					)}
				</main>
			</div>
			<AnimatePresence>
				{selectedMemory && (
					<MemoryDetailPanel
						memory={selectedMemory}
						onClose={() => setSelectedMemory(null)}
						onUpdate={(id, content) =>
							updateMemoryMutation.mutate({
								memoryId: id,
								newContent: content
							})
						}
						onDelete={(id) => deleteMemoryMutation.mutate(id)}
					/>
				)}
			</AnimatePresence>
			<button
				onClick={openCreateModal}
				className="fixed bottom-6 right-6 z-40 p-4 bg-brand-orange text-black rounded-full shadow-lg hover:bg-brand-orange/90 transition-transform hover:scale-105"
				aria-label="Add new memory"
			>
				<IconPlus size={24} strokeWidth={2.5} />
			</button>
			<AnimatePresence>
				{isCreateModalOpen && (
					<CreateMemoryModal
						onClose={closeCreateModal}
						onCreate={(content) =>
							createMemoryMutation.mutate(content)
						}
						userDetails={userDetails}
					/>
				)}
			</AnimatePresence>
		</div>
	)
}
