"use client"

import { useRouter } from "next/navigation"
import { useEffect, useState } from "react"
import {
	IconChevronRight,
	IconDotsVertical,
	IconUser,
	IconHome,
	IconLogout,
	IconSettings,
	IconEdit,
	IconTrash,
	IconMessage, // Added for chat icons
	IconPlus // Added for new chat button
} from "@tabler/icons-react"
import SlideButton from "./SlideButton"
import EncryptButton from "./EncryptButton"
import Speeddial from "./SpeedDial"
import toast from "react-hot-toast"
import ModalDialog from "./ModalDialog"
import React from "react"

/**
 * Enhanced Sidebar Component with improved aesthetics and user experience
 *
 * @param {Object} props Component properties
 * @param {Object} props.userDetails User profile information
 * @param {Function} props.setSidebarVisible Control sidebar visibility
 * @param {boolean} props.isSidebarVisible Current sidebar visibility state
 * @param {string} props.chatId Active chat identifier
 * @param {Function} props.setChatId Update active chat
 * @param {boolean} props.fromChat Whether component is rendered in chat context
 */
const Sidebar = ({
	userDetails,
	setSidebarVisible,
	isSidebarVisible,
	chatId,
	setChatId,
	fromChat
}) => {
	const router = useRouter()
	const [chats, setChats] = useState([])
	const [isRenameDialogOpen, setIsRenameDialogOpen] = useState(false)
	const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false)
	const [currentChat, setCurrentChat] = useState(null)
	const [newChatName, setNewChatName] = useState("")
	const [searchQuery, setSearchQuery] = useState("") // Added for chat search

	// Enhanced user menu toggle with smooth animation
	const toggleUserMenu = () => {
		const userMenu = document.getElementById("user-menu")
		const userProfile = document.getElementById("user-profile")
		const userMenuContent = document.getElementById("user-menu-content")

		if (userMenu) {
			userMenu.classList.toggle("hidden")
			userMenu.classList.toggle("animate-fade-in")

			if (!userMenu.classList.contains("hidden")) {
				userMenuContent.style.width = `${userProfile.offsetWidth}px`
			}
		}
	}

	// Enhanced chat fetching with loading state
	const fetchChats = async () => {
		try {
			const response = await window.electron?.invoke("get-chats")
			if (response?.data?.chats) {
				setChats(response.data.chats)
			}
		} catch (error) {
			toast.error("Failed to load chats. Please try again.")
		}
	}

	const logout = async () => {
		try {
			await window.electron.logOut()
			toast.success("Logged out successfully")
		} catch (error) {
			toast.error("Logout failed. Please try again.")
		}
	}

	useEffect(() => {
		fetchChats()
	}, [chatId])

	// Enhanced navigation with loading state
	const handleChatNavigation = (id) => {
		if (fromChat) {
			setChatId(id)
		} else {
			router.push(`/chat?chatId=${id}`)
		}
	}

	// Enhanced chat management functions with better error handling
	const handleRenameChat = async (id, newName) => {
		if (!newName.trim()) {
			toast.error("Please enter a valid chat name")
			return
		}

		try {
			await window.electron?.invoke("rename-chat", { id, newName })
			await fetchChats()
			setIsRenameDialogOpen(false)
			setNewChatName("")
			toast.success("Chat renamed successfully")
		} catch (error) {
			toast.error("Failed to rename chat. Please try again.")
		}
	}

	const handleDeleteChat = async (id) => {
		try {
			await window.electron?.invoke("delete-chat", { id })
			fetchChats()
			setIsDeleteDialogOpen(false)
			toast.success("Chat deleted successfully")
			setChatId("home")
		} catch (error) {
			toast.error("Failed to delete chat. Please try again.")
		}
	}

	// Enhanced speed dial with modern styling
	const speedDialActions = [
		{
			label: "Profile",
			action: () => router.push("/profile"),
			icon: (
				<IconUser className="w-6 h-6 text-white hover:text-blue-400 transition-colors" />
			)
		},
		{
			label: "Settings",
			action: () => router.push("/settings"),
			icon: (
				<IconSettings className="w-6 h-6 text-white hover:text-blue-400 transition-colors" />
			)
		},
		{
			label: "Logout",
			action: logout,
			icon: (
				<IconLogout className="w-6 h-6 text-white hover:text-red-400 transition-colors" />
			)
		}
	]

	// Filter chats based on search query
	const filteredChats = chats.filter((chat) =>
		chat.title.toLowerCase().includes(searchQuery.toLowerCase())
	)

	return (
		<>
			<div
				id="sidebar"
				className={`w-1/5 h-full flex flex-col bg-gradient-to-b from-smokeblack to-gray-900 overflow-y-scroll no-scrollbar transform transition-all duration-300 shadow-xl ${
					isSidebarVisible
						? "translate-x-0 opacity-100 z-40 pointer-events-auto"
						: "-translate-x-full opacity-0 z-0 pointer-events-none"
				}`}
				onMouseLeave={() => setSidebarVisible(false)}
			>
				{/* Enhanced logo section with gradient effect */}
				<div className="flex items-center justify-center px-2 md:px-10 py-6 w-full bg-gradient-to-r from-gray-900 to-smokeblack">
					<img
						src="/images/half-logo-dark.svg"
						alt="Logo"
						className="w-12 h-12 transform hover:scale-110 transition-transform"
					/>
					<span className="text-3xl text-white font-semibold ml-4 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
						Sentient
					</span>
				</div>

				{/* Enhanced navigation section */}
				<div className="flex flex-col gap-6 mt-6 p-6 h-[65%]">
					<SlideButton
						onClick={() => handleChatNavigation("home")}
						primaryColor="#000000"
						className="mb-4 cursor-pointer hover:scale-105 transition-transform"
						text="Home"
						icon={<IconHome className="w-5 h-5" />}
					/>

					{/* Enhanced search input */}
					<div className="relative mb-4">
						<input
							type="text"
							placeholder="Search chats..."
							value={searchQuery}
							onChange={(e) => setSearchQuery(e.target.value)}
							className="w-full px-4 py-2 bg-gray-800 text-white rounded-lg focus:ring-2 focus:ring-blue-400 outline-none"
						/>
					</div>

					<div className="flex justify-between items-center mb-4">
						<h2 className="text-lg text-white font-semibold">
							Your Chats
						</h2>
						<button
							onClick={() => handleChatNavigation("new")}
							className="p-2 rounded-full bg-blue-500 hover:bg-blue-600 transition-colors"
						>
							<IconPlus className="w-5 h-5 text-white" />
						</button>
					</div>

					{/* Enhanced chat list */}
					<ul className="space-y-3 flex flex-col gap-3 overflow-x-hidden overflow-y-scroll no-scrollbar">
						{filteredChats.map((chat) => (
							<EncryptButton
								onClick={() => handleChatNavigation(chat.id)}
								className="w-full text-sm bg-gray-800 rounded-lg p-3 cursor-pointer border border-gray-700 hover:border-blue-400 hover:bg-gray-700 text-white transition-all duration-200 relative group"
								onEdit={() => {
									setCurrentChat(chat)
									setIsRenameDialogOpen(true)
								}}
								onDelete={() => {
									setCurrentChat(chat)
									setIsDeleteDialogOpen(true)
								}}
								key={chat.id}
							>
								<div className="flex items-center">
									<IconMessage className="w-4 h-4 mr-2 text-blue-400" />
									{chat.title}
								</div>
							</EncryptButton>
						))}
					</ul>
				</div>

				{/* Enhanced user profile section */}
				<div
					id="user-profile"
					className="sticky bottom-0 w-full bg-gray-900 border-t border-gray-800 p-4"
				>
					<div className="flex items-center space-x-4">
						<div className="rounded-full overflow-hidden w-12 h-12 ring-2 ring-blue-400 ring-offset-2 ring-offset-gray-900">
							{userDetails["picture"] ? (
								<img
									src={userDetails["picture"]}
									alt="User"
									className="w-full h-full object-cover cursor-pointer hover:opacity-80 transition-opacity"
									onClick={toggleUserMenu}
								/>
							) : (
								<IconUser className="w-full h-full p-2 bg-gray-700 text-white cursor-pointer hover:bg-gray-600 transition-colors" />
							)}
						</div>
						<div className="flex-1">
							<p
								className="text-lg text-white font-medium cursor-pointer hover:text-blue-400 transition-colors"
								onClick={toggleUserMenu}
							>
								{userDetails["given_name"]}
							</p>
							<p className="text-sm text-gray-400">Online</p>
						</div>
					</div>
				</div>

				{/* Enhanced speed dial */}
				<div className="absolute right-4 bottom-20">
					<Speeddial
						icon={<IconDotsVertical className="text-white" />}
						direction="up"
						actionButtons={speedDialActions}
						tooltipPosition="left"
					/>
				</div>
			</div>

			{/* Enhanced sidebar toggle */}
			<div
				className="absolute top-0 left-0 bg-gradient-to-r from-matteblack to-transparent w-[5%] h-full z-10 flex items-center justify-start cursor-pointer"
				onMouseEnter={() => setSidebarVisible(true)}
			>
				<div className="ml-3 transform hover:scale-110 transition-transform">
					<IconChevronRight className="text-white w-6 h-6 animate-pulse" />
				</div>
			</div>

			{/* Enhanced modal dialogs */}
			{isRenameDialogOpen && (
				<ModalDialog
					title="Rename Chat"
					inputPlaceholder="Enter new chat name"
					inputValue={newChatName}
					onInputChange={setNewChatName}
					onCancel={() => setIsRenameDialogOpen(false)}
					onConfirm={() =>
						handleRenameChat(currentChat.id, newChatName)
					}
					confirmButtonText="Rename"
					confirmButtonColor="bg-blue-500 hover:bg-blue-600"
					confirmButtonBorderColor="border-blue-500"
					confirmButtonIcon={IconEdit}
					showInput={true}
				/>
			)}

			{isDeleteDialogOpen && (
				<ModalDialog
					title="Delete Chat"
					description="Are you sure you want to delete this chat? This action cannot be undone."
					onCancel={() => setIsDeleteDialogOpen(false)}
					onConfirm={() => handleDeleteChat(currentChat.id)}
					confirmButtonText="Delete"
					confirmButtonColor="bg-red-500 hover:bg-red-600"
					confirmButtonBorderColor="border-red-500"
					confirmButtonIcon={IconTrash}
					showInput={false}
				/>
			)}
		</>
	)
}

export default Sidebar
