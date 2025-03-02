"use client"

import { useRouter } from "next/navigation"
import { useEffect, useState } from "react"
import {
	IconChevronRight,
	IconUser,
	IconHome,
	IconLogout,
	IconEdit,
	IconTrash,
	IconTemplate,
	IconAdjustments
} from "@tabler/icons-react"
import toast from "react-hot-toast"
import ModalDialog from "./ModalDialog"
import React from "react"

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
	const [pricing, setPricing] = useState("free")

	const toggleUserMenu = () => {
		const userMenu = document.getElementById("user-menu")
		const userProfile = document.getElementById("user-profile")
		const userMenuContent = document.getElementById("user-menu-content")

		if (userMenu) {
			userMenu.classList.toggle("hidden")

			if (!userMenu.classList.contains("hidden")) {
				userMenuContent.style.width = `${userProfile.offsetWidth}px`
			}
		}
	}

	const fetchChats = async () => {
		try {
			const response = await window.electron?.invoke("get-chats")
			if (response?.data?.chats) {
				setChats(response.data.chats)
			}
		} catch (error) {
			toast.error(`Error fetching chats: ${error}`)
		}
	}

	const fetchPricingPlan = async () => {
		try {
			const response = await window.electron?.invoke("fetch-pricing-plan")
			setPricing(response || "free")
		} catch (error) {
			toast.error("Error fetching pricing plan.")
		}
	}

	const logout = async () => {
		await window.electron.logOut()
	}

	useEffect(() => {
		fetchChats()
		fetchPricingPlan()
	}, [chatId])

	const handleChatNavigation = (id) => {
		if (fromChat) {
			setChatId(id)
		} else {
			router.push(`/chat?chatId=${id}`)
		}
	}

	const handleRenameChat = async (id, newName) => {
		try {
			await window.electron?.invoke("rename-chat", { id, newName })
			await fetchChats()
			setIsRenameDialogOpen(false)
			setNewChatName("")
			toast.success("Chat renamed successfully")
		} catch (error) {
			toast.error(`Error renaming chat: ${error}`)
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
			toast.error(`Error deleting chat: ${error}`)
		}
	}

	return (
		<>
			<div
				id="sidebar"
				className={`w-1/5 h-full flex flex-col bg-smokeblack overflow-y-auto transform transition-all duration-300 ${
					isSidebarVisible
						? "translate-x-0 opacity-100 z-40 pointer-events-auto"
						: "-translate-x-full opacity-0 z-0 pointer-events-none"
				}`}
				onMouseLeave={() => setSidebarVisible(false)}
			>
				{/* Logo and title section */}
				<div className="flex items-center px-6 py-6">
					<div className="flex items-center justify-center rounded-xl w-12 h-12">
						<img
							src="/images/half-logo-dark.svg"
							alt="Logo"
							className="w-8 h-8"
						/>
					</div>
					<span className="text-2xl text-white font-extralight ml-3">
						Sentient
					</span>
				</div>

				{/* Navigation menu */}
				<div className="flex flex-col px-4 pt-4 pb-8 flex-grow">
					{/* Home/Chat navigation button */}
					<button
						onClick={() => handleChatNavigation("home")}
						className="cursor-pointer flex items-center gap-3 text-white bg-matteblack hover:bg-neutral-800 hover:text-lightblue rounded-lg px-4 py-3 mb-4 w-full text-left"
					>
						<IconHome className="w-5 h-5" />
						<span className="text-base text-white font-normal">
							Home
						</span>
					</button>

					{/* Your Chats section */}
					<div className="mt-4 mb-2">
						<h2 className="text-xl text-white font-normal px-4 mb-2">
							Your Chats
						</h2>
						<div className="space-y-1 max-h-48 overflow-y-auto">
							{chats.map((chat) => (
								<div key={chat.id} className="relative group">
									<button
										onClick={() =>
											handleChatNavigation(chat.id)
										}
										className="flex items-center gap-3 text-[#9ca3af] hover:text-white border border-gray-600 hover:bg-neutral-800 rounded-lg px-4 py-2 w-full text-left text-sm"
									>
										<span className="truncate">
											{chat.title}
										</span>
									</button>
									<div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
										<button
											onClick={() => {
												setCurrentChat(chat)
												setIsRenameDialogOpen(true)
											}}
											className="p-1 text-gray-400 hover:text-lightblue"
										>
											<IconEdit className="w-4 h-4" />
										</button>
										<button
											onClick={() => {
												setCurrentChat(chat)
												setIsDeleteDialogOpen(true)
											}}
											className="p-1 text-gray-400 hover:text-lightblue"
										>
											<IconTrash className="w-4 h-4" />
										</button>
									</div>
								</div>
							))}
						</div>
					</div>

					{/* Additional menu items similar to image */}
					<div className="mt-4">
						{/* PRO Button Sample */}
						{/* <button className="flex items-center justify-between w-full text-left px-4 py-2 rounded-lg text-[#9ca3af] hover:text-white hover:bg-[#323541]">
							<div className="flex items-center gap-3">
								<IconTemplate className="w-5 h-5" />
								<span className="text-base">Templates</span>
							</div>
							<span className="bg-darkblue text-lightblue text-xs font-medium px-2 py-1 rounded">
								PRO
							</span>
						</button> */}

						<button
							onClick={() => router.push("/profile")}
							className="cursor-pointer flex items-center justify-between w-full text-left px-4 py-2 rounded-lg text-white hover:text-lightblue hover:bg-neutral-800 mt-1"
						>
							<div className="flex items-center gap-3">
								<IconTemplate className="w-5 h-5" />
								<span className="text-base text-white">
									Profile
								</span>
							</div>
						</button>

						<button
							onClick={() => router.push("/settings")}
							className="cursor-pointer flex items-center gap-3 w-full text-left px-4 py-2 rounded-lg text-white hover:text-lightblue hover:bg-neutral-800 mt-1"
						>
							<IconAdjustments className="w-5 h-5" />
							<span className="text-base text-white">
								Settings
							</span>
						</button>
					</div>

					{/* Pro Plan banner */}
					<div className="mt-auto mb-6 mx-2">
						<div className="bg-gradient-to-br from-darkblue to-lightblue rounded-xl p-4 relative overflow-hidden">
							<div className="absolute top-0 right-0 w-24 h-24 rounded-full bg-white opacity-10 -mr-8 -mt-8"></div>
							<div className="flex items-center mb-2">
								<div className="bg-white bg-opacity-20 p-1 rounded-lg">
									<img
										src="/images/half-logo-dark.svg"
										alt="Logo"
										className="w-6 h-6"
									/>
								</div>
								<span className="text-xl font-bold text-white ml-2">
									Pro Plan
								</span>
							</div>
							<p className="text-white text-sm mb-4">
								Unlimited access to all features!
							</p>
							<div className="flex items-center justify-between">
								<span className="text-white font-bold">
									$3 / mo
								</span>
								<button
									onClick={() => router.push("/settings")}
									className="bg-white text-black font-medium py-1 px-4 rounded-full"
								>
									Get
								</button>
							</div>
						</div>
					</div>

					{/* Logout Button */}
					<button
						onClick={logout}
						className="flex items-center justify-between px-6 py-2 text-[#9ca3af] hover:text-white"
					>
						<span className="text-base">Log out</span>
						<IconLogout className="w-5 h-5" />
					</button>
				</div>

				{/* User profile section at the bottom */}
				<div
					id="user-profile"
					className="px-6 py-3 mt-auto border-t border-[#373a43]"
				>
					<div
						className="flex items-center space-x-3"
						onClick={() => toggleUserMenu()}
					>
						<div className="rounded-full overflow-hidden w-10 h-10 shrink-0">
							{userDetails["picture"] ? (
								<img
									src={userDetails["picture"]}
									alt="User"
									className="w-full h-full object-cover cursor-pointer"
								/>
							) : (
								<div className="bg-[#323541] w-full h-full flex items-center justify-center">
									<IconUser className="w-6 h-6 text-[#9ca3af]" />
								</div>
							)}
						</div>
						<div>
							<p className="text-sm text-white cursor-pointer font-medium">
								{userDetails["given_name"]}
							</p>
							<p className="text-xs text-[#9ca3af]">
								Current Plan:{" "}
								{pricing === "free" ? "Free" : "Pro"}
							</p>
						</div>
					</div>
				</div>
			</div>

			{/* Chevron icon to toggle sidebar visibility */}
			<div
				className="absolute top-0 left-0 bg-matteblack w-[5%] h-full z-10 flex items-center justify-start"
				onMouseEnter={() => setSidebarVisible(true)} // Show sidebar on mouse enter
			>
				<div className="ml-3">
					<IconChevronRight className="text-white w-6 h-6 animate-pulse font-bold" />
					{/* Chevron right icon, white color, size, pulse animation, bold font weight */}
				</div>
			</div>

			{/* Modal dialog for renaming chat */}
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
					confirmButtonColor="bg-[#ff7e5f]"
					confirmButtonBorderColor="border-[#ff7e5f]"
					confirmButtonIcon={IconEdit}
					showInput={true}
				/>
			)}

			{/* Modal dialog for deleting chat */}
			{isDeleteDialogOpen && (
				<ModalDialog
					title="Delete Chat"
					description="Are you sure you want to delete this chat?"
					onCancel={() => setIsDeleteDialogOpen(false)}
					onConfirm={() => handleDeleteChat(currentChat.id)}
					confirmButtonText="Delete"
					confirmButtonColor="bg-red-600"
					confirmButtonBorderColor="border-red-600"
					confirmButtonIcon={IconTrash}
					showInput={false}
				/>
			)}
		</>
	)
}

export default Sidebar
