"use client"
import { useRouter } from "next/navigation"
import { useEffect, useState, React } from "react"
import {
	IconChevronRight,
	IconUser,
	IconLogout,
	IconTemplate,
	IconAdjustments,
	IconMessage,
	IconChecklist
} from "@tabler/icons-react"
import toast from "react-hot-toast"

const Sidebar = ({ userDetails, setSidebarVisible, isSidebarVisible }) => {
	const router = useRouter()
	const [pricing, setPricing] = useState("free")

	const toggleUserMenu = () => {
		const userMenu = document.getElementById("user-menu")
		if (userMenu) userMenu.classList.toggle("hidden")
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
		fetchPricingPlan()
	}, [])

	return (
		<>
			<div
				id="sidebar"
				className={`w-1/5 h-full flex flex-col bg-smokeblack overflow-y-auto transform transition-all duration-300 ${isSidebarVisible ? "translate-x-0 opacity-100 z-40 pointer-events-auto" : "-translate-x-full opacity-0 z-0 pointer-events-none"}`}
				onMouseLeave={() => setSidebarVisible(false)}
			>
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
				<div className="flex flex-col px-4 pt-4 pb-8 flex-grow">
					<button
						onClick={() => router.push("/chat")}
						className="cursor-pointer flex items-center gap-3 w-full text-left px-4 py-2 rounded-lg text-white hover:text-lightblue hover:bg-neutral-800 mt-1"
					>
						<IconMessage className="w-5 h-5" />
						<span className="text-base text-white">Chat</span>
					</button>
					<button
						onClick={() => router.push("/profile")}
						className="cursor-pointer flex items-center gap-3 w-full text-left px-4 py-2 rounded-lg text-white hover:text-lightblue hover:bg-neutral-800 mt-1"
					>
						<IconTemplate className="w-5 h-5" />
						<span className="text-base text-white">Profile</span>
					</button>
					<button
						onClick={() => router.push("/settings")}
						className="cursor-pointer flex items-center gap-3 w-full text-left px-4 py-2 rounded-lg text-white hover:text-lightblue hover:bg-neutral-800 mt-1"
					>
						<IconAdjustments className="w-5 h-5" />
						<span className="text-base text-white">Settings</span>
					</button>
					<button
						onClick={() => router.push("/tasks")}
						className="cursor-pointer flex items-center gap-3 w-full text-left px-4 py-2 rounded-lg text-white hover:text-lightblue hover:bg-neutral-800 mt-1"
					>
						<IconChecklist className="w-5 h-5" />
						<span className="text-base text-white">Tasks</span>
					</button>
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
					<button
						onClick={logout}
						className="flex items-center justify-between px-6 py-2 text-[#9ca3af] hover:text-white"
					>
						<span className="text-base">Log out</span>
						<IconLogout className="w-5 h-5" />
					</button>
				</div>
				<div
					id="user-profile"
					className="px-6 py-3 mt-auto border-t border-[#373a43]"
				>
					<div
						className="flex items-center space-x-3"
						onClick={toggleUserMenu}
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
			<div
				className="absolute top-0 left-0 bg-matteblack w-[5%] h-full z-10 flex items-center justify-start"
				onMouseEnter={() => setSidebarVisible(true)}
			>
				<div className="ml-3">
					<IconChevronRight className="text-white w-6 h-6 animate-pulse font-bold" />
				</div>
			</div>
		</>
	)
}

export default Sidebar
