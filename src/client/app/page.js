"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import AnimatedLogo from "@components/ui/AnimatedLogo"
import InteractiveNetworkBackground from "@components/ui/InteractiveNetworkBackground"
import React from "react"

/**
 * Home Component - The initial loading/redirect screen of the application.
 *
 * This component now immediately redirects users to the /chat page.
 * All authentication and onboarding checks are handled by middleware and the LayoutWrapper.
 * @returns {React.ReactNode} - The Home component UI.
 */
const Home = () => {
	const router = useRouter()

	useEffect(() => {
		// Immediately redirect to the primary user interface.
		router.replace("/chat")
	}, [router])

	// Show a loading screen while the redirect is happening.
	return (
		<div className="flex-1 min-h-screen flex flex-col items-center justify-center relative overflow-hidden">
			<div className="absolute inset-0 z-[-1] network-grid-background">
				<InteractiveNetworkBackground />
			</div>
			<div className="absolute -top-[250px] left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-brand-orange/10 rounded-full blur-3xl -z-10" />
			<div className="relative z-10 flex flex-col items-center justify-center h-full backdrop-blur-xs">
				<AnimatedLogo />
				<h1 className="text-white text-4xl mt-4">BUYASOUL</h1>
			</div>
		</div>
	)
}

export default Home
