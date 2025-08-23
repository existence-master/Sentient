"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { useUser } from "@auth0/nextjs-auth0"
import AnimatedLogo from "@components/AnimatedLogo"
import toast from "react-hot-toast"
import InteractiveNetworkBackground from "@components/ui/InteractiveNetworkBackground"
import React from "react"

/**
 * Home Component - The initial loading screen of the application.
 *
 * This component is responsible for checking the user's authentication and onboarding status,
 * then redirecting them to the appropriate page (/auth/login, /onboarding, or /chat).
 * It displays a loading animation during this process.
 * The final destination for an authenticated and onboarded user is /chat.
 * @returns {React.ReactNode} - The Home component UI.
 */
const Home = () => {
	const router = useRouter()
	const isSelfHost = process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost"
	const {
		user,
		error: authError,
		isLoading: isAuthLoading
	} = useUser({
		disabled: isSelfHost // Disable the hook in self-host mode
	})

	const [onboarded, setOnboarded] = useState(null)
	const [isCheckingOnboarding, setIsCheckingOnboarding] = useState(false)

	useEffect(() => {
		// If we reach here, the user is authenticated. Now, check their onboarding status.
		const checkOnboardingStatus = async () => {
			setIsCheckingOnboarding(true)
			try {
				const response = await fetch("/api/user/data") // Uses the same API endpoint

				if (response.status === 401) {
					// This case is unlikely if useUser() works but is good to handle.
					toast.error("Session expired. Please log in again.")
					router.push("/auth/logout")
					return
				}

				if (!response.ok) {
					const errorData = await response.json()
					throw new Error(
						errorData.message || "Failed to fetch user data"
					)
				}

				const result = await response.json()
				const onboardingCompleted =
					result?.data?.onboardingComplete ?? false
				setOnboarded(onboardingCompleted)
			} catch (error) {
				toast.error(`Error initializing app: ${error.message}`)
				// Default to sending to onboarding on error for safety.
				setOnboarded(false)
			} finally {
				setIsCheckingOnboarding(false)
			}
		}

		if (isSelfHost) {
			// In self-host mode, we bypass Auth0 checks and go straight to onboarding check.
			if (onboarded === null && !isCheckingOnboarding) {
				checkOnboardingStatus()
			}
			return
		}

		if (isAuthLoading) {
			// Still waiting for Auth0 to determine the user's session.
			return
		}

		if (authError) {
			console.error("Auth0 authentication error:", authError)
			toast.error("Your session has expired. Redirecting to login.")
			// Redirect to login to re-attempt authentication.
			router.push("/auth/login")
			return
		}

		if (!user) {
			// User is not authenticated. Redirect to the Auth0 login page.
			router.push("/auth/login")
			return
		}

		// Only check onboarding if it hasn't been determined yet.
		if (onboarded === null) {
			checkOnboardingStatus()
		}
	}, [isAuthLoading, user, authError, router, onboarded, isSelfHost])

	// This effect handles the redirection after onboarding status is known.
	useEffect(() => {
		if (isSelfHost) {
			// In self-host mode, user is always "present", just need onboarding status.
			if (onboarded !== null) {
				if (onboarded) {
					router.push("/chat")
				} else {
					router.push("/onboarding")
				}
			}
			return
		}

		// Only redirect if auth is loaded, user exists, and onboarding status is determined.
		if (!isAuthLoading && user && onboarded !== null) {
			if (onboarded) {
				router.push("/chat")
			} else {
				router.push("/onboarding")
			}
		}
	}, [isAuthLoading, user, onboarded, router, isSelfHost])

	// Show a loading screen while checking authentication or onboarding status.
	return (
		<div className="flex-1 min-h-screen flex flex-col items-center justify-center relative overflow-hidden">
			<div className="absolute inset-0 z-[-1] network-grid-background">
				<InteractiveNetworkBackground />
			</div>
			<div className="absolute -top-[250px] left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-brand-orange/10 rounded-full blur-3xl -z-10" />
			<div className="relative z-10 flex flex-col items-center justify-center h-full backdrop-blur-xs">
				<AnimatedLogo />
				<h1 className="text-white text-4xl mt-4">Sentient</h1>
			</div>
		</div>
	)
}

export default Home
