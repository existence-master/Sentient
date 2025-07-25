"use client"

import posthog from "posthog-js"
import { PostHogProvider as PHProvider } from "posthog-js/react"
import { useEffect } from "react"

export function PostHogProvider({ children }) {
	useEffect(() => {
		if (process.env.NEXT_PUBLIC_POSTHOG_KEY) {
			posthog.init(process.env.NEXT_PUBLIC_POSTHOG_KEY, {
				api_host: "/ingest",
				ui_host:
					process.env.NEXT_PUBLIC_POSTHOG_HOST ||
					"https://us.posthog.com",
				defaults: "2025-05-24",
				capture_exceptions: true,
				debug: process.env.NODE_ENV === "development"
			})
		}
	}, [])

	if (!process.env.NEXT_PUBLIC_POSTHOG_KEY) return <>{children}</>
	return <PHProvider client={posthog}>{children}</PHProvider>
}
