// src/client/app/api/integrations/connected/route.js
import { NextResponse } from "next/server"
import { withAuth } from "@lib/api-utils"

const appServerUrl =
	process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost"
		? process.env.INTERNAL_APP_SERVER_URL
		: process.env.NEXT_PUBLIC_APP_SERVER_URL

export const GET = withAuth(async function GET(request, { authHeader }) {
	try {
		// This reuses the same backend endpoint but we will filter on the client
		const response = await fetch(`${appServerUrl}/integrations/sources`, {
			method: "GET",
			headers: { "Content-Type": "application/json", ...authHeader },
			cache: "no-store" // Prevent Next.js from caching this server-side fetch
		})

		const data = await response.json()
		if (!response.ok) {
			throw new Error(data.detail || "Failed to fetch integrations")
		}
		return NextResponse.json(data, {
			headers: { "Cache-Control": "no-store, max-age=0" }
		})
	} catch (error) {
		console.error("API Error in /integrations/connected:", error)
		return NextResponse.json({ error: error.message }, { status: 500 })
	}
})
