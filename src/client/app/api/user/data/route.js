// src/client/app/api/user/data/route.js
import { NextResponse } from "next/server"
import { withAuth } from "@lib/api-utils"

const appServerUrl =
	process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost"
		? process.env.INTERNAL_APP_SERVER_URL
		: process.env.NEXT_PUBLIC_APP_SERVER_URL

export const GET = withAuth(async function GET(request, { authHeader }) {
	try {
		const response = await fetch(`${appServerUrl}/api/get-user-data`, {
			method: "POST",
			headers: { "Content-Type": "application/json", ...authHeader },
			cache: "no-store" // Prevent Next.js from caching this server-side fetch
		})

		if (!response.ok) {
			const data = await response.json()
			throw new Error(
				data.message || "Failed to fetch user data from backend"
			)
		}
		const data = await response.json()
		return NextResponse.json(data)
	} catch (error) {
		console.error("API Error in /user/data:", error)
		return NextResponse.json(
			{ message: "Internal Server Error", error: error.message },
			{ status: 500 }
		)
	}
})
