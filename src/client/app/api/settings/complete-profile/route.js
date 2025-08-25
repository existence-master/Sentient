import { NextResponse } from "next/server"
import { withAuth } from "@lib/api-utils"

const appServerUrl =
	process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost"
		? process.env.INTERNAL_APP_SERVER_URL
		: process.env.NEXT_PUBLIC_APP_SERVER_URL

export const POST = withAuth(async function POST(request, { authHeader }) {
	try {
		const body = await request.json()
		const response = await fetch(
			`${appServerUrl}/api/settings/complete-profile`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader },
				body: JSON.stringify(body)
			}
		)

		const data = await response.json()
		if (!response.ok) {
			throw new Error(data.detail || "Failed to complete profile.")
		}
		return NextResponse.json(data)
	} catch (error) {
		console.error("API Error in /settings/complete-profile (POST):", error)
		return NextResponse.json(
			{ error: "Internal Server Error", details: error.message },
			{ status: 500 }
		)
	}
})
