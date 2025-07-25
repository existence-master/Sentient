// src/client/app/api/tasks/route.js
import { NextResponse } from "next/server"
import { withAuth } from "@lib/api-utils"

const appServerUrl =
	process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost"
		? process.env.INTERNAL_APP_SERVER_URL
		: process.env.NEXT_PUBLIC_APP_SERVER_URL

export const GET = withAuth(async function GET(request, { authHeader }) {
	try {
		const response = await fetch(`${appServerUrl}/agents/fetch-tasks`, {
			method: "POST",
			headers: { "Content-Type": "application/json", ...authHeader }
		})

		const data = await response.json()
		if (!response.ok) {
			throw new Error(data.error || "Failed to fetch tasks")
		}
		return NextResponse.json(data)
	} catch (error) {
		console.error("API Error in /tasks:", error)
		return NextResponse.json({ error: error.message }, { status: 500 })
	}
})
