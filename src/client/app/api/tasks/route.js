// src/client/app/api/tasks/route.js
import { NextResponse } from "next/server"
import { withAuth } from "@lib/api-utils"

const appServerUrl =
	process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost"
		? process.env.INTERNAL_APP_SERVER_URL
		: process.env.NEXT_PUBLIC_APP_SERVER_URL

export const POST = withAuth(async function POST(request, { authHeader }) {
	try {
		// The backend endpoint is a POST, but it doesn't require a body.
		// We use POST here to align with the backend's expectation.
		const response = await fetch(`${appServerUrl}/tasks/fetch-tasks`, {
			method: "POST",
			headers: { "Content-Type": "application/json", ...authHeader }
		})

		if (!response.ok) {
			const errorText = await response.text()
			let detail = "Failed to fetch tasks"
			try {
				const errorJson = JSON.parse(errorText)
				detail = errorJson.detail || errorJson.error || errorText
			} catch (e) {}
			return NextResponse.json(
				{ error: detail },
				{ status: response.status }
			)
		}
		const data = await response.json()
		return NextResponse.json(data, {
			headers: { "Cache-Control": "no-store, max-age=0" }
		})
	} catch (error) {
		console.error("API Error in /tasks:", error)
		return NextResponse.json({ error: error.message }, { status: 500 })
	}
})
