import { NextResponse } from "next/server"
import { withAuth } from "@lib/api-utils"

const appServerUrl =
	process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost"
		? process.env.INTERNAL_APP_SERVER_URL
		: process.env.NEXT_PUBLIC_APP_SERVER_URL

export const GET = withAuth(async function GET(request, { authHeader }) {
	const backendUrl = new URL(`${appServerUrl}/memories`)

	try {
		const response = await fetch(backendUrl.toString(), {
			method: "GET",
			headers: { "Content-Type": "application/json", ...authHeader },
			cache: "no-store" // Prevent Next.js from caching this server-side fetch
		})

		const data = await response.json()
		if (!response.ok) {
			throw new Error(data.detail || "Failed to fetch memories")
		}
		return NextResponse.json(data, {
			headers: { "Cache-Control": "no-store, max-age=0" }
		})
	} catch (error) {
		console.error("API Error in /memories:", error)
		return NextResponse.json({ error: error.message }, { status: 500 })
	}
})

export const POST = withAuth(async function POST(request, { authHeader }) {
	const backendUrl = new URL(`${appServerUrl}/memories`)
	try {
		const body = await request.json()
		const response = await fetch(backendUrl.toString(), {
			method: "POST",
			headers: { "Content-Type": "application/json", ...authHeader },
			body: JSON.stringify(body)
		})

		const data = await response.json()
		if (!response.ok) {
			return NextResponse.json(
				{ error: data.detail || "Failed to create memory" },
				{ status: response.status }
			)
		}
		return NextResponse.json(data, { status: response.status })
	} catch (error) {
		console.error("API Error in /memories (POST):", error)
		return NextResponse.json({ error: error.message }, { status: 500 })
	}
})
