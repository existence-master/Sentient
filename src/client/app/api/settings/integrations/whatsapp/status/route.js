import { NextResponse } from "next/server"
import { withAuth } from "@lib/api-utils"

const appServerUrl =
	process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost"
		? process.env.INTERNAL_APP_SERVER_URL
		: process.env.NEXT_PUBLIC_APP_SERVER_URL

export const GET = withAuth(async function GET(request, { authHeader }) {
	try {
		const response = await fetch(
			`${appServerUrl}/integrations/whatsapp/connect/status`,
			{
				method: "GET",
				headers: { "Content-Type": "application/json", ...authHeader },
				cache: "no-store"
			}
		)

		const data = await response.json()
		if (!response.ok) {
			throw new Error(data.detail || "Failed to get WhatsApp status")
		}
		return NextResponse.json(data, {
			headers: { "Cache-Control": "no-store, max-age=0" }
		})
	} catch (error) {
		console.error("API Error in /whatsapp/connect/status:", error)
		return NextResponse.json({ error: error.message }, { status: 500 })
	}
})
