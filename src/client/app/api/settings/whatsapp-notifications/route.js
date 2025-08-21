import { NextResponse } from "next/server"
import { withAuth } from "@lib/api-utils"

const appServerUrl =
	process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost"
		? process.env.INTERNAL_APP_SERVER_URL
		: process.env.NEXT_PUBLIC_APP_SERVER_URL

// Handler for updating or setting the WhatsApp number for notifications
export const POST = withAuth(async function POST(request, { authHeader }) {
	try {
		const body = await request.json() // { whatsapp_notifications_number: "..." or "" }
		// When a number is saved, notifications are implicitly enabled.
		// When it's removed, they are disabled.
		const isEnabling =
			body.whatsapp_notifications_number &&
			body.whatsapp_notifications_number.trim() !== ""

		const response = await fetch(
			`${appServerUrl}/api/settings/whatsapp-notifications`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader },
				body: JSON.stringify({ ...body, enabled: isEnabling })
			}
		)

		const data = await response.json()
		if (!response.ok) {
			throw new Error(
				data.detail || "Failed to update WhatsApp notification number"
			)
		}
		return NextResponse.json(data)
	} catch (error) {
		console.error(
			"API Error in /api/settings/whatsapp-notifications (POST):",
			error
		)
		return NextResponse.json({ error: error.message }, { status: 500 })
	}
})

// Handler for getting the current WhatsApp notification settings
export const GET = withAuth(async function GET(request, { authHeader }) {
	try {
		const response = await fetch(
			`${appServerUrl}/api/settings/whatsapp-notifications`,
			{
				method: "GET",
				headers: { "Content-Type": "application/json", ...authHeader }
			}
		)
		const data = await response.json()
		if (!response.ok) {
			throw new Error(
				data.detail || "Failed to fetch WhatsApp notification settings"
			)
		}
		return NextResponse.json(data, {
			headers: { "Cache-Control": "no-store, max-age=0" }
		})
	} catch (error) {
		console.error(
			"API Error in /api/settings/whatsapp-notifications (GET):",
			error
		)
		return NextResponse.json({ error: error.message }, { status: 500 })
	}
})
