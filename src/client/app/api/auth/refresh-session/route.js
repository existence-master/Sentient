import { NextResponse } from "next/server"
import { auth0 } from "@lib/auth0"

// This route is only for Auth0 environments
export async function GET(request) {
	if (process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost") {
		return NextResponse.json(
			{
				status: "ok",
				message: "Self-host mode, no refresh needed."
			},
			{
				headers: { "Cache-Control": "no-store, max-age=0" }
			}
		)
	}

	const res = new NextResponse()

	try {
		const session = await auth0.getSession(request, res)
		if (!session) {
			return NextResponse.json(
				{ error: "No session found" },
				{ status: 401 }
			)
		}

		// Force a token refresh to get new claims (like roles)
		await auth0.getAccessToken(request, res, {
			refresh: true
		})

		const newHeaders = new Headers(res.headers)
		newHeaders.set("Cache-Control", "no-store, max-age=0")
		// The new session cookie is now on the `res` object.
		// Return a success response with the new headers.
		return NextResponse.json(
			{ status: "ok" },
			{ status: 200, headers: newHeaders }
		)
	} catch (error) {
		console.error("Error refreshing session in main app:", error.message)
		return NextResponse.json(
			{ error: "Failed to refresh session" },
			{ status: 500 }
		)
	}
}
