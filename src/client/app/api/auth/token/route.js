import { NextResponse } from "next/server"
import { auth0 } from "@lib/auth0"

/**
 * API route to securely get a token for authenticating with backend services.
 * - In Auth0 mode, it gets the user's access token from their session.
 * - In selfhost mode, it returns the static self-host token.
 */
export async function GET() {
	if (process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost") {
		const token = process.env.SELF_HOST_AUTH_TOKEN
		if (!token) {
			return NextResponse.json(
				{ message: "SELF_HOST_AUTH_TOKEN not configured" },
				{ status: 500 }
			)
		}
		return NextResponse.json(
			{ accessToken: token },
			{
				headers: { "Cache-Control": "no-store, max-age=0" }
			}
		)
	}
	try {
		const tokenResult = await auth0.getAccessToken()
		// Handle both `accessToken` and `token` properties for robustness
		const token = tokenResult?.accessToken || tokenResult?.token
		if (!token) {
			return NextResponse.json(
				{ message: "Not authenticated or access token is missing" },
				{ status: 401 }
			)
		}
		return NextResponse.json(
			{ accessToken: token },
			{
				headers: { "Cache-Control": "no-store, max-age=0" }
			}
		)
	} catch (error) {
		console.error("Error in /api/auth/token:", error)
		// If the error is specifically because of a missing session,
		// return a 401 Unauthorized status, which is more appropriate
		// and allows the client to handle it by redirecting to login.
		if (error.code === "missing_session") {
			return NextResponse.json(
				{ message: error.message },
				{ status: 401 }
			)
		}
		return NextResponse.json(
			{ message: "Internal Server Error", error: error.message },
			{ status: 500 }
		)
	}
}
