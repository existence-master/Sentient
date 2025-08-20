import { NextResponse } from "next/server"
import { withAuth } from "@lib/api-utils"

const appServerUrl =
	process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost"
		? process.env.INTERNAL_APP_SERVER_URL
		: process.env.NEXT_PUBLIC_APP_SERVER_URL

// Increase the body size limit for this specific route to handle large chat histories.
// The default is 4MB, which can be exceeded if the context window includes large messages.
export const config = {
	api: {
		bodyParser: {
			sizeLimit: "10mb" // Set to match the Nginx limit
		}
	}
}

export const POST = withAuth(async function POST(request, { authHeader }) {
	try {
		const { messages } = await request.json()

		// Fetch user pricing/credits to pass to the backend
		const pricingResponse = await fetch(
			`${appServerUrl}/api/get-user-data`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader }
			}
		)
		const userData = await pricingResponse.json()
		const pricing = userData?.data?.pricing || "free"
		const credits = userData?.data?.proCredits || 0

		const backendResponse = await fetch(`${appServerUrl}/chat/message`, {
			method: "POST",
			headers: { "Content-Type": "application/json", ...authHeader },
			body: JSON.stringify({
				// Pass `messages` array instead of `input`
				messages,
				pricing, // Note: pricing and credits are still sent for backend logic
				credits
			}),
			// IMPORTANT: duplex must be set to 'half' to stream response body in Next.js Edge/Node runtime
			duplex: "half"
		})

		if (!backendResponse.ok) {
			const errorText = await backendResponse
				.text()
				.catch(() => "Unknown backend error")
			let errorJson = {}
			try {
				errorJson = JSON.parse(errorText)
			} catch (e) {
				// Not a JSON error, use the raw text
			}
			// Return the error from the backend with its original status code
			return NextResponse.json(
				{ detail: errorJson.detail || errorText },
				{ status: backendResponse.status }
			)
		}

		// Return the streaming response directly to the client
		return new Response(backendResponse.body, {
			status: 200,
			headers: {
				// Use application/x-ndjson for newline-delimited JSON streams
				"Content-Type": "application/x-ndjson",
				"Cache-Control": "no-cache",
				Connection: "keep-alive",
				"X-Accel-Buffering": "no", // Disable buffering on Netlify/Vercel
				"Content-Encoding": "identity"
			}
		})
	} catch (error) {
		console.error("API Error in /chat/message:", error)
		return NextResponse.json(
			{ detail: "Internal Server Error", error: error.message },
			{ status: 500 }
		)
	}
})
