import { NextResponse } from "next/server"
import { withAuth } from "@lib/api-utils"

const appServerUrl =
	process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost"
		? process.env.INTERNAL_APP_SERVER_URL
		: process.env.NEXT_PUBLIC_APP_SERVER_URL

// GET handler to fetch current privacy filters for a specific service
export const GET = withAuth(async function GET(request, { authHeader }) {
	const { searchParams } = new URL(request.url)
	const serviceName = searchParams.get("service")

	if (!serviceName) {
		return NextResponse.json(
			{ error: "Service name parameter is required" },
			{ status: 400 }
		)
	}

	try {
		const response = await fetch(`${appServerUrl}/api/get-user-data`, {
			method: "POST",
			headers: { "Content-Type": "application/json", ...authHeader }
		})

		if (!response.ok) {
			throw new Error("Failed to fetch user data from backend.")
		}

		const data = await response.json()
		const allFilters = data?.data?.privacyFilters

		let serviceFilters = {}

		// Handle backward compatibility and new structured format
		if (Array.isArray(allFilters)) {
			// Old format: a flat array of keywords. Convert it for the new UI.
			serviceFilters = { keywords: allFilters, emails: [], labels: [] }
		} else if (typeof allFilters === "object" && allFilters !== null) {
			// New format: an object with keys for each service.
			serviceFilters = allFilters[serviceName] || {}
		}

		// Ensure the object has all keys to prevent UI errors
		const finalFilters = {
			keywords: serviceFilters.keywords || [],
			emails: serviceFilters.emails || [],
			labels: serviceFilters.labels || []
		}

		return NextResponse.json(
			{ filters: finalFilters },
			{
				headers: { "Cache-Control": "no-store, max-age=0" }
			}
		)
	} catch (error) {
		console.error(
			`API Error in /settings/privacy-filters?service=${serviceName} (GET):`,
			error
		)
		return NextResponse.json(
			{ error: "Internal Server Error", details: error.message },
			{ status: 500 }
		)
	}
})

// POST handler to update privacy filters for a specific service
export const POST = withAuth(async function POST(request, { authHeader }) {
	try {
		const { service, filters } = await request.json()

		if (
			!service ||
			typeof filters !== "object" ||
			filters === null ||
			!Array.isArray(filters.keywords)
		) {
			return NextResponse.json(
				{
					error: "Invalid payload. 'service' and a 'filters' object with 'keywords' array are required."
				},
				{ status: 400 }
			)
		}

		const backendResponse = await fetch(
			`${appServerUrl}/api/settings/privacy-filters`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader },
				body: JSON.stringify({ service, filters }) // Forward the new structure
			}
		)

		const data = await backendResponse.json()
		if (!backendResponse.ok) {
			throw new Error(data.detail || "Failed to update privacy filters.")
		}
		return NextResponse.json(data)
	} catch (error) {
		console.error("API Error in /settings/privacy-filters (POST):", error)
		return NextResponse.json(
			{ error: "Internal Server Error", details: error.message },
			{ status: 500 }
		)
	}
})
