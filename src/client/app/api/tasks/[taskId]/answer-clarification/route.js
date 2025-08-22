import { NextResponse } from "next/server"
import { withAuth } from "@lib/api-utils"

const appServerUrl =
	process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost"
		? process.env.INTERNAL_APP_SERVER_URL
		: process.env.NEXT_PUBLIC_APP_SERVER_URL

export const POST = withAuth(async function POST(
	request,
	{ params, authHeader }
) {
	const { taskId } = params
	if (!taskId) {
		return NextResponse.json(
			{ error: "Task ID is required" },
			{ status: 400 }
		)
	}

	try {
		const body = await request.json() // { requestId, answer }
		const response = await fetch(
			`${appServerUrl}/tasks/${taskId}/clarifications`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader },
				body: JSON.stringify({
					request_id: body.requestId,
					answer: body.answer
				})
			}
		)

		const data = await response.json()
		if (!response.ok) {
			throw new Error(data.detail || "Failed to submit answer")
		}
		return NextResponse.json(data)
	} catch (error) {
		console.error(
			"API Error in /tasks/[taskId]/answer-clarification:",
			error
		)
		return NextResponse.json({ error: error.message }, { status: 500 })
	}
})
