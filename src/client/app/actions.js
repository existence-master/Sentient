"use server"

import { auth0 } from "@lib/auth0"
import { MongoClient } from "mongodb"
import webpush from "web-push"

// --- DB Connection ---
const MONGO_URI = process.env.MONGO_URI
const MONGO_DB_NAME = process.env.MONGO_DB_NAME

let cachedClient = null
let cachedDb = null

async function connectToDatabase() {
	if (cachedClient && cachedDb) {
		return { client: cachedClient, db: cachedDb }
	}

	if (!MONGO_URI) {
		throw new Error(
			"MONGO_URI is not defined in the environment variables."
		)
	}

	const client = new MongoClient(MONGO_URI)
	await client.connect()
	const db = client.db(MONGO_DB_NAME)

	cachedClient = client
	cachedDb = db

	return { client, db }
}

// --- WebPush Setup ---
if (
	process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY &&
	process.env.VAPID_PRIVATE_KEY &&
	process.env.VAPID_ADMIN_EMAIL
) {
	webpush.setVapidDetails(
		process.env.VAPID_ADMIN_EMAIL,
		process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY,
		process.env.VAPID_PRIVATE_KEY
	)
} else {
	console.warn(
		"VAPID details not fully configured. Push notifications may not work."
	)
}

// --- Server Actions ---

export async function subscribeUser(subscription) {
	const session = await auth0.getSession()
	if (!session?.user) {
		throw new Error("Not authenticated")
	}
	const userId = session.user.sub

	try {
		const { db } = await connectToDatabase()
		const userProfiles = db.collection("user_profiles")

		await userProfiles.updateOne(
			{ user_id: userId },
			{ $addToSet: { "userData.pwa_subscriptions": subscription } },
			{ upsert: true }
		)

		return { success: true }
	} catch (error) {
		console.error("[Actions] Error saving subscription:", error)
		return {
			success: false,
			error: "Failed to save subscription to database."
		}
	}
}

export async function unsubscribeUser(endpoint) {
	const session = await auth0.getSession()
	if (!session?.user) {
		throw new Error("Not authenticated")
	}
	const userId = session.user.sub

	try {
		const { db } = await connectToDatabase()
		const userProfiles = db.collection("user_profiles")

		await userProfiles.updateOne(
			{ user_id: userId },
			{ $pull: { "userData.pwa_subscriptions": { endpoint: endpoint } } }
		)

		return { success: true }
	} catch (error) {
		console.error("[Actions] Error removing subscription:", error)
		return {
			success: false,
			error: "Failed to remove subscription from database."
		}
	}
}

export async function sendNotificationToCurrentUser(payload) {
	const session = await auth0.getSession()
	if (!session?.user) {
		throw new Error("Not authenticated")
	}
	const userId = session.user.sub

	if (
		!process.env.NEXT_PUBLIC_VAPID_PUBLIC_KEY ||
		!process.env.VAPID_PRIVATE_KEY
	) {
		console.error(
			"[Actions] VAPID keys not configured. Cannot send push notification."
		)
		return {
			success: false,
			error: "VAPID keys not configured on the server."
		}
	}

	try {
		const { db } = await connectToDatabase()
		const userProfile = await db
			.collection("user_profiles")
			.findOne(
				{ user_id: userId },
				{ projection: { "userData.pwa_subscriptions": 1 } }
			)

		const subscriptions = userProfile?.userData?.pwa_subscriptions

		if (
			!subscriptions ||
			!Array.isArray(subscriptions) ||
			subscriptions.length === 0
		) {
			return { success: false, error: "No subscription found for user." }
		}

		let successCount = 0
		const promises = subscriptions.map((subscription) => {
			return webpush
				.sendNotification(subscription, JSON.stringify(payload))
				.then(() => {
					successCount++
				})
				.catch(async (error) => {
					console.error(
						`[Actions] Error sending push notification to an endpoint for user ${userId}:`,
						error.statusCode
					)
					if (error.statusCode === 410 || error.statusCode === 404) {
						console.log(
							`[Actions] Subscription for user ${userId} is invalid. Removing from DB.`
						)
						await unsubscribeUser(subscription.endpoint)
					}
				})
		})

		await Promise.all(promises)

		if (successCount > 0) {
			return {
				success: true,
				message: `Sent notifications to ${successCount} of ${subscriptions.length} devices.`
			}
		} else {
			return {
				success: false,
				error: "Failed to send notifications to any device."
			}
		}
	} catch (error) {
		console.error(
			`[Actions] General error sending push notifications to user ${userId}:`,
			error
		)
		return { success: false, error: "A general error occurred." }
	}
}
