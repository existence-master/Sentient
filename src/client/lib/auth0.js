import { Auth0Client } from "@auth0/nextjs-auth0/server"

const isSelfHost = process.env.NEXT_PUBLIC_ENVIRONMENT === "selfhost"

/**
 * v4 reads AUTH0_DOMAIN (tenant host, no protocol) — not AUTH0_ISSUER_BASE_URL.
 * Support issuer URL in env for teams that only set that.
 */
function normalizeAuth0Domain() {
	let d = process.env.AUTH0_DOMAIN?.trim()
	if (d) {
		return d.replace(/^https?:\/\//, "").split("/")[0]
	}
	const issuer = process.env.AUTH0_ISSUER_BASE_URL?.trim()
	if (!issuer) return undefined
	try {
		return new URL(issuer).hostname
	} catch {
		return undefined
	}
}

/**
 * SDK calls `new URL(base)`; values like "localhost:3000" without protocol throw Invalid URL.
 */
function resolveAppBaseUrl() {
	let raw =
		process.env.APP_BASE_URL?.trim() ||
		process.env.NEXT_PUBLIC_APP_BASE_URL?.trim() ||
		"http://localhost:3000"
	if (!/^https?:\/\//i.test(raw)) {
		raw = `http://${raw}`
	}
	try {
		return new URL(raw).origin
	} catch {
		console.warn(
			"[auth0] Invalid APP_BASE_URL / NEXT_PUBLIC_APP_BASE_URL; using http://localhost:3000"
		)
		return "http://localhost:3000"
	}
}

const resolvedAuth0Domain = normalizeAuth0Domain()

// SDK crashes if domain is undefined (issuer getter uses this.domain.startsWith)
export const auth0 =
	isSelfHost || !resolvedAuth0Domain
		? null
		: new Auth0Client({
				domain: resolvedAuth0Domain,
				clientId: process.env.AUTH0_CLIENT_ID?.trim(),
				clientSecret: process.env.AUTH0_CLIENT_SECRET?.trim(),
				appBaseUrl: resolveAppBaseUrl(),
				secret: process.env.AUTH0_SECRET?.trim(),
				authorizationParameters: {
					scope: process.env.AUTH0_SCOPE,
					audience: process.env.AUTH0_AUDIENCE
				},
				async beforeSessionSaved(session, idToken) {
					return session
				}
			})

/** False when using Auth0 mode but AUTH0_DOMAIN / issuer URL is missing */
export const isAuth0ClientReady = Boolean(auth0)

export async function getBackendAuthHeader() {
	if (isSelfHost) {
		const staticToken = process.env.SELF_HOST_AUTH_TOKEN
		if (!staticToken) {
			console.error("SELF_HOST_AUTH_TOKEN is not set for selfhost mode.")
			return null
		}
		return { Authorization: `Bearer ${staticToken}` }
	}

	if (!auth0) {
		console.error(
			"[auth0] Client not initialized. Set AUTH0_DOMAIN (or AUTH0_ISSUER_BASE_URL) and Auth0 app credentials in .env.local, or use NEXT_PUBLIC_ENVIRONMENT=selfhost."
		)
		return null
	}

	try {
		const tokenResult = await auth0.getAccessToken()
		const accessToken = tokenResult?.accessToken || tokenResult?.token

		if (!accessToken) {
			console.error(
				"lib/auth: Cannot create backend auth header, access token is missing."
			)
			return null
		}
		return { Authorization: `Bearer ${accessToken}` }
	} catch (error) {
		console.error("Error getting access token for backend header:", error)
		return null
	}
}
