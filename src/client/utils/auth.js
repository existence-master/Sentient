/**
 * @file auth.js
 * @description Manages authentication, token storage, and user-specific data related to
 * authorization and application features (like pricing, credits, beta status) for the Electron application.
 * It interacts with Auth0 for authentication, Keytar for secure local storage of tokens and user data,
 * and a backend server for encryption/decryption services and fetching user metadata.
 *
 * Key functionalities:
 * - Provides URLs for Auth0 authentication and logout.
 * - Handles OAuth 2.0 token exchange (authorization code and refresh token flows).
 * - Securely stores refresh tokens and other sensitive user data using Keytar, encrypted via a backend service.
 * - Manages in-memory state for access tokens and user profiles.
 * - Provides getters for accessing the current access token and user profile.
 * - Implements logout functionality, clearing local and in-memory session data.
 * - Fetches and stores user-specific data like pricing tier, referral status, beta status, and daily credits
 *   using Keytar, scoped by user ID.
 * - Ensures that user-specific data is stored and retrieved using a consistent `userId_keyName` convention in Keytar.
 */

import { jwtDecode } from "jwt-decode"
import dotenv from "dotenv"
import keytar from "keytar"
import os from "os"
import fetch from "node-fetch"
import path, { dirname } from "path"
import { fileURLToPath } from "url"
import { app } from "electron" // Electron's app module for path utilities

// --- Constants and Path Configurations ---
const isWindows = process.platform === "win32"
const __filename = fileURLToPath(import.meta.url) // Current file's absolute path
const __dirname = dirname(__filename) // Current file's directory name
let dotenvPath // Path to the .env file, differs for dev and packaged app

// Determine .env file path based on application packaging status
if (app.isPackaged) {
	// Production: Packaged application
	dotenvPath = isWindows
		? path.join(process.resourcesPath, ".env") // Windows: .env in resources folder
		: path.join(app.getPath("home"), ".sentient.env") // Linux/macOS: User-specific .env
	console.log(
		`[auth.js] PRODUCTION: Loading .env from packaged path: ${dotenvPath}`
	)
} else {
	// Development: Running from source
	dotenvPath = path.resolve(__dirname, "../.env") // Relative to this file, up one level to project root's src, then one more to root .env
	console.log(
		`[auth.js] DEVELOPMENT: Loading .env from dev path: ${dotenvPath}`
	)
}

// Load environment variables from the determined .env file
const dotenvResult = dotenv.config({ path: dotenvPath })
if (dotenvResult.error) {
	console.error(
		`[auth.js] ERROR: Failed to load .env file from ${dotenvPath}. Details:`,
		dotenvResult.error
	)
	// Consider whether to throw an error or proceed with defaults if critical env vars are missing.
}

// --- Configuration Variables ---
const auth0Domain = process.env.AUTH0_DOMAIN // Auth0 tenant domain
const clientId = process.env.AUTH0_CLIENT_ID // Auth0 application client ID
const appServerUrl = process.env.APP_SERVER_URL || "http://localhost:5000" // Backend server URL
const auth0Audience = process.env.AUTH0_AUDIENCE // THIS IS NOW YOUR CUSTOM API AUDIENCE

// --- Keytar Configuration ---
const keytarService = "electron-openid-oauth" // Main service name for Keytar entries
const keytarAccountRefreshToken = os.userInfo().username

// --- In-Memory State ---
let accessToken = null
let profile = null // Decoded ID token (contains user profile info like 'sub', 'name', 'email', and custom claims)
// Note: Refresh token is NOT stored in memory.

// --- Getters for In-Memory State ---
export function getAccessToken() {
	return accessToken
}

export function getProfile() {
	return profile
}

// --- Auth0 URLs ---
export function getAuthenticationURL() {
	if (!auth0Domain || !clientId) {
		console.error(
			"[auth.js] Auth0 domain or client ID is missing in environment variables."
		)
		throw new Error("Auth0 domain and/or client ID are not configured.")
	}

	if (!auth0Audience) {
		console.error(
			"[auth.js] Auth0 Audience (AUTH0_AUDIENCE) for custom API is missing. This is crucial for getting an API-specific access token."
		)
		throw new Error(
			"Auth0 Custom API Audience is not configured. Please set AUTH0_AUDIENCE in your .env file."
		)
	}

	// Define the scopes (permissions) your Electron app needs from the custom API
	// These should match the permissions you defined in Auth0 for your API
	// and those that your Electron client application is authorized to request.
	const apiScopes = [
		"openid", // Standard OIDC scope, always required
		"profile", // To get user profile information in the ID token
		"email", // To get user's email in the ID token
		"offline_access", // To request a refresh token for long-lived sessions

		// Scopes for your custom "Sentient App API"
		// Review this list and include only what your Electron app will eventually need to enable features for.
		"read:chat",
		"write:chat",
		"use:elaborator",
		"read:profile", // For client-side display of some profile aspects or initial data load
		"write:profile", // If client directly initiates profile updates (though often done via specific actions)
		"scrape:linkedin",
		"scrape:reddit",
		"scrape:twitter",
		"manage:google_auth", // If client triggers Google auth flow directly
		"read:memory",
		"write:memory",
		"read:tasks",
		"write:tasks",
		"read:notifications",
		"read:config",
		"write:config",
		"admin:user_metadata" // Be cautious with requesting admin-level scopes directly for end-user tokens.
		// Typically, admin actions are performed by a backend with its own M2M token.
		// However, if the user IS an admin and the app has admin features, this might be applicable.
	].join(" ")

	// Construct the authorization URL
	// response_type=code: For Authorization Code Grant flow
	// redirect_uri: Where Auth0 redirects after authentication. Must be in "Allowed Callback URLs".
	// audience: Your custom API identifier.
	// scope: The permissions your app is requesting.
	return `https://${auth0Domain}/authorize?audience=${encodeURIComponent(
		auth0Audience
	)}&scope=${encodeURIComponent(
		apiScopes
	)}&response_type=code&client_id=${clientId}&redirect_uri=http://localhost/callback`
}

export function getLogOutUrl() {
	if (!auth0Domain || !clientId) {
		console.error(
			"[auth.js] Auth0 domain or client ID is missing for logout URL."
		)
		throw new Error("Auth0 domain/client ID not configured for logout.")
	}
	// returnTo: URL to redirect to after logout (must be in "Allowed Logout URLs" in Auth0 app settings)
	return `https://${auth0Domain}/v2/logout?client_id=${clientId}&returnTo=http://localhost/logout`
}

// --- Encryption/Decryption Service Interaction (Backend) ---
async function encrypt(data) {
	try {
		const response = await fetch(`${appServerUrl}/encrypt`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ data })
		})
		if (!response.ok) {
			const errorText = await response.text()
			throw new Error(
				`Encryption service failed: ${response.statusText}. Details: ${errorText}`
			)
		}
		const result = await response.json()
		if (!result.encrypted_data) {
			throw new Error(
				"Encryption service response missing encrypted_data."
			)
		}
		return result.encrypted_data
	} catch (error) {
		console.error(`[auth.js] Error during encryption: ${error.message}`)
		throw error
	}
}

async function decrypt(encryptedData) {
	try {
		const response = await fetch(`${appServerUrl}/decrypt`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ encrypted_data: encryptedData })
		})
		if (!response.ok) {
			const errorText = await response.text()
			throw new Error(
				`Decryption service failed: ${response.statusText}. Details: ${errorText}`
			)
		}
		const result = await response.json()
		if (typeof result.decrypted_data === "undefined") {
			throw new Error(
				"Decryption service response missing decrypted_data."
			)
		}
		return result.decrypted_data
	} catch (error) {
		console.error(`[auth.js] Error during decryption: ${error.message}`)
		throw error
	}
}

// --- Token Management Functions ---
export async function refreshTokens() {
	console.log("[auth.js] Attempting to refresh tokens...")
	const savedEncryptedRefreshToken = await keytar.getPassword(
		keytarService,
		keytarAccountRefreshToken
	)

	if (!savedEncryptedRefreshToken) {
		console.warn(
			"[auth.js] No refresh token in Keytar. User needs to log in."
		)
		await logout() // Ensure clean state
		throw new Error("No refresh token available.")
	}

	if (!auth0Audience) {
		// Crucial for custom API
		console.error("[auth.js] Auth0 Audience missing for refreshTokens.")
		await logout()
		throw new Error(
			"Auth0 Audience configuration missing for token refresh."
		)
	}

	let decryptedToken
	try {
		decryptedToken = await decrypt(savedEncryptedRefreshToken)
	} catch (decryptError) {
		console.error(
			"[auth.js] Failed to decrypt stored refresh token:",
			decryptError.message
		)
		await logout()
		throw new Error("Failed to decrypt refresh token. Please log in again.")
	}

	const refreshOptions = {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({
			grant_type: "refresh_token",
			client_id: clientId,
			refresh_token: decryptedToken,
			audience: auth0Audience // Request token for your custom API
			// Scopes are typically not re-requested here; they are bound to the refresh token grant
		})
	}

	try {
		const response = await fetch(
			`https://${auth0Domain}/oauth/token`,
			refreshOptions
		)
		const data = await response.json()

		if (!response.ok || data.error) {
			console.error(
				"[auth.js] Auth0 token refresh failed:",
				data.error_description || data.error || response.statusText
			)
			await logout()
			throw new Error(
				`Token refresh failed: ${
					data.error_description || data.error || "Unknown reason"
				}`
			)
		}

		if (!data.access_token || !data.id_token) {
			console.error(
				"[auth.js] Refresh response missing access_token or id_token."
			)
			await logout()
			throw new Error(
				"Incomplete token response from Auth0 during refresh."
			)
		}

		accessToken = data.access_token
		console.log(
			"[auth.js] Auth0 /oauth/token response (loadTokens):",
			JSON.stringify(data, null, 2)
		)
		try {
			const decodedAccessToken = jwtDecode(accessToken) // Use jwt-decode
			console.log(
				"[auth.js] Decoded Access Token PAYLOAD (loadTokens):",
				decodedAccessToken
			)
			// Specifically log the permissions claim
			if (decodedAccessToken.permissions) {
				console.log(
					"[auth.js] Permissions in Access Token (loadTokens):",
					decodedAccessToken.permissions
				)
			} else {
				console.warn(
					"[auth.js] 'permissions' claim MISSING in Access Token (loadTokens)."
				)
			}
		} catch (e) {
			console.error(
				"[auth.js] Failed to decode access token for payload inspection (loadTokens).",
				e
			)
		}
		// Make sure profile is also decoded from data.id_token
		profile = jwtDecode(data.id_token)
		console.log("[auth.js] Decoded ID Token PAYLOAD (loadTokens):", profile)

		// Log received tokens for debugging (sensitive in production logs)
		// console.log("[auth.js] RAW ACCESS TOKEN received (refresh):", data.access_token);
		try {
			const decodedHeader = jwtDecode(data.access_token, { header: true })
			console.log(
				"[auth.js] Decoded Access Token Header (refresh):",
				decodedHeader
			)
			// Example of accessing custom claims if added by Auth0 Action
			// const customRole = profile[`${auth0Audience}/role`]; // Use your actual namespace
			// console.log("[auth.js] Custom role from refreshed ID token:", customRole);
		} catch (e) {
			console.error(
				"[auth.js] Failed to decode access token header for inspection (refresh)."
			)
		}

		if (data.refresh_token && data.refresh_token !== decryptedToken) {
			console.log(
				"[auth.js] Auth0 returned a new refresh token. Updating in Keytar."
			)
			const encryptedNewRefreshToken = await encrypt(data.refresh_token)
			await keytar.setPassword(
				keytarService,
				keytarAccountRefreshToken,
				encryptedNewRefreshToken
			)
		}

		console.log(
			`[auth.js] Tokens refreshed successfully for user: ${profile?.sub}`
		)
		if (profile?.sub) {
			await setCheckinInKeytar(profile.sub)
		}
	} catch (error) {
		console.error(
			"[auth.js] Unexpected error during token refresh:",
			error.message
		)
		if (!error.message.includes("Token refresh failed")) await logout() // Avoid double logout
		throw error
	}
}

export async function loadTokens(callbackURL) {
	console.log("[auth.js] Loading tokens from authorization code...")
	const url = new URL(callbackURL)
	const code = url.searchParams.get("code")

	if (!code) {
		console.error("[auth.js] Authorization code missing in callback URL.")
		throw new Error("Authorization code missing.")
	}

	if (!auth0Audience) {
		// Crucial for custom API
		console.error("[auth.js] Auth0 Audience missing for loadTokens.")
		await logout() // Clean up before erroring
		throw new Error(
			"Auth0 Audience configuration missing for token exchange."
		)
	}

	const params = {
		grant_type: "authorization_code",
		client_id: clientId,
		code: code,
		redirect_uri: "http://localhost/callback", // Must match redirect_uri in /authorize
		audience: auth0Audience // Request token for your custom API
	}
	const options = {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(params)
	}

	try {
		const response = await fetch(
			`https://${auth0Domain}/oauth/token`,
			options
		)
		const data = await response.json()

		if (!response.ok || data.error) {
			console.error(
				"[auth.js] Auth0 token exchange failed:",
				data.error_description || data.error || response.statusText
			)
			await logout()
			throw new Error(
				`Token exchange failed: ${
					data.error_description || data.error || "Unknown reason"
				}`
			)
		}

		if (!data.access_token || !data.id_token || !data.refresh_token) {
			console.error(
				"[auth.js] Token exchange response missing required tokens."
			)
			await logout()
			throw new Error("Incomplete token response from Auth0 after login.")
		}

		accessToken = data.access_token
		console.log(
			"[auth.js] Auth0 /oauth/token response (loadTokens):",
			JSON.stringify(data, null, 2)
		)
		try {
			const decodedAccessToken = jwtDecode(accessToken) // Use jwt-decode
			console.log(
				"[auth.js] Decoded Access Token PAYLOAD (loadTokens):",
				decodedAccessToken
			)
			// Specifically log the permissions claim
			if (decodedAccessToken.permissions) {
				console.log(
					"[auth.js] Permissions in Access Token (loadTokens):",
					decodedAccessToken.permissions
				)
			} else {
				console.warn(
					"[auth.js] 'permissions' claim MISSING in Access Token (loadTokens)."
				)
			}
		} catch (e) {
			console.error(
				"[auth.js] Failed to decode access token for payload inspection (loadTokens).",
				e
			)
		}
		// Make sure profile is also decoded from data.id_token
		profile = jwtDecode(data.id_token)
		console.log("[auth.js] Decoded ID Token PAYLOAD (loadTokens):", profile)
		const newRefreshToken = data.refresh_token

		// Log received tokens for debugging (sensitive in production logs)
		// console.log("[auth.js] RAW ACCESS TOKEN received (loadTokens):", data.access_token);
		try {
			const decodedHeader = jwtDecode(data.access_token, { header: true })
			console.log(
				"[auth.js] Decoded Access Token Header (loadTokens):",
				decodedHeader
			)
			// Example of accessing custom claims if added by Auth0 Action
			// const customRole = profile[`${auth0Audience}/role`]; // Use your actual namespace
			// console.log("[auth.js] Custom role from new ID token:", customRole);
		} catch (e) {
			console.error(
				"[auth.js] Failed to decode access token header for inspection (loadTokens)."
			)
		}

		if (!profile || !profile.sub) {
			console.error("[auth.js] Invalid ID token or 'sub' missing.")
			await logout()
			throw new Error("Invalid ID token. User ID not determined.")
		}
		const userId = profile.sub
		console.log(`[auth.js] Tokens loaded successfully for user: ${userId}`)

		const encryptedNewRefreshToken = await encrypt(newRefreshToken)
		await keytar.setPassword(
			keytarService,
			keytarAccountRefreshToken,
			encryptedNewRefreshToken
		)
		console.log("[auth.js] New refresh token stored in Keytar.")

		await setCheckinInKeytar(userId)
	} catch (error) {
		console.error("[auth.js] Error loading tokens:", error.message)
		await logout()
		throw error
	}
}

export async function logout() {
	console.log("[auth.js] Performing user logout...")
	const userIdBeforeClear = profile?.sub

	try {
		await keytar.deletePassword(keytarService, keytarAccountRefreshToken)
		console.log(
			`[auth.js] Deleted refresh token from Keytar for OS user: '${keytarAccountRefreshToken}'.`
		)
	} catch (error) {
		console.error(
			`[auth.js] Error deleting refresh token for OS user '${keytarAccountRefreshToken}':`,
			error.message
		)
	}

	if (userIdBeforeClear) {
		console.log(
			`[auth.js] Deleting Keytar entries for user ID: '${userIdBeforeClear}'...`
		)
		const userSpecificKeySuffixes = [
			"pricing",
			"checkin",
			"referralCode",
			"referrerStatus",
			"betaUserStatus",
			"proCredits",
			"creditsCheckin"
		]
		for (const keySuffix of userSpecificKeySuffixes) {
			const accountName = getUserKeytarAccount(
				userIdBeforeClear,
				keySuffix
			)
			try {
				await keytar.deletePassword(keytarService, accountName)
			} catch (err) {
				// Warn if a specific key doesn't exist, but don't stop logout
				// console.warn(`[auth.js] Keytar entry '${accountName}' not found for deletion or error: ${err.message}`);
			}
		}
		console.log(
			`[auth.js] Finished deleting user-specific Keytar entries for ${userIdBeforeClear}.`
		)
	} else {
		console.log(
			"[auth.js] User ID not available, skipping user-specific Keytar entry deletion."
		)
	}

	accessToken = null
	profile = null
	console.log("[auth.js] In-memory session cleared. Logout complete.")
}

// --- User-Specific Data Management with Keytar ---
function getUserKeytarAccount(userId, key) {
	if (!userId) {
		console.error(
			"[auth.js] Cannot generate Keytar account name: userId missing."
		)
		throw new Error("Attempted to use Keytar without userId.")
	}
	return `${userId}_${key}`
}

async function setCheckinInKeytar(userId) {
	if (!userId) {
		console.warn("[auth.js] Cannot set check-in: userId missing.")
		return
	}
	try {
		const accountName = getUserKeytarAccount(userId, "checkin")
		const ts = Math.floor(Date.now() / 1000).toString()
		const encryptedTs = await encrypt(ts)
		await keytar.setPassword(keytarService, accountName, encryptedTs)
		console.log(`[auth.js] General check-in updated for user ${userId}.`)
	} catch (error) {
		console.error(
			`[auth.js] Error setting check-in for user ${userId}:`,
			error.message
		)
	}
}

// This function now fetches the role from custom claims in the ID token if available,
// or falls back to fetching from the backend.
export async function fetchAndSetUserRole() {
	if (!profile || !profile.sub) {
		throw new Error(
			"Cannot fetch user role: User not logged in or profile incomplete."
		)
	}
	const userId = profile.sub
	const customClaimNamespace = auth0Audience.endsWith("/")
		? auth0Audience
		: `${auth0Audience}/` // Ensure trailing slash for namespace

	// Try to get role from custom claims first
	let roleFromClaims = profile[`${customClaimNamespace}role`] // Adjust claim name if different in your Auth0 Action

	if (Array.isArray(roleFromClaims)) {
		// If roles are an array (e.g., from event.user.roles)
		roleFromClaims = roleFromClaims.length > 0 ? roleFromClaims[0] : null // Take the first role, or null
	}

	if (roleFromClaims) {
		console.log(
			`[auth.js] User role '${roleFromClaims}' found in token claims for user ${userId}. Storing as pricing tier.`
		)
		const accountName = getUserKeytarAccount(userId, "pricing")
		const encryptedPricingTier = await encrypt(
			roleFromClaims.toString().toLowerCase()
		) // Ensure string and lowercase
		await keytar.setPassword(
			keytarService,
			accountName,
			encryptedPricingTier
		)
		console.log(
			`[auth.js] Pricing tier ('${roleFromClaims}') stored from claims for user ${userId}.`
		)
		await setCheckinInKeytar(userId) // Update check-in time
		return // Role set from claims, no need to call backend
	} else {
		console.log(
			`[auth.js] Role not found in token claims for user ${userId}. Falling back to backend API /get-role.`
		)
	}

	// Fallback: Fetch from backend /get-role if not in claims
	const token = getAccessToken()
	if (!token) {
		throw new Error(
			"Cannot fetch user role from backend: Access token missing."
		)
	}

	console.log(
		`[auth.js] Fetching user role for ${userId} from backend /get-role...`
	)
	try {
		const response = await fetch(`${appServerUrl}/get-role`, {
			method: "POST", // Backend expects POST
			headers: {
				"Content-Type": "application/json",
				Authorization: `Bearer ${token}`
			}
		})
		if (!response.ok) {
			const errorText = await response.text()
			throw new Error(
				`Error fetching role from backend: ${response.statusText}. Details: ${errorText}`
			)
		}
		const { role } = await response.json()

		if (role) {
			console.log(
				`[auth.js] Role '${role}' received from backend. Storing as pricing for ${userId}.`
			)
			const accountName = getUserKeytarAccount(userId, "pricing")
			const encryptedPricingTier = await encrypt(
				role.toString().toLowerCase()
			)
			await keytar.setPassword(
				keytarService,
				accountName,
				encryptedPricingTier
			)
			console.log(
				`[auth.js] Pricing tier ('${role}') stored from backend for user ${userId}.`
			)
			await setCheckinInKeytar(userId)
		} else {
			console.warn(
				`[auth.js] No role returned from backend for ${userId}. Pricing not stored.`
			)
		}
	} catch (error) {
		console.error(
			`[auth.js] Error fetching/setting role for ${userId} from backend: ${error.message}`
		)
		throw new Error(
			`Failed to fetch/set user role for ${userId} from backend.`
		)
	}
}

// --- Keytar Getters/Setters for User-Specific Data (largely unchanged, rely on userId) ---

export async function getPricingFromKeytar(userId) {
	if (!userId) {
		console.warn("[auth.js] getPricing: userId missing.")
		return null
	}
	const accountName = getUserKeytarAccount(userId, "pricing")
	try {
		const encPricing = await keytar.getPassword(keytarService, accountName)
		return encPricing ? await decrypt(encPricing) : null
	} catch (error) {
		console.error(
			`[auth.js] Error getting pricing for ${userId}:`,
			error.message
		)
		return null
	}
}

export async function getCheckinFromKeytar(userId) {
	if (!userId) {
		console.warn("[auth.js] getCheckin: userId missing.")
		return null
	}
	const accountName = getUserKeytarAccount(userId, "checkin")
	try {
		const encCheckin = await keytar.getPassword(keytarService, accountName)
		if (!encCheckin) return null
		return parseInt(await decrypt(encCheckin), 10)
	} catch (error) {
		console.error(
			`[auth.js] Error getting checkin for ${userId}:`,
			error.message
		)
		return null
	}
}

export async function getCreditsFromKeytar(userId) {
	if (!userId) {
		console.warn("[auth.js] getCredits: userId missing. Defaulting to 0.")
		return 0
	}
	const accountName = getUserKeytarAccount(userId, "proCredits")
	try {
		const encCredits = await keytar.getPassword(keytarService, accountName)
		if (!encCredits) return 0
		return parseInt(await decrypt(encCredits), 10) || 0
	} catch (error) {
		console.error(
			`[auth.js] Error getting credits for ${userId}:`,
			error.message
		)
		return 0
	}
}

export async function getCreditsCheckinFromKeytar(userId) {
	if (!userId) {
		console.warn("[auth.js] getCreditsCheckin: userId missing.")
		return null
	}
	const accountName = getUserKeytarAccount(userId, "creditsCheckin")
	try {
		const encCheckin = await keytar.getPassword(keytarService, accountName)
		if (!encCheckin) return null
		return parseInt(await decrypt(encCheckin), 10)
	} catch (error) {
		console.error(
			`[auth.js] Error getting credits checkin for ${userId}:`,
			error.message
		)
		return null
	}
}

export async function setCreditsInKeytar(userId, credits) {
	if (!userId) {
		console.warn("[auth.js] setCredits: userId missing.")
		return
	}
	const accountName = getUserKeytarAccount(userId, "proCredits")
	try {
		const encCredits = await encrypt(credits.toString())
		await keytar.setPassword(keytarService, accountName, encCredits)
		console.log(
			`[auth.js] Pro credits set to ${credits} for user ${userId}.`
		)
	} catch (error) {
		console.error(
			`[auth.js] Error setting credits for ${userId}:`,
			error.message
		)
	}
}

export async function setCreditsCheckinInKeytar(userId) {
	if (!userId) {
		console.warn("[auth.js] setCreditsCheckin: userId missing.")
		return
	}
	const accountName = getUserKeytarAccount(userId, "creditsCheckin")
	try {
		const ts = Math.floor(Date.now() / 1000).toString()
		const encTs = await encrypt(ts)
		await keytar.setPassword(keytarService, accountName, encTs)
		console.log(
			`[auth.js] Daily credits checkin updated for user ${userId}.`
		)
	} catch (error) {
		console.error(
			`[auth.js] Error setting credits checkin for ${userId}:`,
			error.message
		)
	}
}

// Fetching referral/beta status should ideally come from token claims first.
// If claims are not available or need to be refreshed, then call backend.

export async function getBetaUserStatusFromKeytar(userId) {
	if (!userId) {
		console.warn("[auth.js] getBetaStatus: userId missing.")
		return null
	}
	// First, try to get from profile claims if profile is loaded
	if (profile && profile.sub === userId) {
		const customClaimNamespace = auth0Audience.endsWith("/")
			? auth0Audience
			: `${auth0Audience}/`
		const betaStatusClaim = profile[`${customClaimNamespace}betaUserStatus`]
		if (typeof betaStatusClaim === "boolean") {
			// console.log(`[auth.js] Beta status for ${userId} from token claim: ${betaStatusClaim}`);
			return betaStatusClaim
		}
	}
	// Fallback to Keytar if not in claims or profile not matching
	const accountName = getUserKeytarAccount(userId, "betaUserStatus")
	try {
		const betaStatusStr = await keytar.getPassword(
			keytarService,
			accountName
		)
		return betaStatusStr === null ? null : betaStatusStr === "true"
	} catch (error) {
		console.error(
			`[auth.js] Error getting beta status (Keytar) for ${userId}:`,
			error.message
		)
		return null
	}
}

export async function setBetaUserStatusInKeytar(userId, betaUserStatus) {
	if (!userId) {
		console.warn("[auth.js] setBetaStatus: userId missing.")
		return
	}
	const accountName = getUserKeytarAccount(userId, "betaUserStatus")
	try {
		await keytar.setPassword(
			keytarService,
			accountName,
			betaUserStatus.toString()
		)
		console.log(
			`[auth.js] Beta status set to '${betaUserStatus}' in Keytar for ${userId}.`
		)
	} catch (error) {
		console.error(
			`[auth.js] Error setting beta status (Keytar) for ${userId}:`,
			error.message
		)
	}
}

export async function fetchAndSetReferralCode() {
	if (!profile || !profile.sub)
		throw new Error("fetchReferralCode: User not logged in.")
	const userId = profile.sub
	const token = getAccessToken()
	if (!token) throw new Error("fetchReferralCode: Access token missing.")

	// Try from claims first
	const customClaimNamespace = auth0Audience.endsWith("/")
		? auth0Audience
		: `${auth0Audience}/`
	const referralCodeClaim = profile[`${customClaimNamespace}referralCode`]
	if (referralCodeClaim) {
		console.log(
			`[auth.js] Referral code '${referralCodeClaim}' from token claim for ${userId}. Storing.`
		)
		const accountName = getUserKeytarAccount(userId, "referralCode")
		const encryptedCode = await encrypt(referralCodeClaim)
		await keytar.setPassword(keytarService, accountName, encryptedCode)
		return referralCodeClaim
	}

	console.log(
		`[auth.js] Fetching referral code for ${userId} from backend /get-referral-code...`
	)
	try {
		const response = await fetch(`${appServerUrl}/get-referral-code`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Authorization: `Bearer ${token}`
			}
		})
		if (!response.ok) {
			/* ... error handling ... */ throw new Error(
				`Backend error /get-referral-code: ${response.statusText}`
			)
		}
		const { referralCode } = await response.json()
		if (referralCode) {
			const accountName = getUserKeytarAccount(userId, "referralCode")
			const encryptedCode = await encrypt(referralCode)
			await keytar.setPassword(keytarService, accountName, encryptedCode)
			console.log(
				`[auth.js] Referral code ('${referralCode}') from backend stored for ${userId}.`
			)
			return referralCode
		} else {
			console.warn(
				`[auth.js] No referral code from backend for ${userId}.`
			)
			return null
		}
	} catch (error) {
		console.error(
			`[auth.js] Error fetching/setting referral code for ${userId}: ${error.message}`
		)
		throw error
	}
}

export async function fetchAndSetReferrerStatus() {
	if (!profile || !profile.sub)
		throw new Error("fetchReferrerStatus: User not logged in.")
	const userId = profile.sub
	const token = getAccessToken()
	if (!token) throw new Error("fetchReferrerStatus: Access token missing.")

	// Try from claims first
	const customClaimNamespace = auth0Audience.endsWith("/")
		? auth0Audience
		: `${auth0Audience}/`
	const referrerStatusClaim = profile[`${customClaimNamespace}referrerStatus`]
	if (typeof referrerStatusClaim === "boolean") {
		console.log(
			`[auth.js] Referrer status '${referrerStatusClaim}' from token claim for ${userId}. Storing.`
		)
		const accountName = getUserKeytarAccount(userId, "referrerStatus")
		const encryptedStatus = await encrypt(referrerStatusClaim.toString())
		await keytar.setPassword(keytarService, accountName, encryptedStatus)
		return referrerStatusClaim
	}

	console.log(
		`[auth.js] Fetching referrer status for ${userId} from backend /get-referrer-status...`
	)
	try {
		const response = await fetch(`${appServerUrl}/get-referrer-status`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Authorization: `Bearer ${token}`
			}
		})
		if (!response.ok) {
			/* ... error handling ... */ throw new Error(
				`Backend error /get-referrer-status: ${response.statusText}`
			)
		}
		const { referrerStatus } = await response.json() // Expects boolean
		const accountName = getUserKeytarAccount(userId, "referrerStatus")
		const encryptedStatus = await encrypt(referrerStatus.toString())
		await keytar.setPassword(keytarService, accountName, encryptedStatus)
		console.log(
			`[auth.js] Referrer status ('${referrerStatus}') from backend stored for ${userId}.`
		)
		return referrerStatus
	} catch (error) {
		console.error(
			`[auth.js] Error fetching/setting referrer status for ${userId}: ${error.message}`
		)
		throw error
	}
}

export async function fetchAndSetBetaUserStatus() {
	if (!profile || !profile.sub)
		throw new Error("fetchBetaStatus: User not logged in.")
	const userId = profile.sub
	const token = getAccessToken()
	if (!token) throw new Error("fetchBetaStatus: Access token missing.")

	// Try from claims first
	const customClaimNamespace = auth0Audience.endsWith("/")
		? auth0Audience
		: `${auth0Audience}/`
	const betaStatusClaim = profile[`${customClaimNamespace}betaUserStatus`]
	if (typeof betaStatusClaim === "boolean") {
		console.log(
			`[auth.js] Beta status '${betaStatusClaim}' from token claim for ${userId}. Storing.`
		)
		await setBetaUserStatusInKeytar(userId, betaStatusClaim) // Uses existing setter
		return betaStatusClaim
	}

	console.log(
		`[auth.js] Fetching beta status for ${userId} from backend /get-beta-user-status...`
	)
	try {
		const response = await fetch(`${appServerUrl}/get-beta-user-status`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Authorization: `Bearer ${token}`
			}
		})
		if (!response.ok) {
			/* ... error handling ... */ throw new Error(
				`Backend error /get-beta-user-status: ${response.statusText}`
			)
		}
		const { betaUserStatus } = await response.json() // Expects boolean
		await setBetaUserStatusInKeytar(userId, betaUserStatus)
		console.log(
			`[auth.js] Beta status ('${betaUserStatus}') from backend stored for ${userId}.`
		)
		return betaUserStatus
	} catch (error) {
		console.error(
			`[auth.js] Error fetching/setting beta status for ${userId}: ${error.message}`
		)
		throw error
	}
}

export async function getReferralCodeFromKeytar(userId) {
	if (!userId) {
		console.warn("[auth.js] getReferralCode: userId missing.")
		return null
	}
	// Try from claims first if profile matches
	if (profile && profile.sub === userId) {
		const customClaimNamespace = auth0Audience.endsWith("/")
			? auth0Audience
			: `${auth0Audience}/`
		const code = profile[`${customClaimNamespace}referralCode`]
		if (code) return code
	}
	const accountName = getUserKeytarAccount(userId, "referralCode")
	try {
		const encCode = await keytar.getPassword(keytarService, accountName)
		return encCode ? await decrypt(encCode) : null
	} catch (error) {
		console.error(
			`[auth.js] Error getting referral code (Keytar) for ${userId}:`,
			error.message
		)
		return null
	}
}

export async function getReferrerStatusFromKeytar(userId) {
	if (!userId) {
		console.warn("[auth.js] getReferrerStatus: userId missing.")
		return null
	}
	// Try from claims first
	if (profile && profile.sub === userId) {
		const customClaimNamespace = auth0Audience.endsWith("/")
			? auth0Audience
			: `${auth0Audience}/`
		const status = profile[`${customClaimNamespace}referrerStatus`]
		if (typeof status === "boolean") return status
	}
	const accountName = getUserKeytarAccount(userId, "referrerStatus")
	try {
		const encStatus = await keytar.getPassword(keytarService, accountName)
		if (encStatus === null) return null
		return (await decrypt(encStatus)) === "true"
	} catch (error) {
		console.error(
			`[auth.js] Error getting referrer status (Keytar) for ${userId}:`,
			error.message
		)
		return null
	}
}
