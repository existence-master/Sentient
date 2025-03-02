/**
 * @file Authentication and authorization utilities.
 * This module handles user authentication using Auth0, token management,
 * secure storage of tokens using Keytar, and user profile and role retrieval.
 * It also includes functionalities for managing user credits, referral status, and beta user status.
 */

import { jwtDecode } from "jwt-decode"
import dotenv from "dotenv"
import keytar from "keytar"
import os from "os"
import fetch from "node-fetch"
import path, { dirname } from "path"
import { fileURLToPath } from "url"
import { app } from "electron"

/**
 * Boolean indicating if the operating system is Windows.
 * @constant {boolean}
 */
const isWindows = process.platform === "win32"
/**
 * Boolean indicating if the operating system is Linux.
 * @constant {boolean}
 */

/**
 * The current file's path.
 * @constant {string}
 * @private
 */
const __filename = fileURLToPath(import.meta.url)
/**
 * The directory name of the current file.
 * @constant {string}
 * @private
 */
const __dirname = dirname(__filename)

/**
 * Path to the .env file, determined based on whether the app is packaged or not.
 * @type {string}
 */
let dotenvPath

if (app.isPackaged) {
	dotenvPath = isWindows
		? path.join(process.resourcesPath, ".env")
		: path.join(app.getPath("home"), ".sentient.env")
} else {
	dotenvPath = path.resolve(__dirname, "../.env")
}

// Load environment variables from .env file
dotenv.config({ path: dotenvPath })

/**
 * Auth0 domain URL from environment variables.
 * @constant {string}
 */
const auth0Domain = process.env.AUTH0_DOMAIN
/**
 * Auth0 client ID from environment variables.
 * @constant {string}
 */
const clientId = process.env.AUTH0_CLIENT_ID
/**
 * Service name used for storing credentials in Keytar.
 * @constant {string}
 */
const keytarService = "electron-openid-oauth"
/**
 * Account name used for storing credentials in Keytar, based on the current user's username.
 * @constant {string}
 */
const keytarAccount = os.userInfo().username

/**
 * In-memory storage for the access token.
 * @type {string | null}
 * @private
 */
let accessToken = null
/**
 * In-memory storage for the user profile.
 * @type {object | null}
 * @private
 */
let profile = null
/**
 * In-memory storage for the refresh token.
 * @type {string | null}
 * @private
 */
let refreshToken = null

/**
 * Returns the current access token.
 * @function getAccessToken
 * @returns {string | null} The current access token, or null if not set.
 */
export function getAccessToken() {
	return accessToken
}

/**
 * Returns the current user profile.
 * @function getProfile
 * @returns {object | null} The current user profile, or null if not set.
 */
export function getProfile() {
	return profile
}

/**
 * Generates the authentication URL for Auth0.
 * This URL is used to redirect the user to the Auth0 login page.
 * @function getAuthenticationURL
 * @returns {string} The Auth0 authentication URL.
 */
export function getAuthenticationURL() {
	return (
		`https://${auth0Domain}/authorize?` +
		`scope=openid profile offline_access&` +
		`response_type=code&` +
		`client_id=${clientId}&` +
		`redirect_uri=http://localhost/callback`
	)
}

/**
 * Encrypts data by sending it to the backend server for encryption.
 * @async
 * @function encrypt
 * @param {any} data - The data to be encrypted.
 * @returns {Promise<string>} A promise that resolves with the encrypted data.
 * @throws {Error} If the encryption process fails.
 */
async function encrypt(data) {
	try {
		const response = await fetch(`${process.env.APP_SERVER_URL}/encrypt`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ data })
		})

		if (!response.ok) {
			throw new Error("Failed to encrypt data")
		}

		/** @type {{ encrypted_data: string }} */ const result =
			await response.json() // Type assertion for response body
		return result.encrypted_data
	} catch (error) {
		await console.log(`Error during encryption: ${error}`)
		throw error
	}
}

/**
 * Decrypts data by sending it to the backend server for decryption.
 * @async
 * @function decrypt
 * @param {string} encryptedData - The encrypted data to be decrypted.
 * @returns {Promise<any>} A promise that resolves with the decrypted data.
 * @throws {Error} If the decryption process fails.
 */
async function decrypt(encryptedData) {
	try {
		const response = await fetch(`${process.env.APP_SERVER_URL}/decrypt`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ encrypted_data: encryptedData })
		})

		if (!response.ok) {
			throw new Error("Failed to decrypt data")
		}

		/** @type {{ decrypted_data: string }} */ const result =
			await response.json() // Type assertion for response body
		return result.decrypted_data
	} catch (error) {
		await console.log(`Error during decryption: ${error}`)
		throw error
	}
}

/**
 * Refreshes the access token using the refresh token stored in Keytar.
 * If successful, updates the access token and user profile in memory.
 * @async
 * @function refreshTokens
 * @returns {Promise<void>} A promise that resolves when tokens are refreshed successfully.
 * @throws {Error} If there is no refresh token available or if token refresh fails.
 */
export async function refreshTokens() {
	// Retrieve the encrypted refresh token from Keytar
	const savedRefreshToken = await keytar.getPassword(
		keytarService,
		keytarAccount
	)

	if (savedRefreshToken) {
		// Decrypt the refresh token
		const decryptedToken = await decrypt(savedRefreshToken)

		const refreshOptions = {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({
				grant_type: "refresh_token",
				client_id: clientId,
				refresh_token: decryptedToken // Use decrypted refresh token
			})
		}

		try {
			// Exchange refresh token for new tokens
			const response = await fetch(
				`https://${auth0Domain}/oauth/token`,
				refreshOptions
			)
			/** @type {{ access_token: string, id_token: string }} */ const data =
				await response.json() // Type assertion for response body

			accessToken = data.access_token // Update access token in memory
			profile = jwtDecode(data.id_token) // Decode and update user profile in memory
		} catch (error) {
			// Logout user if refresh fails
			await logout()
			throw error
		}
	} else {
		// Throw error if no refresh token is found
		throw new Error("No available refresh token.")
	}
}

/**
 * Loads tokens from Auth0 callback URL after successful authentication.
 * Stores the refresh token in Keytar after encrypting it.
 * @async
 * @function loadTokens
 * @param {string} callbackURL - The callback URL received from Auth0 after authentication.
 * @returns {Promise<void>} A promise that resolves when tokens are loaded and stored successfully.
 * @throws {Error} If token loading or storage fails.
 */
export async function loadTokens(callbackURL) {
	const url = new URL(callbackURL)
	const params = {
		grant_type: "authorization_code",
		client_id: clientId,
		code: url.searchParams.get("code"), // Extract authorization code from callback URL
		redirect_uri: "http://localhost/callback"
	}

	const options = {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(params)
	}

	try {
		// Exchange authorization code for tokens
		const response = await fetch(
			`https://${auth0Domain}/oauth/token`,
			options
		)
		/** @type {{ access_token: string, id_token: string, refresh_token: string }} */ const data =
			await response.json() // Type assertion for response body

		accessToken = data.access_token // Set access token in memory
		profile = jwtDecode(data.id_token) // Decode and set user profile in memory
		refreshToken = data.refresh_token // Set refresh token in memory

		if (refreshToken) {
			// Encrypt and save refresh token to Keytar
			const encryptedToken = await encrypt(refreshToken)
			await keytar.setPassword(
				keytarService,
				keytarAccount,
				encryptedToken
			)
		}
	} catch (error) {
		// Logout user if token loading fails
		await logout()
		throw error
	}
}

/**
 * Logs out the user by deleting tokens from Keytar and resetting in-memory tokens.
 * @async
 * @function logout
 * @returns {Promise<void>} A promise that resolves when logout process is complete.
 */
export async function logout() {
	await keytar.deletePassword(keytarService, keytarAccount) // Delete main refresh token
	await keytar.deletePassword(keytarService, "pricing") // Delete pricing info
	await keytar.deletePassword(keytarService, "checkin") // Delete last check-in timestamp
	await keytar.deletePassword(keytarService, "referralCode") // Delete referral code
	await keytar.deletePassword(keytarService, "referrerStatus") // Delete referrer status
	await keytar.deletePassword(keytarService, "betaUserStatus") // Delete beta user status
	accessToken = null // Reset access token
	profile = null // Reset user profile
	refreshToken = null // Reset refresh token
}

/**
 * Returns the Auth0 logout URL.
 * @function getLogOutUrl
 * @returns {string} The Auth0 logout URL.
 */
export function getLogOutUrl() {
	return `https://${auth0Domain}/v2/logout`
}

/**
 * Fetches the user's role from the backend server and sets the pricing plan in Keytar.
 * @async
 * @function fetchAndSetUserRole
 * @returns {Promise<void>} A promise that resolves when user role and pricing are fetched and set successfully.
 * @throws {Error} If fetching user role or setting pricing fails.
 */
export async function fetchAndSetUserRole() {
	const userId = profile.sub // Get user ID from profile

	try {
		// Fetch user role from backend server
		const response = await fetch("http://localhost:5005/get-role", {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
			},
			body: JSON.stringify({ user_id: userId }) // Send user ID in request body
		})

		if (!response.ok) {
			throw new Error(`Error fetching role: ${response.statusText}`)
		}

		/** @type {{ role: string }} */ const { role } = await response.json() // Type assertion for response body

		if (role) {
			// Encrypt and store pricing and check-in timestamp in Keytar
			let encryptedPricing = await encrypt(role)
			let encryptedDate = await encrypt(
				Math.floor(Date.now() / 1000).toString()
			)

			await keytar.setPassword(keytarService, "pricing", encryptedPricing) // Store encrypted pricing
			await keytar.setPassword(keytarService, "checkin", encryptedDate) // Store encrypted check-in timestamp
		} else {
			console.warn("No role found for user.")
		}
	} catch (error) {
		await console.log(`Error fetching and setting user role: ${error}`)
		throw new Error("Failed to fetch or set user role.")
	}
}

/**
 * Retrieves the pricing plan from Keytar and decrypts it.
 * @async
 * @function getPricingFromKeytar
 * @returns {Promise<string | null>} A promise that resolves with the pricing plan or null if not found.
 * @throws {Error} If fetching or decrypting pricing fails.
 */
export async function getPricingFromKeytar() {
	try {
		// Retrieve encrypted pricing from Keytar
		const encryptedPricing = await keytar.getPassword(
			keytarService,
			"pricing"
		)
		if (!encryptedPricing) return null // Return null if no pricing found
		// Decrypt and return pricing
		const pricing = await decrypt(encryptedPricing)
		return pricing
	} catch (error) {
		await console.log(`Error fetching pricing from keytar: ${error}`)
		return null
	}
}

/**
 * Retrieves the last check-in timestamp from Keytar, decrypts and parses it as an integer.
 * @async
 * @function getCheckinFromKeytar
 * @returns {Promise<number | null>} A promise that resolves with the check-in timestamp as a number or null if not found.
 * @throws {Error} If fetching, decrypting or parsing check-in timestamp fails.
 */
export async function getCheckinFromKeytar() {
	try {
		// Retrieve encrypted check-in timestamp from Keytar
		const encryptedCheckin = await keytar.getPassword(
			keytarService,
			"checkin"
		)
		if (!encryptedCheckin) return null // Return null if no check-in timestamp found
		// Decrypt, parse and return check-in timestamp
		const checkin = await decrypt(encryptedCheckin)
		return parseInt(checkin, 10)
	} catch (error) {
		await console.log(`Error fetching checkin from keytar: ${error}`)
		return null
	}
}

/**
 * Retrieves the pro credits balance from Keytar, decrypts and parses it as an integer.
 * @async
 * @function getCreditsFromKeytar
 * @returns {Promise<number>} A promise that resolves with the pro credits balance as a number, defaults to 0 if not found or error occurs.
 * @throws {Error} If fetching, decrypting or parsing pro credits fails.
 */
export async function getCreditsFromKeytar() {
	try {
		// Retrieve encrypted pro credits from Keytar
		const encryptedCredits = await keytar.getPassword(
			keytarService,
			"proCredits"
		)
		if (!encryptedCredits) return 0 // Return 0 if no pro credits found
		// Decrypt, parse and return pro credits
		const credits = await decrypt(encryptedCredits)
		return parseInt(credits, 10) || 0
	} catch (error) {
		await console.log(`Error fetching proCredits from keytar: ${error}`)
		return 0
	}
}

/**
 * Retrieves the last credits check-in timestamp from Keytar, decrypts and parses it as an integer.
 * @async
 * @function getCreditsCheckinFromKeytar
 * @returns {Promise<number | null>} A promise that resolves with the credits check-in timestamp as a number or null if not found.
 * @throws {Error} If fetching, decrypting or parsing credits check-in timestamp fails.
 */
export async function getCreditsCheckinFromKeytar() {
	try {
		// Retrieve encrypted credits check-in timestamp from Keytar
		const encryptedCheckin = await keytar.getPassword(
			keytarService,
			"creditsCheckin"
		)
		if (!encryptedCheckin) return null // Return null if no credits check-in timestamp found
		// Decrypt, parse and return credits check-in timestamp
		const checkin = await decrypt(encryptedCheckin)
		return parseInt(checkin, 10)
	} catch (error) {
		await console.log(`Error fetching cheditsCheckin from keytar: ${error}`)
		return null
	}
}

/**
 * Sets the pro credits balance in Keytar after encrypting it.
 * @async
 * @function setCreditsInKeytar
 * @param {number} credits - The pro credits balance to set.
 * @returns {Promise<void>} A promise that resolves when pro credits are set and stored successfully.
 * @throws {Error} If encrypting or storing pro credits fails.
 */
export async function setCreditsInKeytar(credits) {
	try {
		// Encrypt and store pro credits in Keytar
		const encryptedCredits = await encrypt(credits.toString())
		await keytar.setPassword(keytarService, "proCredits", encryptedCredits)
	} catch (error) {
		await console.log(`Error saving proCredits to keytar: ${error}`)
	}
}

/**
 * Sets the credits check-in timestamp in Keytar after encrypting the current timestamp.
 * @async
 * @function setCreditsCheckinInKeytar
 * @returns {Promise<void>} A promise that resolves when credits check-in timestamp is set and stored successfully.
 * @throws {Error} If encrypting or storing credits check-in timestamp fails.
 */
export async function setCreditsCheckinInKeytar() {
	try {
		// Encrypt and store current timestamp as credits check-in timestamp in Keytar
		const currentDate = Math.floor(Date.now() / 1000).toString()
		const encryptedDate = await encrypt(currentDate)
		await keytar.setPassword(keytarService, "creditsCheckin", encryptedDate)
	} catch (error) {
		await console.log(`Error saving creditsCheckin to keytar: ${error}`)
	}
}

/**
 * Retrieves the beta user status from Keytar.
 * @async
 * @function getBetaUserStatusFromKeytar
 * @returns {Promise<boolean | null>} A promise that resolves with the beta user status (true/false) or null if not found.
 * @throws {Error} If fetching beta user status from Keytar fails.
 */
export async function getBetaUserStatusFromKeytar() {
	try {
		// Retrieve beta user status from Keytar
		const betaUserStatus = await keytar.getPassword(
			keytarService,
			"betaUserStatus"
		)
		if (!betaUserStatus) return null // Return null if beta user status not found
		return betaUserStatus === "true" // Return boolean value of beta user status
	} catch (error) {
		await console.log(`Error fetching betaUserStatus from keytar: ${error}`)
		return null
	}
}

/**
 * Sets the beta user status in Keytar.
 * @async
 * @function setBetaUserStatusInKeytar
 * @param {boolean} betaUserStatus - The beta user status to set (true or false).
 * @returns {Promise<void>} A promise that resolves when beta user status is set and stored successfully.
 * @throws {Error} If storing beta user status in Keytar fails.
 */
export async function setBetaUserStatusInKeytar(betaUserStatus) {
	try {
		// Store beta user status in Keytar
		await keytar.setPassword(
			keytarService,
			"betaUserStatus",
			betaUserStatus.toString()
		)
	} catch (error) {
		await console.log(`Error saving betaUserStatus to keytar: ${error}`)
	}
}

/**
 * Resets pro credits if the current date is different from the last credits check-in date.
 * @async
 * @function resetCreditsIfNecessary
 * @returns {Promise<void>} A promise that resolves after checking and possibly resetting pro credits.
 * @throws {Error} If there is an error during the credit reset process.
 */
export async function resetCreditsIfNecessary() {
	try {
		const currentDate = new Date().toDateString()
		const checkinTimestamp = await getCreditsCheckinFromKeytar()
		const checkinDate = checkinTimestamp
			? new Date(checkinTimestamp * 1000).toDateString()
			: null

		if (checkinDate !== currentDate) {
			// Reset credits and update check-in timestamp if dates are different
			await setCreditsInKeytar(5)
			await setCreditsCheckinInKeytar()
		} else {
			// Do nothing if it's the same day
		}
	} catch (error) {
		await console.log(`Error resetting proCredits: ${error}`)
		throw new Error("Error checking credits")
	}
}

/**
 * Fetches the referral code from the backend server and stores it in Keytar after encryption.
 * @async
 * @function fetchAndSetReferralCode
 * @returns {Promise<string>} A promise that resolves with the fetched referral code.
 * @throws {Error} If fetching or setting referral code fails.
 */
export async function fetchAndSetReferralCode() {
	try {
		// Fetch referral code from backend server
		const response = await fetch(
			"http://localhost:5005/get-referral-code",
			{
				method: "POST",
				headers: {
					"Content-Type": "application/json"
				},
				body: JSON.stringify({ user_id: profile.sub }) // Send user ID in request body
			}
		)

		if (!response.ok) {
			throw new Error(
				`Error fetching referral code: ${response.statusText}`
			)
		}

		/** @type {{ referralCode: string }} */ const { referralCode } =
			await response.json() // Type assertion for response body
		// Encrypt and store referral code in Keytar
		const encryptedReferralCode = await encrypt(referralCode)
		await keytar.setPassword(
			keytarService,
			"referralCode",
			encryptedReferralCode
		)

		return referralCode
	} catch (error) {
		await console.log(`Error fetching and setting referral code: ${error}`)
		throw error
	}
}

/**
 * Fetches the referrer status from the backend server and stores it in Keytar after encryption.
 * @async
 * @function fetchAndSetReferrerStatus
 * @returns {Promise<boolean>} A promise that resolves with the fetched referrer status (true/false).
 * @throws {Error} If fetching or setting referrer status fails.
 */
export async function fetchAndSetReferrerStatus() {
	try {
		// Fetch referrer status from backend server
		const response = await fetch(
			"http://localhost:5005/get-referrer-status",
			{
				method: "POST",
				headers: {
					"Content-Type": "application/json"
				},
				body: JSON.stringify({ user_id: profile.sub }) // Send user ID in request body
			}
		)

		if (!response.ok) {
			throw new Error(
				`Error fetching referrer status: ${response.statusText}`
			)
		}

		/** @type {{ referrerStatus: boolean }} */ const { referrerStatus } =
			await response.json() // Type assertion for response body
		// Encrypt and store referrer status in Keytar
		const encryptedReferrerStatus = await encrypt(referrerStatus.toString())
		await keytar.setPassword(
			keytarService,
			"referrerStatus",
			encryptedReferrerStatus
		)

		return referrerStatus
	} catch (error) {
		await console.log(
			`Error fetching and setting referrer status: ${error}`
		)
		throw error
	}
}

/**
 * Fetches the beta user status from the backend server and stores it in Keytar.
 * @async
 * @function fetchAndSetBetaUserStatus
 * @returns {Promise<boolean>} A promise that resolves with the fetched beta user status (true/false).
 * @throws {Error} If fetching or setting beta user status fails.
 */
export async function fetchAndSetBetaUserStatus() {
	try {
		// Fetch beta user status from backend server
		const response = await fetch(
			"http://localhost:5005/get-beta-user-status",
			{
				method: "POST",
				headers: {
					"Content-Type": "application/json"
				},
				body: JSON.stringify({ user_id: profile.sub }) // Send user ID in request body
			}
		)

		if (!response.ok) {
			throw new Error(
				`Error fetching beta user status: ${response.statusText}`
			)
		}

		/** @type {{ betaUserStatus: boolean }} */ const { betaUserStatus } =
			await response.json() // Type assertion for response body

		console.log(`Beta user status: ${betaUserStatus}`)
		await setBetaUserStatusInKeytar(betaUserStatus) // Store beta user status in Keytar

		return betaUserStatus
	} catch (error) {
		await console.log(
			`Error fetching and setting beta user status: ${error}`
		)
		throw error
	}
}

/**
 * Retrieves the referral code from Keytar, decrypting it before returning.
 * @async
 * @function getReferralCodeFromKeytar
 * @returns {Promise<string | null>} A promise that resolves with the referral code or null if not found.
 * @throws {Error} If fetching or decrypting referral code fails.
 */
export async function getReferralCodeFromKeytar() {
	try {
		// Retrieve encrypted referral code from Keytar
		const encryptedReferralCode = await keytar.getPassword(
			keytarService,
			"referralCode"
		)
		if (!encryptedReferralCode) return null // Return null if referral code not found
		// Decrypt and return referral code
		const referralCode = await decrypt(encryptedReferralCode)
		return referralCode
	} catch (error) {
		await console.log(`Error fetching referral code form keytar: ${error}`)
		return null
	}
}

/**
 * Retrieves the referrer status from Keytar, decrypting it and parsing as a boolean before returning.
 * @async
 * @function getReferrerStatusFromKeytar
 * @returns {Promise<boolean | null>} A promise that resolves with the referrer status (true/false) or null if not found.
 * @throws {Error} If fetching or decrypting referrer status fails.
 */
export async function getReferrerStatusFromKeytar() {
	try {
		// Retrieve encrypted referrer status from Keytar
		const encryptedReferrerStatus = await keytar.getPassword(
			keytarService,
			"referrerStatus"
		)
		if (!encryptedReferrerStatus) return null // Return null if referrer status not found
		// Decrypt, parse and return referrer status
		const referrerStatus = await decrypt(encryptedReferrerStatus)
		return referrerStatus === "true"
	} catch (error) {
		await console.log(
			`Error fetching referrer status from keytar: ${error}`
		)
		return null
	}
}

/**
 * Sets an API key for a given provider by sending it to the Python backend.
 * @param {string} provider - The name of the provider.
 * @param {string} apiKey - The API key to store.
 * @returns {Promise<{ success: boolean, error?: string }>} - Success status and optional error message.
 */
export async function setApiKey(provider, apiKey) {
	try {
		const response = await fetch("http://localhost:5005/set-api-key", {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
			},
			body: JSON.stringify({ provider, api_key: apiKey })
		})
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`)
		}
		const data = await response.json()
		return { success: data.success } // { success: true }
	} catch (error) {
		console.error(`Error setting API key for ${provider}:`, error)
		return { success: false, error: error.message }
	}
}

/**
 * Checks if an API key exists for a given provider by querying the Python backend.
 * @param {string} provider - The name of the provider.
 * @returns {Promise<boolean>} - True if the key exists, false otherwise.
 */
export async function hasApiKey(provider) {
	try {
		const response = await fetch("http://localhost:5005/has-api-key", {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
			},
			body: JSON.stringify({ provider })
		})
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`)
		}
		const data = await response.json()
		return data.exists // true or false
	} catch (error) {
		console.error(`Error checking API key for ${provider}:`, error)
		return false // Return false on error, consistent with original behavior
	}
}

/**
 * Deletes the API key for a given provider by sending a request to the Python backend.
 * @param {string} provider - The name of the provider.
 * @returns {Promise<{ success: boolean, error?: string }>} - Success status and optional error message.
 */
export async function deleteApiKey(provider) {
	try {
		const response = await fetch("http://localhost:5005/delete-api-key", {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
			},
			body: JSON.stringify({ provider })
		})
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`)
		}
		const data = await response.json()
		return { success: data.success }
	} catch (error) {
		console.error(`Error deleting API key for ${provider}:`, error)
		return { success: false, error: error.message }
	}
}
