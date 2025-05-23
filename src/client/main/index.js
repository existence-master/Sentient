/**
 * @file Main process file for the Electron application.
 * Handles application lifecycle, window management, inter-process communication (IPC),
 * auto-updates, and backend interactions including WebSocket communication.
 *
 * Key responsibilities include:
 * - Managing the main application window and authentication windows.
 * - Setting up IPC handlers to facilitate communication between the main process and renderer processes.
 * - Handling application startup, including mode detection (development/production) and path configurations.
 * - Managing auto-updates for packaged applications.
 * - Establishing and maintaining a WebSocket connection to the backend for real-time notifications.
 * - Performing user authentication checks and managing user-specific data using Keytar.
 * - Relaying requests from the renderer to the backend API, including authentication headers.
 *
 * This file has been updated to consistently use user_id (obtained from a validated Access Token)
 * for all backend calls and Keytar operations, ensuring user-specific data handling.
 * HTTP methods for backend calls have been aligned with server expectations (e.g., POST for most actions).
 * WebSocket communication now includes an authentication step.
 */

import { app, BrowserWindow, ipcMain, dialog, Notification } from "electron"
import path, { dirname } from "path"
import { fileURLToPath } from "url"
import fetch from "node-fetch"
import dotenv from "dotenv"
import pkg from "electron-updater"
const { autoUpdater } = pkg
import serve from "electron-serve"
import WebSocket from "ws"

// --- Authentication and Utility Imports ---
import {
	getProfile,
	getAccessToken, // Getter for the current access token
	refreshTokens, // Function to refresh authentication tokens
	getCheckinFromKeytar, // Functions for Keytar (secure storage) interactions
	getPricingFromKeytar,
	getCreditsFromKeytar,
	setCreditsInKeytar,
	getCreditsCheckinFromKeytar,
	setCreditsCheckinInKeytar,
	fetchAndSetUserRole, // Functions to fetch and store user-specific data
	fetchAndSetReferralCode,
	getReferrerStatusFromKeytar,
	fetchAndSetReferrerStatus,
	fetchAndSetBetaUserStatus,
	getReferralCodeFromKeytar,
	getBetaUserStatusFromKeytar,
	setBetaUserStatusInKeytar
} from "../utils/auth.js"
import { getPrivateData } from "../utils/api.js" // API utility (may need user context)
import { createAuthWindow, createLogoutWindow } from "./auth.js" // Window creation functions

// --- Constants and Path Configurations ---
const isWindows = process.platform === "win32"
const isLinux = process.platform === "linux"
let basePath // Base path for application data, differs for packaged/dev
let dotenvPath // Path to the .env file
let appOutDir // Output directory for packaged app resources

const __filename = fileURLToPath(import.meta.url) // Current file's path
const __dirname = dirname(__filename) // Current file's directory

// Configure paths based on whether the app is packaged (production) or not (development)
if (app.isPackaged) {
	console.log("Application is running in PRODUCTION mode (packaged).")
	// Define paths for production environment
	basePath = isWindows
		? path.join(
				app.getPath("home"),
				"AppData",
				"Local",
				"Programs",
				"Sentient"
			)
		: path.join(app.getPath("home"), ".config", "Sentient")
	dotenvPath = isWindows
		? path.join(process.resourcesPath, ".env") // .env expected in resources for Windows packaged app
		: path.join(app.getPath("home"), ".sentient.env") // User-specific .env for Linux
	appOutDir = path.join(__dirname, "../out") // Standard output directory for packaged app
} else {
	console.log("Application is running in DEVELOPMENT mode.")
	// Define paths for development environment
	dotenvPath = path.resolve(__dirname, "../../.env") // .env in project root
	appOutDir = path.join(__dirname, "../out") // Output directory within project structure
}

// Load environment variables from the determined .env file path
dotenv.config({ path: dotenvPath })

// --- Helper Functions ---

/**
 * Safely retrieves the user ID (Auth0 'sub' claim) from the current user profile.
 * Logs an error and returns null if the profile or 'sub' field is not available,
 * which typically means the user is not logged in or profile data is missing.
 *
 * @returns {string | null} The user ID ('sub') or null if not available.
 */
function getUserIdFromProfile() {
	const userProfile = getProfile() // Retrieve profile from auth utility
	if (!userProfile || !userProfile.sub) {
		console.error(
			"Error: Cannot get user ID. User profile or 'sub' field is missing. User might not be logged in or profile data is incomplete."
		)
		return null
	}
	return userProfile.sub
}

/**
 * Creates the Authorization header object required for authenticated API requests.
 * Uses the current access token.
 *
 * @returns {{ Authorization: string } | null} An object with the Authorization header
 *                                             (e.g., { Authorization: 'Bearer <token>' })
 *                                             or null if the access token is not available.
 */
function getAuthHeader() {
	const token = getAccessToken() // Retrieve access token from auth utility
	if (!token) {
		console.error(
			"Error: Cannot create authorization header. Access token is missing."
		)
		return null
	}
	return { Authorization: `Bearer ${token}` }
}

// --- WebSocket Connection Management ---
let ws // WebSocket instance
let wsAuthenticated = false // Flag to track if the WebSocket connection is authenticated

/**
 * Establishes and manages the WebSocket connection to the backend server.
 * Handles connection opening, message receiving (including authentication),
 * closing, and errors. Attempts to authenticate upon connection.
 */
function connectWebSocket() {
	// Prevent multiple connection attempts if already open or connecting
	if (
		ws &&
		(ws.readyState === WebSocket.OPEN ||
			ws.readyState === WebSocket.CONNECTING)
	) {
		console.log(
			"WebSocket connection is already open or in the process of connecting."
		)
		return
	}

	wsAuthenticated = false // Reset authentication status for new connection attempts
	console.log(
		"Attempting to establish WebSocket connection to ws://localhost:5000/ws..."
	)
	try {
		ws = new WebSocket("ws://localhost:5000/ws") // Backend WebSocket URL

		// --- WebSocket Event Handlers ---

		// On Connection Open: Attempt authentication
		ws.onopen = () => {
			console.log(
				"WebSocket connection opened successfully. Attempting authentication..."
			)
			const token = getAccessToken()
			const userId = getUserIdFromProfile() // For logging context

			if (token) {
				console.log(
					`WebSocket: Sending authentication message for user ${userId || "UNKNOWN_USER"}.`
				)
				// Send authentication message with token to the server
				ws.send(JSON.stringify({ type: "auth", token: token }))
			} else {
				console.error(
					"WebSocket Error: Cannot authenticate - Access token is not available."
				)
				ws.close(
					1008,
					"Access token unavailable for WebSocket authentication."
				) // 1008: Policy Violation
			}
		}

		// On Message Received: Process incoming messages from the server
		ws.onmessage = (event) => {
			try {
				const messageData = JSON.parse(event.data)
				console.log(
					"WebSocket message received from server. Type:",
					messageData.type
				)

				// Handle Authentication Response
				if (messageData.type === "auth_success") {
					wsAuthenticated = true
					console.log(
						`WebSocket authenticated successfully for user: ${messageData.user_id}.`
					)
					// Potentially trigger UI updates or fetch initial data now that connection is authenticated
					return // Authentication message handled
				} else if (messageData.type === "auth_failure") {
					console.error(
						`WebSocket authentication failed: ${messageData.message}.`
					)
					wsAuthenticated = false
					ws.close(1008, "WebSocket authentication failed by server.")
					// Optionally notify the UI about the authentication failure
					return // Authentication message handled
				}

				// Optional: Ignore further messages if not authenticated
				// if (!wsAuthenticated) {
				//     console.warn("WebSocket: Ignoring incoming message - connection is not authenticated.");
				//     return;
				// }

				// Process Other Message Types (e.g., Notifications)
				let notificationTitle = ""
				let notificationBody = ""

				switch (messageData.type) {
					case "task_completed":
						notificationTitle = "Task Completed!"
						notificationBody = `Task "${messageData.description}" has been completed successfully.`
						break
					case "task_error":
						notificationTitle = "Task Error"
						notificationBody = `An error occurred with task "${messageData.description}": ${messageData.error}`
						break
					case "memory_operation_completed":
						notificationTitle = "Memory Operation Successful"
						notificationBody = `Memory operation (ID: ${messageData.operation_id}) completed.`
						break
					case "memory_operation_error":
						notificationTitle = "Memory Operation Failed"
						notificationBody = `Memory operation (ID: ${messageData.operation_id}) failed: ${messageData.error}`
						break
					case "task_approval_pending":
						notificationTitle = "Approval Needed"
						notificationBody = `Task "${messageData.description}" requires your approval to proceed.`
						break
					case "task_cancelled":
						notificationTitle = "Task Cancelled"
						notificationBody = `Task "${messageData.description}" was cancelled: ${messageData.error || "No reason provided."}`
						break
					case "new_message": // Example: If backend pushes new chat messages
						notificationTitle = "New Message Received!"
						notificationBody =
							messageData.message?.substring(0, 100) +
							(messageData.message?.length > 100 ? "..." : "")
						break
					// Add handlers for other custom notification types from the backend
					default:
						console.log(
							"WebSocket: Ignoring unknown message type received from server:",
							messageData.type
						)
				}

				// If a notification was constructed, display it
				if (notificationTitle && notificationBody) {
					new Notification({
						title: notificationTitle,
						body: notificationBody
					}).show()
				}
			} catch (e) {
				console.error(
					"Error processing incoming WebSocket message:",
					e,
					"Raw data:",
					event.data
				)
			}
		}

		// On Connection Close: Log closure and potentially handle reconnection
		ws.onclose = (event) => {
			wsAuthenticated = false
			console.log(
				`WebSocket connection closed. Code: ${event.code}, Reason: "${event.reason || "No reason specified"}".`
			)
			// Optional: Implement reconnection logic (e.g., with exponential backoff)
			// setTimeout(connectWebSocket, 5000); // Simple retry after 5 seconds
		}

		// On Error: Log WebSocket errors
		ws.onerror = (error) => {
			wsAuthenticated = false
			console.error(
				"WebSocket connection error:",
				error?.message || error
			)
			// Consider triggering reconnection logic here as well
		}
	} catch (e) {
		console.error("Failed to initiate WebSocket connection attempt:", e)
		// setTimeout(connectWebSocket, 5000); // Retry on initial connection error
	}
}

// --- Main Application Window Management ---
let mainWindow // Holds the main BrowserWindow instance
// `electron-serve` for serving packaged app content from `app://-`
const appServe = app.isPackaged ? serve({ directory: appOutDir }) : null

/**
 * Creates and configures the main application window.
 * Loads the appropriate URL based on development or production mode.
 * Sets up window event handlers for closing and cleanup.
 */
export const createAppWindow = () => {
	mainWindow = new BrowserWindow({
		width: 2000, // Initial window dimensions
		height: 1500,
		webPreferences: {
			preload: path.join(__dirname, "preload.js"), // Path to preload script
			contextIsolation: true, // Recommended for security
			enableRemoteModule: false, // Deprecated and insecure
			devTools: !app.isPackaged // Enable DevTools only in development
		},
		backgroundColor: "#000000", // Window background color
		autoHideMenuBar: true // Hide the default menu bar
	})

	mainWindow.isMainWindow = true // Custom flag to identify the main window

	// Load content into the window
	if (app.isPackaged) {
		// Production: Serve static files from the packaged app's output directory
		appServe(mainWindow)
			.then(() => {
				mainWindow.loadURL("app://-") // Load the app's index.html
				// Attempt WebSocket connection after the packaged app window loads
				mainWindow.webContents.on("did-finish-load", connectWebSocket)
			})
			.catch((err) => console.error("Failed to serve packaged app:", err))
	} else {
		// Development: Load from the development server URL (e.g., React dev server)
		const devUrl = process.env.ELECTRON_APP_URL || "http://localhost:3000"
		console.log(`Attempting to load development URL: ${devUrl}`)
		mainWindow
			.loadURL(devUrl)
			.then(() => {
				console.log("Main window URL loaded successfully.")
				// Attempt WebSocket connection after the development window loads
				connectWebSocket()
			})
			.catch((err) => console.error(`Failed to load URL ${devUrl}:`, err))

		mainWindow.webContents.openDevTools() // Open DevTools automatically in development
		mainWindow.webContents.on(
			"did-fail-load",
			(event, errorCode, errorDescription, validatedURL) => {
				console.error(
					`Failed to load URL ${validatedURL}: ${errorDescription} (Code: ${errorCode})`
				)
			}
		)
	}

	// --- Window Close Confirmation Logic ---
	let isExiting = false // Flag to prevent multiple exit confirmation dialogs
	mainWindow.on("close", async (event) => {
		if (!mainWindow?.isMainWindow) {
			// If this isn't the main window (e.g., auth window), allow normal close
			return
		}
		if (!isExiting) {
			event.preventDefault() // Prevent immediate close
			isExiting = true
			try {
				const { response } = await dialog.showMessageBox(mainWindow, {
					type: "question",
					buttons: ["Yes", "No"],
					defaultId: 1, // "No" is default
					title: "Confirm Exit",
					message: "Are you sure you want to quit Sentient?"
				})

				if (response === 0) {
					// "Yes"
					console.log("User confirmed application exit.")
					await dialog.showMessageBox(mainWindow, {
						type: "info",
						message: "Thank you for using Sentient! Closing now...",
						title: "Quitting Sentient"
					})
					mainWindow.isMainWindow = false // Unmark as main before quitting
					setTimeout(() => app.quit(), 500) // Allow time for dialogs, then quit
				} else {
					// "No" or dialog dismissed
					console.log("User cancelled application exit.")
					isExiting = false // Reset flag
				}
			} catch (error) {
				console.error("Error during exit confirmation dialog:", error)
				dialog.showErrorBox(
					"Exit Error",
					`An error occurred during the shutdown process: ${error.message}.`
				)
				isExiting = false // Reset flag on error
			}
		}
	})

	// On Window Closed: Clean up resources
	mainWindow.on("closed", () => {
		if (mainWindow?.isMainWindow) {
			// Only nullify if it was the main window instance that closed
			mainWindow = null
			// Close WebSocket connection if open
			if (ws && ws.readyState === WebSocket.OPEN) {
				console.log(
					"Closing WebSocket connection as main window is closed."
				)
				ws.close()
			}
		}
	})
}

// --- User Status and Validity Checks (Production Mode Only) ---

/**
 * Performs a series of checks on user-specific information in production mode.
 * This includes pricing, referral status, beta status, credits, and Google credentials.
 * Requires `userId` obtained from a validated profile.
 * If any check fails, it logs an error, triggers a logout, and re-throws the error.
 *
 * @throws {Error} If `userId` is not available or any check fails.
 */
const checkUserInfo = async () => {
	// if (!app.isPackaged) return // Only run in production
	console.log("PRODUCTION MODE: Performing User Information checks...")

	const userId = getUserIdFromProfile() // Get ID from freshly validated profile
	if (!userId) {
		// This should ideally be caught by `checkAuthStatus` first
		throw new Error(
			"User ID not available for user info checks. Authentication might have failed."
		)
	}

	console.log(
		`PRODUCTION MODE: Performing user-specific checks for user: ${userId}`
	)
	try {
		// These functions now interact with Keytar using the `userId` for user-specific data
		await checkPricingStatus(userId)
		await checkReferralCodeAndStatus(userId)
		await checkBetaUserStatus(userId)
		await resetCreditsIfNecessary(userId)
		await checkGoogleCredentials(userId) // Assumes this backend check uses the authenticated user
		console.log(
			`PRODUCTION MODE: User information checks passed successfully for user ${userId}.`
		)
	} catch (error) {
		console.error(
			`PRODUCTION MODE: Error during user info check for ${userId}: ${error.message}. Triggering logout.`
		)
		// Close all windows except logout/auth windows and show logout window
		BrowserWindow.getAllWindows().forEach((win) => {
			if (!win.isLogoutWindow) win.close()
		})
		createLogoutWindow() // Initiate logout process
		throw error // Propagate error to stop further startup actions if needed
	}
}

/**
 * Main validity check function for production mode.
 * Ensures authentication status is valid and then performs user-specific info checks.
 * If checks fail, it attempts to close existing windows and open the authentication window.
 *
 * @throws {Error} If `checkAuthStatus` or `checkUserInfo` fails.
 */
export const checkValidity = async () => {
	console.log(
		`${app.isPackaged ? "PRODUCTION" : "DEVELOPMENT"} MODE: Performing application validity checks...`
	)
	try {
		await checkAuthStatus() // Ensures tokens are fresh and profile (with userId) is set

		if (app.isPackaged) {
			// Only run detailed user info checks in production
			await checkUserInfo() // Uses the userId from the (now validated) profile
		}

		console.log(
			`${app.isPackaged ? "PRODUCTION" : "DEVELOPMENT"} MODE: All validity checks passed successfully.`
		)

		// If all checks passed, ensure the main application window is open.
		// This is crucial for the flow where the auth window successfully authenticates.
		if (!mainWindow || mainWindow.isDestroyed()) {
			console.log(
				"Validity checks passed, main window not found or destroyed. Creating it now."
			)
			createAppWindow()
		} else if (mainWindow.isMinimized()) {
			mainWindow.restore()
			mainWindow.focus()
		} else {
			mainWindow.focus()
		}
	} catch (error) {
		console.error(
			`${app.isPackaged ? "PRODUCTION" : "DEVELOPMENT"} MODE: Validity check sequence failed: ${error.message}. Redirecting to authentication window.`
		)
		// Close main window if it exists and isn't already being closed
		if (
			mainWindow &&
			!mainWindow.isDestroyed() &&
			mainWindow.isMainWindow
		) {
			console.log("Closing main window due to validity check failure.")
			// Set a flag to prevent close confirmation dialog during this forced close
			// Assuming you have a way to bypass it, or just call destroy directly.
			// For simplicity here, we'll just close. If you have confirmation, you might need
			// a flag like `isExitingDueToAuthFailure = true` to bypass it.
			mainWindow.destroy() // Use destroy to bypass 'close' event handlers if needed
			mainWindow = null
		}

		// Close other windows except auth/logout windows
		BrowserWindow.getAllWindows().forEach((win) => {
			if (
				win &&
				!win.isDestroyed() &&
				!win.isAuthWindow &&
				!win.isLogoutWindow
			) {
				win.close()
			}
		})
		createAuthWindow() // Guide user to re-authenticate
		throw error // Propagate error
	}
}

/**
 * Checks and ensures the user's authentication status by attempting to refresh tokens.
 * Sets the global profile and access token if successful.
 *
 * @throws {Error} If token refresh fails or essential profile/token data is missing post-refresh.
 */
const checkAuthStatus = async () => {
	console.log(
		"PRODUCTION MODE: Checking authentication status via token refresh..."
	)
	try {
		await refreshTokens() // Attempts to refresh tokens; sets profile/accessToken via auth.js
		const userProfile = getProfile()
		const token = getAccessToken()
		// After refresh, profile, user ID (sub), and token must be available
		if (!userProfile || !userProfile.sub || !token) {
			throw new Error(
				"Token refresh was successful, but critical profile information (user ID) or access token is missing."
			)
		}
		console.log(
			`PRODUCTION MODE: User is authenticated (ID: ${userProfile.sub}). Access token has been refreshed.`
		)
	} catch (error) {
		console.error(
			`PRODUCTION MODE: Authentication status check failed: ${error.message}`
		)
		throw new Error("User is not authenticated or token refresh failed.")
	}
}

/**
 * Checks the user's referral code and referrer status, fetching from server if not found locally.
 * Uses `userId` for Keytar operations.
 * @param {string} userId - The ID of the user to check.
 * @throws {Error} If checking referral information fails.
 */
const checkReferralCodeAndStatus = async (userId) => {
	console.log(
		`PRODUCTION MODE: Checking Referral Code and Status for user ${userId}...`
	)
	try {
		let referralCode = await getReferralCodeFromKeytar(userId) // User-specific Keytar
		if (!referralCode) {
			console.log(
				"PRODUCTION MODE: Referral code not found locally for user. Fetching from server..."
			)
			referralCode = await fetchAndSetReferralCode() // Fetches for current (profile.sub) user, sets in Keytar
		}
		let referrerStatus = await getReferrerStatusFromKeytar(userId) // User-specific Keytar
		if (referrerStatus === null) {
			// Check for null explicitly, as false is a valid status
			console.log(
				"PRODUCTION MODE: Referrer status not found locally for user. Fetching from server..."
			)
			referrerStatus = await fetchAndSetReferrerStatus() // Fetches and sets in Keytar
		}
		console.log(
			`PRODUCTION MODE: Referral code/status check complete for user ${userId}. Code: ${referralCode}, Status: ${referrerStatus}`
		)
	} catch (error) {
		console.error(
			`PRODUCTION MODE: Error checking referral code/status for user ${userId}: ${error.message}`
		)
		throw new Error(`Referral code/status check failed for user ${userId}.`)
	}
}

/**
 * Checks the user's beta status, fetching from server if not found locally.
 * Uses `userId` for Keytar operations.
 * @param {string} userId - The ID of the user to check.
 * @throws {Error} If checking beta status fails.
 */
const checkBetaUserStatus = async (userId) => {
	console.log(
		`PRODUCTION MODE: Checking Beta User Status for user ${userId}...`
	)
	try {
		let betaUserStatus = await getBetaUserStatusFromKeytar(userId) // User-specific Keytar
		if (betaUserStatus === null) {
			console.log(
				"PRODUCTION MODE: Beta user status not found locally. Fetching from server..."
			)
			betaUserStatus = await fetchAndSetBetaUserStatus() // Fetches and sets in Keytar
		}
		console.log(
			`PRODUCTION MODE: Beta user status check complete for user ${userId}. Status: ${betaUserStatus}`
		)
	} catch (error) {
		console.error(
			`PRODUCTION MODE: Error checking beta user status for ${userId}: ${error.message}`
		)
		throw new Error(`Beta user status check failed for user ${userId}.`)
	}
}

/**
 * Checks the user's pricing plan and last activity check-in.
 * Triggers logout if session is deemed expired due to inactivity (e.g., > 7 days).
 * Uses `userId` for Keytar operations.
 * @param {string} userId - The ID of the user to check.
 * @throws {Error} If pricing status check fails or session expires.
 */
const checkPricingStatus = async (userId) => {
	console.log(
		`PRODUCTION MODE: Checking Pricing Status and Activity for user ${userId}...`
	)
	try {
		const pricing = await getPricingFromKeytar(userId) // User-specific Keytar
		const lastCheckin = await getCheckinFromKeytar(userId) // User-specific Keytar

		if (!pricing || !lastCheckin) {
			console.warn(
				"PRODUCTION MODE: Pricing plan or last check-in timestamp not found locally. Fetching user role from server..."
			)
			// Fetches role (which determines pricing) based on current authenticated user (profile.sub)
			// and sets it in Keytar for this userId.
			await fetchAndSetUserRole()
			console.log(
				"PRODUCTION MODE: User role fetched. Pricing/activity will be re-checked on next validity pass or app restart."
			)
			return // Exit check for now, will be re-evaluated
		}

		const currentTimestamp = Math.floor(Date.now() / 1000)
		const sevenDaysInSeconds = 7 * 24 * 60 * 60 // Inactivity threshold

		if (currentTimestamp - lastCheckin >= sevenDaysInSeconds) {
			console.warn(
				`PRODUCTION MODE: User session for ${userId} has expired due to inactivity (last check-in: ${new Date(lastCheckin * 1000).toISOString()}). Triggering logout.`
			)
			throw new Error("User session expired due to inactivity.") // This will be caught and trigger logout
		}
		console.log(
			`PRODUCTION MODE: Pricing status and activity check complete for user ${userId}. Plan: ${pricing}, Last check-in: ${new Date(lastCheckin * 1000).toISOString()}`
		)
	} catch (error) {
		console.error(
			`PRODUCTION MODE: Error during pricing/session check for user ${userId}: ${error.message}`
		)
		// Re-throw specific inactivity error or a generic one
		throw error.message.includes("inactivity")
			? error
			: new Error(`Pricing status check failed for user ${userId}.`)
	}
}

/**
 * Resets the user's daily "pro" credits if it's a new day.
 * Credits depend on whether the user is a referrer.
 * Uses `userId` for Keytar operations.
 * @param {string} userId - The ID of the user whose credits to check/reset.
 * @throws {Error} If checking/resetting credits fails.
 */
const resetCreditsIfNecessary = async (userId) => {
	console.log(
		`PRODUCTION MODE: Checking Pro Credits reset for user ${userId}...`
	)
	try {
		const currentDateString = new Date().toDateString() // e.g., "Mon Jan 01 2024"
		const lastCreditsCheckinTimestamp =
			await getCreditsCheckinFromKeytar(userId) // User-specific
		const lastCreditsCheckinDateString = lastCreditsCheckinTimestamp
			? new Date(lastCreditsCheckinTimestamp * 1000).toDateString()
			: null

		if (lastCreditsCheckinDateString !== currentDateString) {
			console.log(
				`PRODUCTION MODE: It's a new day or first check for ${userId}. Resetting proCredits...`
			)
			const isUserAReferrer = await getReferrerStatusFromKeytar(userId) // User's own referrer status
			const creditsToSet = isUserAReferrer === true ? 10 : 5 // More credits if they are a referrer

			await setCreditsInKeytar(userId, creditsToSet) // Set credits for this user
			await setCreditsCheckinInKeytar(userId) // Update today's check-in timestamp for this user

			console.log(
				`PRODUCTION MODE: proCredits reset to ${creditsToSet} for user ${userId}.`
			)
		} else {
			console.log(
				`PRODUCTION MODE: Pro Credits already checked/reset today for user ${userId}.`
			)
		}
	} catch (error) {
		console.error(
			`PRODUCTION MODE: Error during proCredits reset check for user ${userId}: ${error.message}`
		)
		throw new Error(
			`Error checking/resetting proCredits for user ${userId}.`
		)
	}
}

/**
 * Checks the validity of Google API credentials by making a backend call.
 * The backend `/authenticate-google` endpoint handles the actual credential validation.
 * Uses `userId` (implicitly via token) for the backend call.
 * @param {string} userId - The ID of the user (for logging and context).
 * @throws {Error} If access token is missing or Google credentials check fails.
 */
const checkGoogleCredentials = async (userId) => {
	console.log(
		`PRODUCTION MODE: Checking Google API Credentials via server for user ${userId}...`
	)
	const authHeader = getAuthHeader() // Get Bearer token header
	if (!authHeader) {
		throw new Error(
			"Cannot check Google credentials: Access token is missing. User might be logged out."
		)
	}

	try {
		const googleAuthCheckUrl = `${process.env.APP_SERVER_URL || "http://127.0.0.1:5000"}/authenticate-google`
		const response = await fetch(googleAuthCheckUrl, {
			method: "POST", // Backend expects POST, user identified by token
			headers: { "Content-Type": "application/json", ...authHeader }
			// Body might not be needed if the backend uses the user_id from the JWT token
		})

		if (!response.ok) {
			let errorDetails = `Server responded with status ${response.status}`
			try {
				const data = await response.json()
				errorDetails = data.error || data.detail || errorDetails // Prefer specific error from backend
			} catch (e) {
				/* Ignore if response is not JSON */
			}
			throw new Error(`Google credentials check failed: ${errorDetails}`)
		}

		const data = await response.json()
		if (data.success) {
			console.log(
				`PRODUCTION MODE: Google credentials check successful for user ${userId}.`
			)
		} else {
			throw new Error(
				data.error ||
					`Google credentials check indicated failure for user ${userId}. Details: ${JSON.stringify(data)}`
			)
		}
	} catch (error) {
		console.error(
			`PRODUCTION MODE: Error during Google credentials check for user ${userId}: ${error.message}`
		)
		throw new Error(
			`Error checking Google credentials for user ${userId}: ${error.message}`
		)
	}
}

// --- Single Instance Lock ---
// Ensures only one instance of the application can run at a time.
const gotTheLock = app.requestSingleInstanceLock()
if (!gotTheLock) {
	app.quit() // Quit if another instance is already running
} else {
	// If this is the primary instance, focus main window on second instance attempt
	app.on("second-instance", () => {
		if (mainWindow) {
			if (mainWindow.isMinimized()) mainWindow.restore()
			mainWindow.focus()
		}
	})
}

// --- Auto Update Logic ---
/**
 * Configures and checks for application updates in production mode.
 * Sets the update channel (beta/latest) based on the user's beta status.
 * Creates the main application window before initiating the update check.
 */
const checkForUpdates = async () => {
	if (!app.isPackaged) return // Auto-updates only for packaged apps
	console.log(
		"PRODUCTION MODE: Configuring and checking for application updates..."
	)

	let isBetaUser = false
	const userId = getUserIdFromProfile() // Get current user's ID for beta status check

	try {
		// Check beta status from Keytar, specific to the logged-in user
		isBetaUser = userId
			? (await getBetaUserStatusFromKeytar(userId)) || false
			: false
		console.log(
			`PRODUCTION MODE: Update check configuration: User ${userId || "N/A"} is on ${isBetaUser ? "beta" : "stable"} channel (based on local Keytar).`
		)
	} catch (error) {
		console.error(
			"PRODUCTION MODE: Failed to get beta status from Keytar for update check, defaulting to stable channel.",
			error
		)
	}

	// Configure `electron-updater` feed URL and channel
	const feedOptions = {
		provider: "github",
		owner: "existence-master", // GitHub repository owner
		repo: "Sentient" // GitHub repository name
	}
	autoUpdater.setFeedURL(feedOptions)
	autoUpdater.allowPrerelease = isBetaUser // Allow prereleases if user is on beta channel
	autoUpdater.channel = isBetaUser ? "beta" : "latest" // Set update channel

	createAppWindow() // Create the main window *before* checking for updates

	try {
		console.log(
			`PRODUCTION MODE: Initiating update check for '${autoUpdater.channel}' channel...`
		)
		await autoUpdater.checkForUpdatesAndNotify() // Check and show system notification if update available
	} catch (error) {
		console.error(
			"PRODUCTION MODE: Error during auto-update check:",
			error.message
		)
		// Update check errors are generally not critical for app startup unless forced.
	}
}

// --- Application Startup Logic ---
/**
 * Main function to start the application.
 * Handles different startup flows for production (packaged) and development environments.
 * In production, includes update checks and validity checks.
 */
const startApp = async () => {
	if (app.isPackaged) {
		// --- PRODUCTION LOGIC (Seems mostly fine, but checkValidity needs to ensure mainWindow exists on success) ---
		console.log("PRODUCTION MODE: Starting application startup sequence.")
		await checkForUpdates() // This creates the main window

		const validityCheckDelay = 5000
		console.log(
			`PRODUCTION MODE: Scheduling application validity checks in ${validityCheckDelay / 1000} seconds.`
		)

		setTimeout(async () => {
			if (process.env.UPDATING !== "true") {
				console.log(
					"PRODUCTION MODE: Performing scheduled application validity checks..."
				)
				try {
					await checkValidity() // This will handle auth and ensure main window is shown if valid
					console.log(
						"PRODUCTION MODE: Application validity checks completed successfully."
					)
				} catch (error) {
					console.error(
						"PRODUCTION MODE: Application validity check sequence failed:",
						error.message
					)
					// checkValidity already handles opening createAuthWindow on failure
				}
			} else {
				console.log(
					"PRODUCTION MODE: Skipping validity checks because an update is in progress (UPDATING=true)."
				)
			}
		}, validityCheckDelay)
	} else {
		// --- DEVELOPMENT LOGIC (This is where the main fix is) ---
		console.log(
			"DEVELOPMENT MODE: Starting application (skipping updates and production validity checks)."
		)
		let isAuthenticated = false
		try {
			console.log(
				"[ELECTRON] [DEV_MODE] Attempting to refresh tokens on startup..."
			)
			await refreshTokens() // Try to restore session
			if (getAccessToken() && getProfile()?.sub) {
				console.log(
					`[ELECTRON] [DEV_MODE] Token refresh successful. User ID: ${getProfile().sub}`
				)
				isAuthenticated = true
			} else {
				console.log(
					"[ELECTRON] [DEV_MODE] No valid session found after token refresh attempt."
				)
			}
		} catch (error) {
			console.warn(
				`[ELECTRON] [DEV_MODE] Failed to refresh tokens on startup (this is normal if not logged in or refresh token is expired/invalid): ${error.message}`
			)
			// isAuthenticated remains false
		}

		if (isAuthenticated) {
			console.log(
				"[ELECTRON] [DEV_MODE] User authenticated. Creating main application window."
			)
			createAppWindow() // Creates window, loads URL.
			mainWindow?.webContents.on("did-finish-load", () => {
				console.log(
					"DEVELOPMENT MODE: Main application window finished loading."
				)
				// connectWebSocket() is called within createAppWindow after URL load
			})
		} else {
			console.log(
				"[ELECTRON] [DEV_MODE] User not authenticated. Creating authentication window."
			)
			createAuthWindow() // Show login screen
			// The auth.js createAuthWindow's callback will handle creating the app window on successful login
		}
	}
}

// --- Electron App Lifecycle Event Handlers ---
app.on("ready", startApp) // Start the application when Electron is ready

app.on("window-all-closed", () => {
	// Quit the app when all windows are closed, except on macOS (darwin)
	if (process.platform !== "darwin") app.quit()
})

app.on("activate", () => {
	// Re-create the main window if the app is activated and no windows are open (macOS dock click)
	if (BrowserWindow.getAllWindows().length === 0) startApp()
})

// --- AutoUpdater Event Listeners (Production Only) ---
// These listeners handle events from `electron-updater` to provide feedback to the user.
if (app.isPackaged) {
	autoUpdater.on("update-available", (info) => {
		console.log("PRODUCTION MODE: Update available. Version:", info.version)
		process.env.UPDATING = "true" // Set flag to indicate update process
		mainWindow?.webContents.send("update-available", info) // Notify renderer
	})

	autoUpdater.on("update-not-available", (info) => {
		console.log("PRODUCTION MODE: No update available at this time.", info)
	})

	autoUpdater.on("error", (err) => {
		console.error(
			"PRODUCTION MODE: Error during auto-update process:",
			err.message
		)
		process.env.UPDATING = "false" // Reset update flag
		mainWindow?.webContents.send("update-error", err.message)
		mainWindow?.webContents.send("update-progress", {
			percent: 0,
			error: true
		}) // Indicate error in progress
	})

	autoUpdater.on("download-progress", (progressObj) => {
		console.log(
			`PRODUCTION MODE: Update download progress: ${progressObj.percent.toFixed(2)}%`
		)
		mainWindow?.webContents.send("update-progress", progressObj) // Send progress to renderer
	})

	autoUpdater.on("update-downloaded", (info) => {
		console.log(
			"PRODUCTION MODE: Update downloaded successfully. Version:",
			info.version
		)
		process.env.UPDATING = "false" // Reset update flag
		mainWindow?.webContents.send("update-downloaded", info) // Notify renderer that download is complete
		// The renderer will typically prompt the user to restart via `ipcMain.on("restart-app")`.
	})
}

// --- Inter-Process Communication (IPC) Handlers ---
// These handlers allow the renderer process to request actions or data from the main process.

// IPC: Restart App (for AutoUpdate)
ipcMain.on("restart-app", () => {
	if (app.isPackaged) {
		console.log(
			"IPC: Received restart-app command. Quitting and installing update."
		)
		autoUpdater.quitAndInstall() // Triggered by renderer after update is downloaded
	} else {
		console.log(
			"IPC: Received restart-app command in DEV mode (no-op for updates)."
		)
	}
})

// IPC: Get User Profile
ipcMain.handle("get-profile", async () => {
	try {
		const userProfile = getProfile() // Get current profile from auth utility
		if (!app.isPackaged && !userProfile) {
			// Provide mock data for development if no profile exists (e.g., auth skipped)
			console.log(
				"IPC: get-profile (DEV mode, no real profile) - Returning mock data."
			)
			return {
				sub: "dev-user-123",
				given_name: "Dev User",
				picture: null
			}
		}
		if (!userProfile) {
			console.log(
				"IPC: get-profile - No user profile available (user likely not logged in)."
			)
			return null
		}
		console.log(
			"IPC: get-profile - Returning user profile for:",
			userProfile.sub
		)
		return userProfile
	} catch (error) {
		console.error("IPC Error: Failed to handle get-profile:", error)
		return null
	}
})

// IPC: Get Private Data (Placeholder - Requires specific implementation)
ipcMain.handle("get-private-data", async () => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader() // Get auth header for backend call

	if (!userId || !authHeader) {
		console.error("IPC: get-private-data - User not authenticated.")
		return { error: "User not authenticated.", status: 401 }
	}

	if (!app.isPackaged) {
		console.log(
			`IPC: get-private-data (DEV mode) - Returning mock private data for user ${userId}.`
		)
		return { mockData: `Some development private data for user ${userId}.` }
	}

	try {
		console.log(
			`PRODUCTION MODE: Handling get-private-data request for user: ${userId}`
		)
		// Assuming `getPrivateData` is an async function that makes a backend call
		// It might need modification to accept `userId` or use the `authHeader`.
		return await getPrivateData(/* pass userId or authHeader if needed */)
	} catch (error) {
		console.error(
			`IPC Error: Failed to get private data for user ${userId}:`,
			error
		)
		return { error: error.message, status: 500 }
	}
})

// IPC: Get Local Beta User Status (from Keytar)
ipcMain.handle("get-beta-user-status-local", async () => {
	const userId = getUserIdFromProfile() // Needed for user-specific Keytar entry

	if (!userId && app.isPackaged) {
		// In production, userId is required
		console.error(
			"IPC: get-beta-user-status-local (PROD) - User ID missing."
		)
		return null
	}

	try {
		if (!app.isPackaged) {
			// Development mock
			console.log(
				"IPC: get-beta-user-status-local (DEV) - Returning mock status (true)."
			)
			return true
		}
		const betaUserStatus = await getBetaUserStatusFromKeytar(userId) // Pass userId
		console.log(
			`IPC: get-beta-user-status-local (PROD) for user ${userId}: ${betaUserStatus}`
		)
		return betaUserStatus
	} catch (error) {
		console.error(
			`IPC Error: Failed to fetch local beta user status for user ${userId || "N/A"}:`,
			error
		)
		return null
	}
})

// IPC: Log Out
ipcMain.on("log-out", () => {
	console.log("IPC: Log-out command received. Initiating logout process.")
	// Close WebSocket if open
	if (ws && ws.readyState === WebSocket.OPEN) {
		console.log("IPC: Closing WebSocket connection due to logout.")
		ws.close()
	}
	// Close all application windows except dedicated logout/auth windows
	BrowserWindow.getAllWindows().forEach((win) => {
		if (!win.isLogoutWindow && !win.isAuthWindow) {
			win.close()
		}
	})
	createLogoutWindow() // `createLogoutWindow` handles token clearing and showing the Auth0 logout page.
})

// IPC: Fetch Local Pricing Plan (from Keytar)
ipcMain.handle("fetch-pricing-plan", async () => {
	const userId = getUserIdFromProfile()
	if (!userId && app.isPackaged) {
		console.warn(
			"IPC: fetch-pricing-plan (PROD) - User ID missing, defaulting to 'free'."
		)
		return "free"
	}

	if (!app.isPackaged) {
		// Development mock
		console.log(
			"IPC: fetch-pricing-plan (DEV) - Returning mock plan ('pro')."
		)
		return "pro"
	}

	try {
		const pricing = await getPricingFromKeytar(userId) // Pass userId
		console.log(
			`IPC: fetch-pricing-plan (PROD) for user ${userId}: ${pricing || "free"}`
		)
		return pricing || "free" // Default to "free" if not found
	} catch (error) {
		console.error(
			`IPC Error: Failed to fetch local pricing plan for user ${userId}:`,
			error
		)
		return "free" // Default on error
	}
})

// IPC: Fetch Local Pro Credits (from Keytar)
ipcMain.handle("fetch-pro-credits", async () => {
	const userId = getUserIdFromProfile()
	if (!userId && app.isPackaged) {
		console.warn(
			"IPC: fetch-pro-credits (PROD) - User ID missing, defaulting to 0 credits."
		)
		return 0
	}

	if (!app.isPackaged) {
		// Development mock
		console.log(
			"IPC: fetch-pro-credits (DEV) - Returning mock credits (999)."
		)
		return 999
	}

	try {
		const credits = await getCreditsFromKeytar(userId) // Pass userId
		console.log(
			`IPC: fetch-pro-credits (PROD) for user ${userId}: ${credits !== null ? credits : 0}`
		)
		return credits !== null ? credits : 0 // Default to 0 if not found
	} catch (error) {
		console.error(
			`IPC Error: Failed to fetch local pro credits for user ${userId}:`,
			error
		)
		return 0 // Default on error
	}
})

// IPC: Decrement Local Pro Credits (in Keytar)
ipcMain.handle("decrement-pro-credits", async () => {
	const userId = getUserIdFromProfile()
	if (!userId && app.isPackaged) {
		console.error(
			"IPC: decrement-pro-credits (PROD) - Cannot decrement, User ID missing."
		)
		return false
	}

	if (!app.isPackaged) {
		// Development simulation
		console.log("IPC: decrement-pro-credits (DEV) - Simulated decrement.")
		return true
	}

	try {
		let credits = await getCreditsFromKeytar(userId) // Pass userId
		credits = Math.max(0, (credits || 0) - 1) // Ensure credits don't go below 0
		await setCreditsInKeytar(userId, credits) // Pass userId
		console.log(
			`IPC: decrement-pro-credits (PROD) - Credits updated to ${credits} for user ${userId}.`
		)
		return true
	} catch (error) {
		console.error(
			`IPC Error: Failed to decrement local pro credits for user ${userId}:`,
			error
		)
		return false
	}
})

// IPC: Fetch Local Referral Code (from Keytar)
ipcMain.handle("fetch-referral-code", async () => {
	const userId = getUserIdFromProfile()
	if (!userId && app.isPackaged) return null

	if (!app.isPackaged) {
		// Development mock
		console.log(
			"IPC: fetch-referral-code (DEV) - Returning mock code 'DEVREFERRAL'."
		)
		return "DEVREFERRAL"
	}

	try {
		const referralCode = await getReferralCodeFromKeytar(userId) // Pass userId
		console.log(
			`IPC: fetch-referral-code (PROD) for user ${userId}: ${referralCode}`
		)
		return referralCode
	} catch (error) {
		console.error(
			`IPC Error: Failed to fetch local referral code for user ${userId}:`,
			error
		)
		return null
	}
})

// IPC: Fetch Local Referrer Status (from Keytar)
ipcMain.handle("fetch-referrer-status", async () => {
	const userId = getUserIdFromProfile()
	if (!userId && app.isPackaged) return null

	if (!app.isPackaged) {
		// Development mock
		console.log(
			"IPC: fetch-referrer-status (DEV) - Returning mock status (true)."
		)
		return true
	}

	try {
		const referrerStatus = await getReferrerStatusFromKeytar(userId) // Pass userId
		console.log(
			`IPC: fetch-referrer-status (PROD) for user ${userId}: ${referrerStatus}`
		)
		return referrerStatus
	} catch (error) {
		console.error(
			`IPC Error: Failed to fetch local referrer status for user ${userId}:`,
			error
		)
		return null
	}
})

// IPC: Fetch Local Beta User Status (from Keytar, similar to get-beta-user-status-local but for general use)
ipcMain.handle("fetch-beta-user-status", async () => {
	const userId = getUserIdFromProfile()
	if (!userId && app.isPackaged) return null

	const isDev = !app.isPackaged
	if (isDev)
		console.log(
			"IPC: Handling fetch-beta-user-status (DEV mode, local Keytar)."
		)

	try {
		const betaUserStatus = await getBetaUserStatusFromKeytar(userId) // Pass userId
		console.log(
			`IPC: fetch-beta-user-status (${isDev ? "DEV" : "PROD"}) for user ${userId || "dev"}: ${betaUserStatus}`
		)
		if (isDev && betaUserStatus === null) return true // Dev default if not set
		return betaUserStatus
	} catch (error) {
		console.error(
			`IPC Error: Failed to fetch local beta status (${isDev ? "DEV" : "PROD"}):`,
			error
		)
		return null
	}
})

// --- Backend Interaction IPC Handlers (Include Auth Header) ---
// These handlers make requests to the backend server, including the Authorization header.

// IPC: Set Referrer (Backend Call)
ipcMain.handle("set-referrer", async (_event, { referralCode }) => {
	const userId = getUserIdFromProfile() // ID of the user making the request
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: set-referrer - User not authenticated.")
		return { error: "User not authenticated.", status: 401 }
	}

	if (!app.isPackaged) {
		// Development skip
		console.log("IPC: set-referrer (DEV) - Simulated successful call.")
		return {
			message: "DEV: Referrer status setting simulated.",
			status: 200
		}
	}

	try {
		const apiUrl = `${process.env.APP_SERVER_URL}/get-user-and-set-referrer-status`
		console.log(
			`IPC: User ${userId} sending referral code '${referralCode}' to backend: ${apiUrl}`
		)
		const response = await fetch(apiUrl, {
			method: "POST",
			headers: { "Content-Type": "application/json", ...authHeader },
			// Backend uses token to identify the user making the request.
			// The body contains the referral code of the user whose status needs to be updated.
			body: JSON.stringify({ referral_code: referralCode })
		})

		if (!response.ok) {
			const errorText = await response.text()
			let detail = errorText
			try {
				detail = JSON.parse(errorText).detail || detail
			} catch {
				/* ignore parse error */
			}
			console.error(
				`IPC Error: Set Referrer API call failed for user ${userId}. Status: ${response.status}, Details: ${detail}`
			)
			return { error: `API Error: ${detail}`, status: response.status }
		}

		const result = await response.json()
		console.log(
			`IPC: Set Referrer API call successful for user ${userId}. Response:`,
			result
		)
		// Optionally, re-fetch the current user's own referrer status if it might have changed as a result.
		// await fetchAndSetReferrerStatus(); // If the backend logic could affect the requesting user's status
		return {
			message:
				result.message || "Referrer request processed successfully.",
			status: 200
		}
	} catch (error) {
		console.error(
			`IPC Error: Exception in set-referrer handler for user ${userId}:`,
			error
		)
		return { error: error.message, status: 500 }
	}
})

// IPC: Invert Beta Status (Backend Call)
ipcMain.handle("invert-beta-user-status", async () => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: invert-beta-user-status - User not authenticated.")
		return { error: "User not authenticated.", status: 401 }
	}

	if (!app.isPackaged) {
		// Development skip
		console.log(
			"IPC: invert-beta-user-status (DEV) - Simulated successful call."
		)
		return { message: "DEV: Beta status inversion simulated.", status: 200 }
	}

	try {
		const apiUrl = `${process.env.APP_SERVER_URL}/get-user-and-invert-beta-user-status`
		console.log(
			`IPC: User ${userId} requesting beta status inversion via backend: ${apiUrl}`
		)
		const response = await fetch(apiUrl, {
			method: "POST",
			headers: { "Content-Type": "application/json", ...authHeader }
			// Backend identifies user via token; body might be empty or specific if needed.
		})

		if (!response.ok) {
			const errorText = await response.text()
			let detail = errorText
			try {
				detail = JSON.parse(errorText).detail || detail
			} catch {
				/* ignore */
			}
			console.error(
				`IPC Error: Invert Beta API call failed for user ${userId}. Status: ${response.status}, Details: ${detail}`
			)
			return { error: `API Error: ${detail}`, status: response.status }
		}

		const result = await response.json()
		console.log(
			`IPC: Invert Beta API call successful for user ${userId}. Response:`,
			result
		)

		// Update local status in Keytar immediately after successful backend call
		const currentLocalStatus = await getBetaUserStatusFromKeytar(userId)
		const newLocalStatus = !currentLocalStatus // Invert
		await setBetaUserStatusInKeytar(userId, newLocalStatus) // Set for this user
		console.log(
			`IPC: Updated local beta status in Keytar to: ${newLocalStatus} for user ${userId}.`
		)

		return {
			message: result.message || "Beta status inverted successfully.",
			status: 200
		}
	} catch (error) {
		console.error(
			`IPC Error: Exception in invert-beta-user-status handler for user ${userId}:`,
			error
		)
		return { error: error.message, status: 500 }
	}
})

// IPC: Set User Data (Backend Call)
ipcMain.handle("set-user-data", async (_event, args) => {
	const { data } = args
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: set-user-data - User not authenticated.")
		return { error: "User not authenticated.", status: 401 }
	}

	console.log(
		`IPC: set-user-data called for user ${userId}. Data keys: ${Object.keys(data || {}).join(", ")}`
	)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/set-user-data`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader },
				body: JSON.stringify({ data: data }) // Backend gets userId from token
			}
		)

		if (!response.ok) {
			const errorDetail = await response
				.json()
				.catch(() => ({ detail: "Unknown error from server." }))
			console.error(
				`IPC Error: Set User Data API call failed for user ${userId}. Status: ${response.status}, Details:`,
				errorDetail
			)
			return {
				message: `Error setting data: ${errorDetail.detail || response.statusText}`,
				status: response.status
			}
		}
		const result = await response.json()
		console.log(
			`IPC: Set User Data API call successful for user ${userId}.`
		)
		return result // Contains { message, status } from backend
	} catch (error) {
		console.error(
			`IPC Error: Exception in set-user-data handler for user ${userId}:`,
			error
		)
		return {
			message: "Error storing user data.",
			status: 500,
			error: error.message
		}
	}
})

// IPC: Add/Merge DB Data (Backend Call)
ipcMain.handle("add-db-data", async (_event, args) => {
	const { data } = args
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: add-db-data - User not authenticated.")
		return { error: "User not authenticated.", status: 401 }
	}

	console.log(
		`IPC: add-db-data called for user ${userId}. Data keys: ${Object.keys(data || {}).join(", ")}`
	)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/add-db-data`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader },
				body: JSON.stringify({ data: data }) // Backend gets userId from token
			}
		)

		if (!response.ok) {
			const errorDetail = await response
				.json()
				.catch(() => ({ detail: "Unknown error from server." }))
			console.error(
				`IPC Error: Add DB Data API call failed for user ${userId}. Status: ${response.status}, Details:`,
				errorDetail
			)
			return {
				message: `Error adding/merging data: ${errorDetail.detail || response.statusText}`,
				status: response.status
			}
		}
		const result = await response.json()
		console.log(`IPC: Add DB Data API call successful for user ${userId}.`)
		return result
	} catch (error) {
		console.error(
			`IPC Error: Exception in add-db-data handler for user ${userId}:`,
			error
		)
		return {
			message: "Error adding/merging user data.",
			status: 500,
			error: error.message
		}
	}
})

// IPC: Get User Data (Backend Call)
ipcMain.handle("get-user-data", async () => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: get-user-data - User not authenticated.")
		return { error: "User not authenticated.", status: 401 }
	}

	console.log(`IPC: get-user-data called for user ${userId}.`)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/get-user-data`,
			{
				method: "POST", // Kept POST as backend identifies user via token
				headers: { "Content-Type": "application/json", ...authHeader }
				// Body is likely empty if user is identified solely by token.
			}
		)

		if (!response.ok) {
			const errorDetail = await response
				.json()
				.catch(() => ({ detail: "Unknown error from server." }))
			console.error(
				`IPC Error: Get User Data API call failed for user ${userId}. Status: ${response.status}, Details:`,
				errorDetail
			)
			return {
				message: `Error fetching data: ${errorDetail.detail || response.statusText}`,
				status: response.status
			}
		}
		const result = await response.json()
		console.log(
			`IPC: Get User Data API call successful for user ${userId}.`
		)
		return { data: result.data, status: result.status } // Backend returns { data, status }
	} catch (error) {
		console.error(
			`IPC Error: Exception in get-user-data handler for user ${userId}:`,
			error
		)
		return {
			message: "Error fetching user data.",
			status: 500,
			error: error.message
		}
	}
})

// IPC: Fetch Chat History (Backend Call)
ipcMain.handle("fetch-chat-history", async () => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: fetch-chat-history - User not authenticated.")
		return { error: "User not authenticated.", status: 401 }
	}

	console.log(`IPC: fetch-chat-history called for user ${userId}.`)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/get-history`,
			{
				method: "POST", // Changed to POST to align with backend
				headers: { "Content-Type": "application/json", ...authHeader }
				// Body is likely empty.
			}
		)

		if (!response.ok) {
			const errorText = await response.text() // Get raw error text for logging
			console.error(
				`IPC Error: Fetch Chat History API call failed for user ${userId}. Status: ${response.status} ${response.statusText}. Details: ${errorText}`
			)
			return {
				message: `Failed to fetch chat history: ${response.statusText}`,
				status: response.status,
				error: errorText
			}
		}
		const data = await response.json()
		console.log(
			`IPC: Fetch Chat History API call successful for user ${userId}.`
		)
		return {
			messages: data.messages,
			activeChatId: data.activeChatId,
			status: 200
		}
	} catch (error) {
		console.error(
			`IPC Error: Exception in fetch-chat-history handler for user ${userId}:`,
			error
		)
		return {
			message: "Error fetching chat history.",
			status: 500,
			error: error.message
		}
	}
})

// IPC: Send Chat Message (Backend Call, Streaming Response)
ipcMain.handle("send-message", async (_event, { input }) => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error(
			"IPC: send-message - User not authenticated. Cannot send message."
		)
		mainWindow?.webContents.send("chat-error", {
			message: "Cannot send message: User authentication required."
		})
		return { message: "User not authenticated.", status: 401 }
	}

	console.log(
		`IPC: send-message called by user ${userId} with input: "${input.substring(0, 50)}..."`
	)
	let pricing = "free" // Default pricing plan
	let credits = 0 // Default credits

	try {
		// Determine pricing and credits based on environment and user's Keytar data
		if (app.isPackaged) {
			pricing = (await getPricingFromKeytar(userId)) || "free"
			credits = (await getCreditsFromKeytar(userId)) || 0
			console.log(
				`PRODUCTION MODE: User ${userId} sending message. Pricing: ${pricing}, Credits: ${credits}`
			)
		} else {
			pricing = "pro" // Development defaults
			credits = 999
			console.log(
				`DEVELOPMENT MODE: User ${userId} sending message. Mock Pricing: ${pricing}, Mock Credits: ${credits}`
			)
		}

		const payload = { input, pricing, credits } // user_id is implicit via token on backend
		console.log(
			"IPC: Sending payload to backend /chat endpoint:",
			JSON.stringify(payload)
		)

		const response = await fetch(`${process.env.APP_SERVER_URL}/chat`, {
			method: "POST",
			headers: { "Content-Type": "application/json", ...authHeader },
			body: JSON.stringify(payload)
		})

		// --- Handle Streaming Response ---
		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`IPC Error: Backend /chat endpoint responded with error for user ${userId}. Status: ${response.status} ${response.statusText}. Details: ${errorText}`
			)
			mainWindow?.webContents.send("chat-error", {
				message: `Server error: ${response.statusText}. ${errorText}`
			})
			throw new Error(
				`Error sending message: Server responded with status ${response.status}`
			)
		}
		if (!response.body) {
			throw new Error("Response from /chat endpoint has no body (null).")
		}

		const readableStream = response.body
		const decoder = new TextDecoder("utf-8")
		let buffer = "" // Accumulator for partial messages
		let finalProFeatureUsed = false // Track if any pro feature was used in the streamed response
		const targetWindow = mainWindow // Assume current main window for sending stream updates
		if (!targetWindow) {
			throw new Error(
				"Application main window not found. Cannot send stream updates."
			)
		}

		console.log(
			`IPC: Starting to process chat stream for user ${userId}...`
		)
		// Iterate over the chunks from the readable stream
		for await (const chunk of readableStream) {
			buffer += decoder.decode(chunk, { stream: true }) // Decode chunk and append to buffer
			const messages = buffer.split("\n") // Split by newline (ndjson format)
			buffer = messages.pop() || "" // Last part might be incomplete, keep it in buffer

			for (const msgStr of messages) {
				if (msgStr.trim() === "") continue // Skip empty lines
				try {
					const parsedMessage = JSON.parse(msgStr)
					// Route message to renderer based on its type
					if (
						parsedMessage.type === "assistantStream" ||
						parsedMessage.type === "assistantMessage"
					) {
						targetWindow.webContents.send("message-stream", {
							messageId: parsedMessage.messageId,
							token:
								parsedMessage.token ||
								parsedMessage.message ||
								"", // Content
							isFinal:
								parsedMessage.done ||
								parsedMessage.type === "assistantMessage" // Is it the last part?
						})
						if (parsedMessage.proUsed) finalProFeatureUsed = true // Mark if pro feature was used
					} else if (parsedMessage.type === "userMessage") {
						targetWindow.webContents.send("user-message-saved", {
							// Confirmation for UI
							id: parsedMessage.id,
							timestamp: parsedMessage.timestamp
						})
					} else if (parsedMessage.type === "intermediary") {
						targetWindow.webContents.send("intermediary-message", {
							// Thinking, searching, etc.
							message: parsedMessage.message,
							id: parsedMessage.id
						})
					} else if (parsedMessage.type === "error") {
						targetWindow.webContents.send("chat-error", {
							message: parsedMessage.message
						})
					}
				} catch (parseError) {
					console.warn(
						`IPC Warning: Error parsing JSON from stream for user ${userId}: ${parseError}. Message part: "${msgStr}"`
					)
				}
			}
		}

		// Process any remaining content in the buffer (last part of the stream)
		if (buffer.trim()) {
			try {
				const parsedMessage = JSON.parse(buffer)
				if (
					parsedMessage.type === "assistantStream" ||
					parsedMessage.type === "assistantMessage"
				) {
					targetWindow.webContents.send("message-stream", {
						messageId: parsedMessage.messageId,
						token:
							parsedMessage.token || parsedMessage.message || "",
						isFinal: true // Assume final if it's the last buffered part
					})
					if (parsedMessage.proUsed) finalProFeatureUsed = true
				}
			} catch (e) {
				console.error(
					`IPC Error: Failed to parse final buffered content for user ${userId}: ${e}. Buffer: "${buffer}"`
				)
			}
		}

		// Decrement credits if a pro feature was used by a free user (in production)
		if (app.isPackaged && pricing === "free" && finalProFeatureUsed) {
			console.log(
				`PRODUCTION MODE: Decrementing credit for user ${userId} (Free plan, Pro feature used during chat).`
			)
			await ipcMain.handle("decrement-pro-credits") // Uses userId internally now
		} else if (!app.isPackaged && finalProFeatureUsed) {
			console.log(
				`DEVELOPMENT MODE: Pro feature used by ${userId} (simulated). Would decrement credit in production if user is on free plan.`
			)
		}

		console.log(`IPC: Chat message streaming complete for user ${userId}.`)
		return { message: "Streaming complete.", status: 200 }
	} catch (error) {
		console.error(
			`IPC Error: Exception in send-message handler for user ${userId}:`,
			error
		)
		mainWindow?.webContents.send("chat-error", {
			message: `Failed to process your message: ${error.message}`
		})
		return { message: `Error: ${error.message}`, status: 500 }
	}

}
)
// IPC: Build Personality (Multi-step Backend Process)
ipcMain.handle("build-personality", async () => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: build-personality - User not authenticated.")
		return { error: "User not authenticated.", status: 401 }
	}

	console.log(`IPC: build-personality process initiated for user ${userId}.`)
	try {
		// Step 1: Create documents based on profile
		console.log(
			`IPC: Calling backend /create-document for user ${userId}...`
		)
		const documentResponse = await fetch(
			`${process.env.APP_SERVER_URL}/create-document`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader }
				// Body might be empty if backend uses token for user context.
			}
		)
		if (!documentResponse.ok) {
			const errTxt = await documentResponse.text()
			throw new Error(
				`Failed /create-document step. Status: ${documentResponse.status}, Details: ${errTxt}`
			)
		}
		const { personality } = await documentResponse.json() // Expects personality summary
		console.log(
			`IPC: Backend /create-document successful for user ${userId}.`
		)

		// Step 2: Update local DB (user profile) with the generated personality summary
		console.log(
			`IPC: Updating local user profile with personality summary for user ${userId}...`
		)
		const addDataResponse = await fetch(
			`${process.env.APP_SERVER_URL}/add-db-data`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader },
				body: JSON.stringify({ data: { userData: { personality } } }) // Add under userData
			}
		)
		if (!addDataResponse.ok) {
			const errTxt = await addDataResponse.text()
			throw new Error(
				`Failed /add-db-data step for personality. Status: ${addDataResponse.status}, Details: ${errTxt}`
			)
		}
		console.log(
			`IPC: Local user profile updated successfully for user ${userId}.`
		)

		// Step 3: Initiate/Update long-term memories (knowledge graph)
		console.log(
			`IPC: Calling backend /initiate-long-term-memories for user ${userId}...`
		)
		const graphResponse = await fetch(
			`${process.env.APP_SERVER_URL}/initiate-long-term-memories`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader }
				// Body might be empty or include flags like { clear_graph: false }
			}
		)
		if (!graphResponse.ok) {
			const errTxt = await graphResponse.text()
			throw new Error(
				`Failed /initiate-long-term-memories step. Status: ${graphResponse.status}, Details: ${errTxt}`
			)
		}
		console.log(
			`IPC: Backend /initiate-long-term-memories successful for user ${userId}.`
		)

		return {
			message:
				"Personality documents and knowledge graph processed successfully.",
			status: 200
		}
	} catch (error) {
		console.error(
			`IPC Error: Exception in build-personality handler for user ${userId}:`,
			error
		)
		return {
			message: `Error building personality: ${error.message}`,
			status: 500
		}
	}
})

// IPC: Reset Long-Term Memories (Knowledge Graph) (Backend Call)
ipcMain.handle("reset-long-term-memories", async () => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: reset-long-term-memories - User not authenticated.")
		return { error: "User not authenticated.", status: 401 }
	}

	console.log(`IPC: reset-long-term-memories called for user ${userId}.`)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/initiate-long-term-memories`,
			{
				// Backend endpoint might be the same, but with a flag to clear
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader },
				body: JSON.stringify({ clear_graph: true }) // Signal to clear the graph
			}
		)

		if (!response.ok) {
			const errorText = await response.text()
			throw new Error(
				`Failed to reset graph. Status: ${response.status}, Details: ${errorText}`
			)
		}
		const result = await response.json()
		console.log(
			`IPC: Knowledge graph reset successful for user ${userId}. Response:`,
			result
		)
		return {
			message: result.message || "Knowledge graph reset successfully.",
			status: 200
		}
	} catch (error) {
		console.error(
			`IPC Error: Exception in reset-long-term-memories handler for user ${userId}:`,
			error
		)
		return {
			message: `Error resetting knowledge graph: ${error.message}`,
			status: 500
		}
	}
})

// IPC: Customize Long-Term Memories (Knowledge Graph) (Backend Call)
ipcMain.handle(
	"customize-long-term-memories",
	async (_event, { newGraphInfo }) => {
		const userId = getUserIdFromProfile()
		const authHeader = getAuthHeader()
		if (!userId || !authHeader) {
			console.error(
				"IPC: customize-long-term-memories - User not authenticated."
			)
			return { error: "User not authenticated.", status: 401 }
		}

		console.log(
			`IPC: customize-long-term-memories called by user ${userId} with info: "${newGraphInfo.substring(0, 50)}..."`
		)
		let credits = 0
		let pricing = "free"

		try {
			if (app.isPackaged) {
				credits = (await getCreditsFromKeytar(userId)) || 0
				pricing = (await getPricingFromKeytar(userId)) || "free"
				console.log(
					`PRODUCTION MODE: Customizing graph for user ${userId}. Credits: ${credits}, Pricing: ${pricing}`
				)
			} else {
				credits = 999
				pricing = "pro" // Dev defaults
				console.log(
					`DEVELOPMENT MODE: Customizing graph for user ${userId} (mock credits/pricing).`
				)
			}

			const response = await fetch(
				`${process.env.APP_SERVER_URL}/customize-long-term-memories`,
				{
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						...authHeader
					},
					body: JSON.stringify({
						information: newGraphInfo
						// Backend might use pricing/credits to gate features or log usage
						// pricing: pricing,
						// credits: credits,
					})
				}
			)

			if (!response.ok) {
				const errorText = await response.text()
				throw new Error(
					`Failed to customize graph. Status: ${response.status}, Details: ${errorText}`
				)
			}
			const result = await response.json()
			console.log(
				`IPC: Customize Graph API call successful for user ${userId}. Response:`,
				result
			)

			// Check if a pro feature was used and decrement credit if applicable
			// Assuming backend response includes a flag like `result.pro_used`
			if (
				app.isPackaged &&
				pricing === "free" &&
				result?.pro_used === true
			) {
				// Example flag
				console.log(
					`PRODUCTION MODE: Decrementing credit for user ${userId} (Graph customization used a pro feature).`
				)
				await ipcMain.handle("decrement-pro-credits") // Uses userId internally
			}

			return {
				message:
					result.message ||
					"Knowledge graph customized successfully.",
				status: 200
			}
		} catch (error) {
			console.error(
				`IPC Error: Exception in customize-long-term-memories handler for user ${userId}:`,
				error
			)
			return {
				message: `Error customizing knowledge graph: ${error.message}`,
				status: 500
			}
		}
	}
)

// IPC: Delete Subgraph (Backend Call)
ipcMain.handle("delete-subgraph", async (_event, { source_name }) => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: delete-subgraph - User not authenticated.")
		return { status: "failure", error: "User not authenticated." }
	}

	console.log(
		`IPC: delete-subgraph called by user ${userId} for source: ${source_name}`
	)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/delete-subgraph`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader },
				body: JSON.stringify({ source: source_name }) // Backend gets user from token
			}
		)

		const result = await response.json() // Try to parse JSON regardless of status for more info
		if (!response.ok || result.status === "failure") {
			// Check for explicit failure in response too
			const errorMsg =
				result.error ||
				result.detail ||
				`Server error ${response.status}`
			console.error(
				`IPC Error: Delete Subgraph API call failed for user ${userId}. Status: ${response.status}, Error: ${errorMsg}`
			)
			throw new Error(errorMsg)
		}
		console.log(
			`IPC: Delete Subgraph API call successful for user ${userId}. Response:`,
			result
		)
		return result // Expects { message } or similar from backend on success
	} catch (error) {
		console.error(
			`IPC Error: Exception in delete-subgraph handler for user ${userId}:`,
			error
		)
		return { status: "failure", error: error.message }
	}
})

// IPC: Fetch Tasks (Backend Call)
ipcMain.handle("fetch-tasks", async () => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: fetch-tasks - User not authenticated.")
		return { error: "User not authenticated.", tasks: [] } // Return empty tasks for safety
	}

	console.log(`IPC: fetch-tasks called for user ${userId}.`)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/fetch-tasks`,
			{
				method: "POST", // Changed to POST to align with backend
				headers: { "Content-Type": "application/json", ...authHeader }
				// Body is likely empty.
			}
		)

		if (!response.ok) {
			const errorText = await response.text()
			throw new Error(
				`HTTP error fetching tasks! Status: ${response.status}. Details: ${errorText}`
			)
		}
		const result = await response.json()
		console.log(
			`IPC: Fetch Tasks API call successful for user ${userId}. Fetched ${result.tasks?.length || 0} tasks.`
		)
		return result // Expects { tasks: [...] }
	} catch (error) {
		console.error(
			`IPC Error: Exception in fetch-tasks handler for user ${userId}:`,
			error
		)
		return { error: error.message, tasks: [] }
	}
})

// IPC: Add Task (Backend Call)
ipcMain.handle("add-task", async (_event, taskData) => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: add-task - User not authenticated.")
		return { error: "User not authenticated." }
	}

	console.log(
		`IPC: add-task called by user ${userId} with description: "${taskData.description.substring(0, 50)}..."`
	)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/add-task`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader },
				body: JSON.stringify({ description: taskData.description }) // Backend gets user from token
			}
		)

		if (!response.ok) {
			const errorText = await response.text()
			throw new Error(
				`HTTP error adding task! Status: ${response.status}. Details: ${errorText}`
			)
		}
		const result = await response.json()
		console.log(
			`IPC: Add Task API call successful for user ${userId}. New task ID: ${result.task_id}`
		)
		return result // Expects { task_id, message }
	} catch (error) {
		console.error(
			`IPC Error: Exception in add-task handler for user ${userId}:`,
			error
		)
		return { error: error.message }
	}
})

// IPC: Update Task (Backend Call)
ipcMain.handle(
	"update-task",
	async (_event, { taskId, description, priority }) => {
		const userId = getUserIdFromProfile()
		const authHeader = getAuthHeader()
		if (!userId || !authHeader) {
			console.error("IPC: update-task - User not authenticated.")
			return { error: "User not authenticated." }
		}

		console.log(
			`IPC: update-task called by user ${userId} for task ID ${taskId}. New priority: ${priority}`
		)
		try {
			const response = await fetch(
				`${process.env.APP_SERVER_URL || "http://localhost:5000"}/update-task`,
				{
					method: "POST", // Changed to POST in backend example
					headers: {
						"Content-Type": "application/json",
						...authHeader
					},
					body: JSON.stringify({
						task_id: taskId,
						description,
						priority
					})
				}
			)

			if (!response.ok) {
				const errorText = await response.text()
				throw new Error(
					`HTTP error updating task! Status: ${response.status}. Details: ${errorText}`
				)
			}
			const result = await response.json()
			console.log(
				`IPC: Update Task API call successful for user ${userId}, task ${taskId}.`
			)
			return result // Expects { message }
		} catch (error) {
			console.error(
				`IPC Error: Exception in update-task handler for user ${userId}, task ${taskId}:`,
				error
			)
			return { error: error.message }
		}
	}
)

// IPC: Delete Task (Backend Call)
ipcMain.handle("delete-task", async (_event, taskId) => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: delete-task - User not authenticated.")
		return { error: "User not authenticated." }
	}

	console.log(
		`IPC: delete-task called by user ${userId} for task ID: ${taskId}`
	)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/delete-task`,
			{
				method: "POST", // Changed to POST in backend example
				headers: { "Content-Type": "application/json", ...authHeader },
				body: JSON.stringify({ task_id: taskId })
			}
		)

		if (!response.ok) {
			const errorText = await response.text()
			throw new Error(
				`HTTP error deleting task! Status: ${response.status}. Details: ${errorText}`
			)
		}
		const result = await response.json()
		console.log(
			`IPC: Delete Task API call successful for user ${userId}, task ${taskId}.`
		)
		return result // Expects { message }
	} catch (error) {
		console.error(
			`IPC Error: Exception in delete-task handler for user ${userId}, task ${taskId}:`,
			error
		)
		return { error: error.message }
	}
})

// IPC: Fetch Short-Term Memories (Backend Call)
ipcMain.handle(
	"fetch-short-term-memories",
	async (_event, { category, limit = 10 }) => {
		const userId = getUserIdFromProfile()
		const authHeader = getAuthHeader()
		if (!userId || !authHeader) {
			console.error(
				"IPC: fetch-short-term-memories - User not authenticated."
			)
			return [] // Return empty array on auth error for consistency
		}

		console.log(
			`IPC: fetch-short-term-memories called for user ${userId}, category: ${category}, limit: ${limit}.`
		)
		try {
			const response = await fetch(
				`${process.env.APP_SERVER_URL || "http://localhost:5000"}/get-short-term-memories`,
				{
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						...authHeader
					},
					body: JSON.stringify({ category, limit }) // Backend gets user from token
				}
			)

			if (!response.ok) {
				const errorText = await response.text()
				throw new Error(
					`Failed to fetch short-term memories. Status: ${response.status}. Details: ${errorText}`
				)
			}
			const memories = await response.json() // Expects a list of memory objects
			console.log(
				`IPC: Fetch Short-Term Memories API call successful for user ${userId}. Fetched ${memories.length} memories.`
			)
			return memories
		} catch (error) {
			console.error(
				`IPC Error: Exception in fetch-short-term-memories handler for user ${userId}:`,
				error
			)
			return [] // Return empty on error
		}
	}
)

// IPC: Add Short-Term Memory (Backend Call)
ipcMain.handle("add-short-term-memory", async (_event, memoryData) => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: add-short-term-memory - User not authenticated.")
		return { error: "User not authenticated." }
	}

	console.log(
		`IPC: add-short-term-memory called by user ${userId}. Category: ${memoryData.category}`
	)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/add-short-term-memory`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader },
				body: JSON.stringify(memoryData) // Backend gets user from token
			}
		)

		if (!response.ok) {
			const errorText = await response.text()
			throw new Error(
				`HTTP error adding short-term memory! Status: ${response.status}. Details: ${errorText}`
			)
		}
		const result = await response.json()
		console.log(
			`IPC: Add Short-Term Memory API call successful for user ${userId}. Memory ID: ${result.memory_id}`
		)
		return result // Expects { memory_id, message }
	} catch (error) {
		console.error(
			`IPC Error: Exception in add-short-term-memory handler for user ${userId}:`,
			error
		)
		return { error: error.message }
	}
})

// IPC: Update Short-Term Memory (Backend Call)
ipcMain.handle("update-short-term-memory", async (_event, memoryData) => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: update-short-term-memory - User not authenticated.")
		return { error: "User not authenticated." }
	}

	console.log(
		`IPC: update-short-term-memory called by user ${userId} for memory ID: ${memoryData.id}`
	)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/update-short-term-memory`,
			{
				method: "POST", // Changed to POST in backend example
				headers: { "Content-Type": "application/json", ...authHeader },
				body: JSON.stringify(memoryData) // Backend gets user from token
			}
		)

		if (!response.ok) {
			const errorText = await response.text()
			throw new Error(
				`HTTP error updating short-term memory! Status: ${response.status}. Details: ${errorText}`
			)
		}
		const result = await response.json()
		console.log(
			`IPC: Update Short-Term Memory API call successful for user ${userId}, memory ID ${memoryData.id}.`
		)
		return result // Expects { message }
	} catch (error) {
		console.error(
			`IPC Error: Exception in update-short-term-memory handler for user ${userId}:`,
			error
		)
		return { error: error.message }
	}
})

// IPC: Delete Short-Term Memory (Backend Call)
ipcMain.handle("delete-short-term-memory", async (_event, memoryData) => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: delete-short-term-memory - User not authenticated.")
		return { error: "User not authenticated." }
	}

	console.log(
		`IPC: delete-short-term-memory called by user ${userId} for memory ID: ${memoryData.id}, category: ${memoryData.category}`
	)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/delete-short-term-memory`,
			{
				method: "POST", // Changed to POST in backend example
				headers: { "Content-Type": "application/json", ...authHeader },
				body: JSON.stringify(memoryData) // Backend gets user from token
			}
		)

		if (!response.ok) {
			const errorText = await response.text()
			throw new Error(
				`HTTP error deleting short-term memory! Status: ${response.status}. Details: ${errorText}`
			)
		}
		const result = await response.json()
		console.log(
			`IPC: Delete Short-Term Memory API call successful for user ${userId}, memory ID ${memoryData.id}.`
		)
		return result // Expects { message }
	} catch (error) {
		console.error(
			`IPC Error: Exception in delete-short-term-memory handler for user ${userId}:`,
			error
		)
		return { error: error.message }
	}
})

// IPC: Clear All Short-Term Memories (Backend Call)
ipcMain.handle("clear-all-short-term-memories", async () => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error(
			"IPC: clear-all-short-term-memories - User not authenticated."
		)
		return { error: "User not authenticated." }
	}

	console.log(`IPC: clear-all-short-term-memories called for user ${userId}.`)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/clear-all-short-term-memories`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json", ...authHeader }
				// Body is likely empty.
			}
		)

		if (!response.ok) {
			const errorText = await response.text()
			throw new Error(
				`HTTP error clearing all short-term memories! Status: ${response.status}. Details: ${errorText}`
			)
		}
		const result = await response.json()
		console.log(
			`IPC: Clear All Short-Term Memories API call successful for user ${userId}.`
		)
		return result // Expects { message }
	} catch (error) {
		console.error(
			`IPC Error: Exception in clear-all-short-term-memories handler for user ${userId}:`,
			error
		)
		return { error: error.message }
	}
})

// IPC: Fetch Long-Term Memories (Knowledge Graph Data for Visualization) (Backend Call)
ipcMain.handle("fetch-long-term-memories", async () => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: fetch-long-term-memories - User not authenticated.")
		return { error: "User not authenticated.", status: 401 }
	}

	console.log(
		`IPC: fetch-long-term-memories (graph data) called for user ${userId}.`
	)
	try {
		const apiUrl = `${process.env.APP_SERVER_URL}/get-graph-data`
		console.log(
			`IPC: Fetching graph data for user ${userId} from backend: ${apiUrl}`
		)
		const response = await fetch(apiUrl, {
			method: "POST", // Kept POST, backend uses token for user identification
			headers: { "Content-Type": "application/json", ...authHeader }
			// Body is likely empty.
		})

		if (!response.ok) {
			let errorText = await response.text() // Get raw text for better error diagnosis
			try {
				errorText = JSON.parse(errorText).detail || errorText
			} catch {
				/* ignore parse if not json */
			}
			console.error(
				`IPC Error: Fetch Graph Data API call failed for user ${userId}. Status: ${response.status}. Details: ${errorText}`
			)
			return {
				error: `Failed to fetch graph data: ${errorText}`,
				status: response.status
			}
		}
		const graphData = await response.json() // Expects { nodes: [...], edges: [...] }
		console.log(
			`IPC: Fetch Graph Data API call successful for user ${userId}. Nodes: ${graphData.nodes?.length}, Edges: ${graphData.edges?.length}`
		)
		return graphData
	} catch (error) {
		console.error(
			`IPC Error: Exception in fetch-long-term-memories handler for user ${userId}:`,
			error
		)
		return {
			error: `Network or processing error while fetching graph data: ${error.message}`,
			status: 500
		}
	}
})

// IPC: Get Notifications (Backend Call)
ipcMain.handle("get-notifications", async () => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: get-notifications - User not authenticated.")
		return { error: "User not authenticated.", status: 401 }
	}

	console.log(`IPC: get-notifications called for user ${userId}.`)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/get-notifications`,
			{
				method: "POST", // Changed to POST to align with backend
				headers: { "Content-Type": "application/json", ...authHeader }
				// Body is likely empty.
			}
		)

		if (!response.ok) {
			const errorDetail = await response
				.json()
				.catch(() => ({ detail: "Unknown error from server." }))
			console.error(
				`IPC Error: Get Notifications API call failed for user ${userId}. Status: ${response.status}, Details:`,
				errorDetail
			)
			return {
				message: `Error fetching notifications: ${errorDetail.detail || response.statusText}`,
				status: response.status
			}
		}
		const result = await response.json() // Expects { notifications: [...] }
		console.log(
			`IPC: Get Notifications API call successful for user ${userId}. Fetched ${result.notifications?.length || 0} notifications.`
		)
		return { notifications: result.notifications, status: 200 }
	} catch (error) {
		console.error(
			`IPC Error: Exception in get-notifications handler for user ${userId}:`,
			error
		)
		return {
			message: "Error fetching notifications.",
			status: 500,
			error: error.message
		}
	}
})

// IPC: Get Task Approval Data (Backend Call)
ipcMain.handle("get-task-approval-data", async (_event, taskId) => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: get-task-approval-data - User not authenticated.")
		return { status: 401, message: "User not authenticated." }
	}
	if (!taskId) {
		console.error("IPC: get-task-approval-data - Task ID is required.")
		return { status: 400, message: "Task ID is required." }
	}

	console.log(
		`IPC: get-task-approval-data called by user ${userId} for task ID: ${taskId}`
	)
	try {
		const apiUrl = `${process.env.APP_SERVER_URL || "http://localhost:5000"}/get-task-approval-data`
		const response = await fetch(apiUrl, {
			method: "POST",
			headers: { "Content-Type": "application/json", ...authHeader },
			body: JSON.stringify({ task_id: taskId })
		})

		let responseBody
		try {
			responseBody = await response.json()
		} catch {
			responseBody = { detail: await response.text() }
		} // Fallback if not JSON

		if (!response.ok) {
			console.error(
				`IPC Error: Get Task Approval Data API call failed for user ${userId}, task ${taskId}. Status: ${response.status}. Details:`,
				responseBody
			)
			return {
				status: response.status,
				message: responseBody.detail || `HTTP error ${response.status}`
			}
		}
		console.log(
			`IPC: Get Task Approval Data API call successful for user ${userId}, task ${taskId}.`
		)
		return { status: 200, approval_data: responseBody.approval_data } // Expects { approval_data: {...} }
	} catch (error) {
		console.error(
			`IPC Error: Exception fetching approval data for task ${taskId}, user ${userId}:`,
			error
		)
		return {
			status: 500,
			message: `Error fetching approval data: ${error.message}`
		}
	}
})

// IPC: Approve Task (Backend Call)
ipcMain.handle("approve-task", async (_event, taskId) => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: approve-task - User not authenticated.")
		return { status: 401, message: "User not authenticated." }
	}
	if (!taskId) {
		console.error("IPC: approve-task - Task ID is required.")
		return { status: 400, message: "Task ID is required." }
	}

	console.log(
		`IPC: approve-task called by user ${userId} for task ID: ${taskId}`
	)
	try {
		const apiUrl = `${process.env.APP_SERVER_URL || "http://localhost:5000"}/approve-task`
		const response = await fetch(apiUrl, {
			method: "POST",
			headers: { "Content-Type": "application/json", ...authHeader },
			body: JSON.stringify({ task_id: taskId })
		})

		let responseBody
		try {
			responseBody = await response.json()
		} catch {
			responseBody = { detail: await response.text() }
		}

		if (!response.ok) {
			console.error(
				`IPC Error: Approve Task API call failed for user ${userId}, task ${taskId}. Status: ${response.status}. Details:`,
				responseBody
			)
			return {
				status: response.status,
				message:
					responseBody.detail ||
					responseBody.message ||
					`HTTP error ${response.status}`
			}
		}
		console.log(
			`IPC: Approve Task API call successful for user ${userId}, task ${taskId}. Result:`,
			responseBody.result
		)
		return {
			status: 200,
			message: responseBody.message,
			result: responseBody.result
		} // Expects { message, result }
	} catch (error) {
		console.error(
			`IPC Error: Exception approving task ${taskId} for user ${userId}:`,
			error
		)
		return {
			status: 500,
			message: `Error approving task: ${error.message}`
		}
	}
})

// IPC: Get Data Sources Configuration (Backend Call)
ipcMain.handle("get-data-sources", async () => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	// Authentication is good practice here if settings are user-specific.
	if (!userId || !authHeader) {
		console.warn(
			"IPC: get-data-sources - User not authenticated or token missing. Proceeding with potential default or error."
		)
		// Depending on backend, this might fail or return defaults. For now, proceed.
		// return { error: "User not authenticated.", data_sources: [] };
	}

	console.log(`IPC: get-data-sources called by user ${userId || "N/A"}.`)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/get_data_sources`,
			{
				method: "POST", // Changed to POST for consistency and potential future user context
				headers: {
					"Content-Type": "application/json",
					...(authHeader || {})
				} // Include auth header if available
				// Body is likely empty.
			}
		)

		if (!response.ok) {
			const errorText = await response.text()
			throw new Error(
				`Failed to fetch data sources configuration. Status: ${response.status}. Details: ${errorText}`
			)
		}
		const data = await response.json() // Expects { data_sources: [...] }
		console.log(
			`IPC: Get Data Sources API call successful. Found ${data.data_sources?.length || 0} sources.`
		)
		return data
	} catch (error) {
		console.error(
			"IPC Error: Exception in get-data-sources handler:",
			error
		)
		return { error: error.message, data_sources: [] }
	}
})

// IPC: Set Data Source Enabled/Disabled (Backend Call)
ipcMain.handle(
	"set-data-source-enabled",
	async (_event, { source, enabled }) => {
		const userId = getUserIdFromProfile()
		const authHeader = getAuthHeader()
		if (!userId || !authHeader) {
			console.error(
				"IPC: set-data-source-enabled - User not authenticated."
			)
			return { error: "User not authenticated." }
		}

		console.log(
			`IPC: set-data-source-enabled called by user ${userId} for source '${source}' to '${enabled}'.`
		)
		try {
			const response = await fetch(
				`${process.env.APP_SERVER_URL || "http://localhost:5000"}/set_data_source_enabled`,
				{
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						...authHeader
					},
					body: JSON.stringify({ source, enabled }) // Backend gets user from token
				}
			)

			if (!response.ok) {
				const errorText = await response.text()
				throw new Error(
					`Failed to set data source status. Status: ${response.status}. Details: ${errorText}`
				)
			}
			const data = await response.json() // Expects { status, message }
			console.log(
				`IPC: Set Data Source Enabled API call successful for user ${userId}, source '${source}'.`
			)
			return data
		} catch (error) {
			console.error(
				`IPC Error: Exception setting data source status for user ${userId}, source '${source}':`,
				error
			)
			return { error: error.message }
		}
	}
)

// IPC: Clear Chat History (Backend Call)
ipcMain.handle("clear-chat-history", async (_event) => {
	const userId = getUserIdFromProfile()
	const authHeader = getAuthHeader()
	if (!userId || !authHeader) {
		console.error("IPC: clear-chat-history - User not authenticated.")
		return { status: 401, error: "User not authenticated." }
	}

	console.log(`IPC: clear-chat-history request received for user ${userId}.`)
	const targetUrl = `${process.env.APP_SERVER_URL || "http://localhost:5000"}/clear-chat-history`
	try {
		const response = await fetch(targetUrl, {
			method: "POST",
			headers: { "Content-Type": "application/json", ...authHeader }
			// Body is likely empty.
		})

		if (!response.ok) {
			let errorDetail = `Backend responded with status ${response.status}`
			try {
				const d = await response.json()
				errorDetail = d.detail || JSON.stringify(d) // Get specific detail if available
			} catch {
				/* Ignore if response not JSON */
			}
			console.error(
				`IPC Error: Clear Chat History API call failed for user ${userId}. Details: ${errorDetail}`
			)
			return {
				status: response.status,
				error: `Failed to clear history: ${errorDetail}`
			}
		}
		const responseData = await response.json() // Expects { message, activeChatId }
		console.log(
			`IPC: Backend successfully cleared chat history for user ${userId}.`
		)
		return {
			status: response.status,
			message: responseData.message,
			activeChatId: responseData.activeChatId
		}
	} catch (error) {
		console.error(
			`IPC Error: Network or processing error during clear-chat-history for user ${userId}:`,
			error
		)
		return {
			status: 500,
			error: `Failed to communicate with backend: ${error.message}`
		}
	}
})

// --- End of File ---
console.log("Electron main process script execution completed initial setup.")
