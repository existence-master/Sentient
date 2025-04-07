import { app, BrowserWindow, ipcMain, dialog, Notification } from "electron"
import path, { dirname } from "path"
import { fileURLToPath } from "url"
import fetch from "node-fetch"
import {
	getProfile,
	refreshTokens,
	getCheckinFromKeytar,
	getPricingFromKeytar,
	getCreditsFromKeytar,
	setCreditsInKeytar,
	getCreditsCheckinFromKeytar,
	setCreditsCheckinInKeytar,
	fetchAndSetUserRole,
	fetchAndSetReferralCode,
	getReferrerStatusFromKeytar,
	fetchAndSetReferrerStatus,
	fetchAndSetBetaUserStatus,
	getReferralCodeFromKeytar,
	getBetaUserStatusFromKeytar,
	setBetaUserStatusInKeytar
} from "../utils/auth.js"
import { getPrivateData } from "../utils/api.js"
import { createAuthWindow, createLogoutWindow } from "./auth.js"
import dotenv from "dotenv"
import pkg from "electron-updater"
const { autoUpdater } = pkg
import serve from "electron-serve"
import WebSocket from "ws"

/**
 * @file Main process file for the Electron application.
 * Handles application lifecycle, window management, inter-process communication (IPC),
 * auto-updates, and server setup/management.
 */

/**
 * Boolean indicating if the operating system is Windows.
 * @constant {boolean}
 */
const isWindows = process.platform === "win32"
/**
 * Boolean indicating if the operating system is Linux.
 * @constant {boolean}
 */
const isLinux = process.platform === "linux"

/**
 * Base path for application files. Differs based on whether the app is packaged or running in development.
 * @type {string}
 */
let basePath
/**
 * Path to the .env file. Differs based on packaged or development environment.
 * @type {string}
 */
let dotenvPath
/**
 * Path to the chats database file (JSON).
 * @type {string}
 */
// let chatsDbPath // Commented out as not used directly in this snippet
/**
 * Path to the user profile database file (JSON).
 * @type {string}
 */
// let userProfileDbPath // Commented out as not used directly in this snippet
/**
 * Output directory for the packaged Electron app.
 * @type {string}
 */
let appOutDir

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

// Determine paths based on whether the application is packaged or not
if (app.isPackaged) {
	console.log("Application is running in PRODUCTION mode (packaged).")
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
		? path.join(process.resourcesPath, ".env")
		: path.join(app.getPath("home"), ".sentient.env")
	appOutDir = path.join(__dirname, "../out")
} else {
	console.log("Application is running in DEVELOPMENT mode.")
	dotenvPath = path.resolve(__dirname, "../.env")
	appOutDir = path.join(__dirname, "../out")
}

let ws // Keep WebSocket connection attempt regardless of mode for now
try {
	ws = new WebSocket("ws://localhost:5000/ws") // Replace with your FastAPI server URL if different

	ws.onopen = () => {
		console.log("WebSocket connection opened with FastAPI")
	}

	ws.onmessage = (event) => {
		try {
			const messageData = JSON.parse(event.data)
			console.log("WebSocket message received:", messageData)

			let notificationTitle = ""
			let notificationBody = ""

			if (messageData.type === "task_completed") {
				const { task_id, description, result } = messageData
				notificationTitle = "Task Completed!"
				notificationBody = `Task "${description}" (ID: ${task_id}) completed successfully.\nResult: ${result?.substring(0, 100)}...`
			} else if (messageData.type === "task_error") {
				const { task_id, description, error } = messageData
				notificationTitle = "Task Error!"
				notificationBody = `Task "${description}" (ID: ${task_id}) encountered an error.\nError: ${error}`
			} else if (messageData.type === "memory_operation_completed") {
				const { operation_id, status, fact } = messageData
				notificationTitle = "Memory Operation Completed!"
				notificationBody = `Memory operation (ID: ${operation_id}) was successful.\nFact: ${fact?.substring(0, 100)}...`
			} else if (messageData.type === "memory_operation_error") {
				const { operation_id, error, fact } = messageData
				notificationTitle = "Memory Operation Error!"
				notificationBody = `Memory operation (ID: ${operation_id}) encountered an error.\nError: ${error}\nFact: ${fact?.substring(0, 100)}...`
			} else if (messageData.type === "new_message") {
				const { message } = messageData
				console.log("New message received:", message)
				notificationTitle = "New Message!"
				notificationBody = message?.substring(0, 100) + "..."
			}

			if (notificationTitle && notificationBody) {
				new Notification({
					title: notificationTitle,
					body: notificationBody
				}).show()
			}
		} catch (e) {
			console.error(
				"Error processing WebSocket message:",
				e,
				"Raw data:",
				event.data
			)
		}
	}

	ws.onclose = () => {
		console.log("WebSocket connection closed")
		// Optional: Implement reconnection logic here if desired
	}

	ws.onerror = (error) => {
		console.error("WebSocket error:", error?.message || error)
	}
} catch (e) {
	console.error("Failed to establish WebSocket connection:", e)
}

// Load environment variables from .env file
dotenv.config({ path: dotenvPath })

/**
 * The main application window instance.
 * @type {BrowserWindow | null}
 */
let mainWindow

/**
 * Serve instance for packaged app, or null in development.
 * @type {ReturnType<typeof serve> | null}
 */
const appServe = app.isPackaged ? serve({ directory: appOutDir }) : null

/**
 * Asynchronously delays execution for a specified number of milliseconds.
 * @param {number} ms - The number of milliseconds to delay.
 * @returns {Promise<void>} A promise that resolves after the delay.
 */
// const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms)); // delay seems unused, commenting out

/**
 * Creates and displays the main application window.
 * Sets up web preferences, loads the appropriate URL based on environment,
 * and handles window close events.
 * @returns {void}
 */
export const createAppWindow = () => {
	mainWindow = new BrowserWindow({
		width: 2000,
		height: 1500,
		webPreferences: {
			preload: path.join(__dirname, "preload.js"),
			contextIsolation: true,
			enableRemoteModule: false, // Keep false for security best practices
			devTools: !app.isPackaged // Enable DevTools ONLY in development
		},
		backgroundColor: "#000000",
		autoHideMenuBar: true // Hides the default menu bar
	})

	// Load app URL based on environment (packaged or development)
	if (app.isPackaged) {
		appServe(mainWindow)
			.then(() => mainWindow.loadURL("app://-"))
			.catch((err) => console.error("Failed to serve packaged app:", err))
	} else {
		// Development mode
		const devUrl = process.env.ELECTRON_APP_URL
		console.log(`Attempting to load URL: ${devUrl}`)
		mainWindow
			.loadURL(devUrl)
			.catch((err) => console.error(`Failed to load URL ${devUrl}:`, err))
		mainWindow.webContents.openDevTools() // Open DevTools automatically in development
		mainWindow.webContents.on(
			"did-fail-load",
			(event, errorCode, errorDescription, validatedURL) => {
				console.error(
					`Failed to load ${validatedURL}: ${errorDescription} (Code: ${errorCode})`
				)
				// Optional: Add retry logic or show an error page
				// mainWindow.webContents.reloadIgnoringCache() // Be careful with automatic reloads, can cause loops
			}
		)
	}

	// Handle window closing event to confirm exit
	let isExiting = false
	mainWindow.on("close", async (event) => {
		if (!isExiting) {
			event.preventDefault() // Prevent default close action
			isExiting = true

			try {
				// Show confirmation dialog before quitting
				const response = await dialog.showMessageBox(mainWindow, {
					type: "question",
					buttons: ["Yes", "No"],
					defaultId: 1,
					title: "Confirm Exit",
					message: "Are you sure you want to quit?"
				})

				// If user confirms exit, attempt to stop servers (if any) and quit the app
				if (response.response === 0) {
					console.log("User confirmed exit.")
					// Add any cleanup tasks here if needed before quitting
					// e.g., await stopNeo4jServer(); await stopAppSubServers();
					await dialog.showMessageBox(mainWindow, {
						// Keep this user feedback
						type: "info",
						message: "Bye friend...",
						title: "Quitting Sentient (click ok to exit)"
					})
					setTimeout(() => app.quit(), 500) // Quit app after a short delay
				} else {
					console.log("User cancelled exit.")
					isExiting = false // Reset exiting flag if user cancels exit
				}
			} catch (error) {
				console.error(
					"Error during exit confirmation or cleanup:",
					error
				)
				// Show error to user, but allow exit attempt to proceed or reset
				dialog.showErrorBox(
					"Exit Error",
					`An error occurred during shutdown: ${error.message}. The application might not close cleanly.`
				)
				// Decide if you want to force quit or reset the flag
				isExiting = false // Resetting might be safer to allow another attempt
				// Alternatively, force quit: setTimeout(() => app.quit(), 500);
			}
		}
	})

	mainWindow.on("closed", () => {
		mainWindow = null // Dereference the window object
	})
}

// --- Conditional Check Functions ---

/**
 * Checks user information validity (ONLY IN PRODUCTION).
 * Includes pricing, referral, beta status, credits, Google creds.
 * Triggers logout on failure.
 * @async
 * @returns {Promise<void>}
 */
const checkUserInfo = async () => {
	if (!app.isPackaged) {
		console.log(
			"DEV MODE: Skipping User Info checks (Pricing, Referral, Beta, Credits, Google)."
		)
		return // Skip entirely in development
	}
	console.log("PROD MODE: Performing User Info checks...")
	try {
		await checkPricingStatus()
		await checkReferralCodeAndStatus()
		await checkBetaUserStatus()
		await resetCreditsIfNecessary()
		await checkGoogleCredentials()
		console.log("PROD MODE: User info checks passed.")
	} catch (error) {
		console.error(
			`PROD MODE: Error during user info check: ${error.message}. Triggering logout.`
		)
		BrowserWindow.getAllWindows().forEach((win) => win.close())
		createLogoutWindow() // Create logout window on check failure
		// Re-throw or handle as needed, maybe notify user more clearly
		throw error // Propagate error to indicate failure
	}
}

/**
 * Checks overall application validity (ONLY IN PRODUCTION).
 * Verifies authentication and user info. Redirects to auth on failure.
 * @async
 * @returns {Promise<void>}
 */
export const checkValidity = async () => {
	if (!app.isPackaged) {
		console.log("DEV MODE: Skipping Validity checks (Auth, User Info).")
		return // Skip entirely in development
	}
	console.log("PROD MODE: Performing Validity checks...")
	try {
		await checkAuthStatus() // Checks tokens
		await checkUserInfo() // Checks pricing, credits etc.
		console.log("PROD MODE: Validity checks passed.")
	} catch (error) {
		console.error(
			`PROD MODE: Validity check failed: ${error.message}. Redirecting to Auth window.`
		)
		// Ensure windows are closed before showing auth window
		BrowserWindow.getAllWindows().forEach((win) => {
			// Avoid closing the auth window if it's already being created
			if (!win.isAuthWindow) {
				// You might need to add a property like this during creation
				win.close()
			}
		})
		createAuthWindow() // Create auth window on validity check failure
		throw error // Propagate error
	}
}

/**
 * Checks user authentication status by refreshing tokens (ONLY IN PRODUCTION).
 * Throws an error if token refresh fails.
 * @async
 * @returns {Promise<void>}
 */
const checkAuthStatus = async () => {
	// This check is implicitly skipped in DEV because checkValidity skips it.
	// No need for an explicit app.isPackaged check here if only called by checkValidity.
	console.log("PROD MODE: Checking authentication status...")
	try {
		await refreshTokens()
		console.log("PROD MODE: User is authenticated.")
	} catch (error) {
		console.error(
			`PROD MODE: Authentication check failed (token refresh): ${error.message}`
		)
		throw new Error("User is not authenticated.") // Let checkValidity handle redirection
	}
}

// --- Status Check Functions (will only run in production via checkUserInfo) ---

const checkReferralCodeAndStatus = async () => {
	console.log("PROD MODE: Checking Referral Code and Status...")
	try {
		let referralCode = await getReferralCodeFromKeytar()
		if (!referralCode) {
			console.log(
				"PROD MODE: Referral code not found in Keytar, fetching..."
			)
			referralCode = await fetchAndSetReferralCode()
			console.log("PROD MODE: Referral code fetched and set.")
		}

		let referrerStatus = await getReferrerStatusFromKeytar()
		// Check explicitly for null, as false is a valid status
		if (referrerStatus === null) {
			console.log(
				"PROD MODE: Referrer status not found in Keytar, fetching..."
			)
			referrerStatus = await fetchAndSetReferrerStatus()
			console.log("PROD MODE: Referrer status fetched and set.")
		}

		console.log("PROD MODE: Referral code and status check complete.")
	} catch (error) {
		console.error(
			`PROD MODE: Error in referral code/status check: ${error.message}`
		)
		throw new Error("Referral code and status check failed.")
	}
}

const checkBetaUserStatus = async () => {
	console.log("PROD MODE: Checking Beta User Status...")
	try {
		let betaUserStatus = await getBetaUserStatusFromKeytar()
		// Check explicitly for null, as false is a valid status
		if (betaUserStatus === null) {
			console.log(
				"PROD MODE: Beta user status not found in Keytar, fetching..."
			)
			betaUserStatus = await fetchAndSetBetaUserStatus()
			console.log("PROD MODE: Beta user status fetched and set.")
		}

		console.log("PROD MODE: Beta user status check complete.")
	} catch (error) {
		console.error(
			`PROD MODE: Error in beta user status check: ${error.message}`
		)
		throw new Error("Beta user status check failed.")
	}
}

const checkPricingStatus = async () => {
	console.log("PROD MODE: Checking Pricing Status and Session Activity...")
	try {
		const pricing = await getPricingFromKeytar()
		const lastCheckin = await getCheckinFromKeytar() // Timestamp of last successful refresh/activity

		if (!pricing || !lastCheckin) {
			console.warn(
				"PROD MODE: Pricing or last checkin not found in Keytar. Fetching user role/pricing..."
			)
			await fetchAndSetUserRole() // This should also set the check-in time implicitly via refreshTokens
			console.log("PROD MODE: User role/pricing fetched and set.")
			// Re-fetch checkin time after role fetch? Depends on fetchAndSetUserRole implementation. Assuming it's handled.
			return // Exit check for this run, will be checked next time
		}

		const currentTimestamp = Math.floor(Date.now() / 1000)
		const sevenDaysInSeconds = 7 * 24 * 60 * 60

		// Check if last check-in was more than 7 days ago
		if (currentTimestamp - lastCheckin >= sevenDaysInSeconds) {
			console.warn(
				`PROD MODE: User session expired due to inactivity (last check-in ${new Date(lastCheckin * 1000).toISOString()}). Triggering logout.`
			)
			// Don't close windows here, let the caller (checkUserInfo) handle it.
			throw new Error("User session expired due to inactivity.")
		}

		console.log(
			"PROD MODE: Pricing status and session activity check complete."
		)
	} catch (error) {
		// Re-throw specific errors or a generic one
		console.error(
			`PROD MODE: Error in pricing/session check: ${error.message}`
		)
		if (error.message.includes("inactivity")) {
			throw error // Propagate inactivity error
		}
		throw new Error("Pricing status check failed.") // Generic failure
	}
}

const resetCreditsIfNecessary = async () => {
	console.log("PROD MODE: Checking if Pro Credits need reset...")
	try {
		const currentDate = new Date().toDateString()
		const checkinTimestamp = await getCreditsCheckinFromKeytar()
		const checkinDate = checkinTimestamp
			? new Date(checkinTimestamp * 1000).toDateString()
			: null

		if (checkinDate !== currentDate) {
			console.log(
				`PROD MODE: Resetting proCredits (Last checkin: ${checkinDate}, Today: ${currentDate}).`
			)
			const referrer = await getReferrerStatusFromKeytar() // Ensure await here

			// Default to non-referrer if status is null/undefined
			const creditsToSet = referrer === true ? 10 : 5
			await setCreditsInKeytar(creditsToSet)
			console.log(`PROD MODE: proCredits reset to ${creditsToSet}.`)

			await setCreditsCheckinInKeytar() // Update credits check-in timestamp
			console.log("PROD MODE: proCredits checkin timestamp updated.")
		} else {
			console.log("PROD MODE: Pro Credits already checked/reset today.")
		}
	} catch (error) {
		console.error(`PROD MODE: Error resetting proCredits: ${error.message}`)
		throw new Error("Error checking/resetting proCredits")
	}
}

const checkGoogleCredentials = async () => {
	console.log("PROD MODE: Checking Google Credentials via server...")
	try {
		// Ensure the server URL is correctly defined in your .env
		const googleAuthCheckUrl = `${process.env.APP_SERVER_URL || "http://127.0.0.1:5000"}/authenticate-google`
		const response = await fetch(googleAuthCheckUrl) // Assumes GET request is sufficient

		if (!response.ok) {
			// Try to get more details from the response body
			let errorDetails = `Server responded with status ${response.status}`
			try {
				const data = await response.json()
				errorDetails = data.error || data.message || errorDetails
			} catch (e) {
				/* Ignore if response is not JSON */
			}
			throw new Error(`Google credentials check failed: ${errorDetails}`)
		}

		const data = await response.json()

		if (data.success) {
			console.log("PROD MODE: Google credentials check successful.")
		} else {
			throw new Error(
				data.error || "Google credentials check indicated failure."
			)
		}
	} catch (error) {
		console.error(
			`PROD MODE: Error checking Google credentials: ${error.message}`
		)
		throw new Error("Error checking Google credentials") // Propagate a standard error message
	}
}

// --- Single Instance Lock ---
const gotTheLock = app.requestSingleInstanceLock()

if (!gotTheLock) {
	console.log("Another instance is already running. Quitting this instance.")
	app.quit()
} else {
	app.on("second-instance", () => {
		console.log("Second instance detected. Focusing main window.")
		// Focus on the main window if a second instance is attempted
		if (mainWindow) {
			if (mainWindow.isMinimized()) mainWindow.restore()
			mainWindow.focus()
		}
	})
}

// --- Auto Update Logic (Conditional) ---

/**
 * Checks for application updates using Electron AutoUpdater (ONLY IN PRODUCTION).
 * Sets feed URL based on beta status (if implemented) and initiates checks.
 * Creates the main window AFTER deciding update path.
 * @async
 * @returns {Promise<void>}
 */
const checkForUpdates = async () => {
	// This function is only called if app.isPackaged is true.
	console.log(
		"PROD MODE: Configuring and checking for application updates..."
	)

	// Fetch beta status from keytar to decide update channel (only needed in prod)
	let isBetaUser = false
	try {
		isBetaUser = (await getBetaUserStatusFromKeytar()) || false // Default to false if not set
		console.log(
			`PROD MODE: User is ${isBetaUser ? "a beta user" : "not a beta user"}.`
		)
	} catch (error) {
		console.error(
			"PROD MODE: Failed to get beta user status for update check, defaulting to stable.",
			error
		)
		isBetaUser = false
	}

	const feedOptions = {
		provider: "github",
		owner: "existence-master",
		repo: "Sentient"
	}

	autoUpdater.setFeedURL(feedOptions)

	if (isBetaUser) {
		console.log("PROD MODE: Allowing pre-releases for beta user.")
		autoUpdater.allowPrerelease = true
	} else {
		console.log("PROD MODE: Checking for stable releases only.")
		autoUpdater.allowPrerelease = false // Explicitly false for stable
	}

	// Create the window *before* checking for updates, so user sees something
	createAppWindow()

	try {
		console.log("PROD MODE: Initiating update check...")
		// Check for updates and notify the user automatically
		await autoUpdater.checkForUpdatesAndNotify()
		console.log(
			"PROD MODE: Update check initiated (notifications handled by autoUpdater)."
		)
	} catch (error) {
		console.error("PROD MODE: Error during update check:", error.message)
		// Decide how to handle check errors, maybe notify user manually or just log
		// dialog.showErrorBox("Update Check Failed", `Could not check for updates: ${error.message}`);
	}
}

/**
 * Starts the application.
 * In Production: Checks for updates, then performs validity checks after a delay.
 * In Development: Skips updates and validity checks, creates window immediately.
 * @async
 * @returns {Promise<void>}
 */
const startApp = async () => {
	if (app.isPackaged) {
		// === Production/Packaged App Logic ===
		console.log("PRODUCTION MODE: Starting application flow.")
		// checkForUpdates now creates the window internally
		await checkForUpdates()

		// Delay validity checks to allow window/update process to potentially start/notify
		const validityCheckDelay = 5000 // 5 seconds
		console.log(
			`PRODUCTION MODE: Scheduling validity checks in ${validityCheckDelay}ms.`
		)
		setTimeout(async () => {
			// Check if an update process was flagged (set by 'update-available' listener)
			if (process.env.UPDATING !== "true") {
				console.log(
					"PRODUCTION MODE: Performing scheduled validity checks..."
				)
				try {
					await checkValidity() // Perform auth and status checks
					console.log(
						"PRODUCTION MODE: Validity checks completed successfully."
					)
				} catch (error) {
					// Error handling (like showing auth window) is done within checkValidity
					console.error(
						"PRODUCTION MODE: Validity check sequence failed.",
						error.message
					)
				}
			} else {
				console.log(
					"PRODUCTION MODE: Skipping scheduled validity check because an update is in progress (UPDATING=true)."
				)
			}
		}, validityCheckDelay)
	} else {
		// === Development Environment Logic ===
		console.log("DEVELOPMENT MODE: Starting application flow.")
		console.log(
			"DEVELOPMENT MODE: Skipping auto-updates and validity checks."
		)
		// Create the window directly without checking for updates or validity
		createAppWindow()
		mainWindow?.webContents.on("did-finish-load", () => {
			console.log("DEVELOPMENT MODE: Main window finished loading.")
			// Optional: Send message to renderer if it needs to know it's in dev
			// mainWindow?.webContents.send('development-mode', true);
		})
	}
}

// --- App Lifecycle Events ---

// Event listener when the app is ready to start
app.on("ready", startApp)

// Event listener for 'window-all-closed' event
app.on("window-all-closed", () => {
	console.log("All windows closed.")
	if (process.platform !== "darwin") {
		console.log("Quitting application (not macOS).")
		app.quit() // Quit app when all windows are closed, except on macOS
	}
})

app.on("activate", () => {
	// On macOS it's common to re-create a window in the app when the
	// dock icon is clicked and there are no other windows open.
	if (BrowserWindow.getAllWindows().length === 0) {
		console.log("App activated with no windows open, creating main window.")
		startApp() // Or just createAppWindow() if startApp logic is complex
	}
})

// --- AutoUpdater Event Listeners (Only relevant in Production) ---

if (app.isPackaged) {
	autoUpdater.on("update-available", (info) => {
		console.log("PROD MODE: Update available:", info)
		process.env.UPDATING = "true" // Set flag indicating update is in progress
		mainWindow?.webContents.send("update-available", info) // Send event to renderer process
	})

	autoUpdater.on("update-not-available", (info) => {
		console.log("PROD MODE: Update not available:", info)
		// Optional: Notify renderer or log
	})

	autoUpdater.on("error", (err) => {
		console.error("PROD MODE: Error in auto-updater:", err.message)
		process.env.UPDATING = "false" // Reset flag on error
		mainWindow?.webContents.send("update-error", err.message) // Notify renderer
		// Reset progress on error
		mainWindow?.webContents.send("update-progress", {
			percent: 0,
			error: true
		})
		// Optional: Show error dialog
		// dialog.showErrorBox("Update Error", `Failed to update: ${err.message}`);
	})

	autoUpdater.on("download-progress", (progressObj) => {
		console.log(`PROD MODE: Download progress: ${progressObj.percent}%`)
		mainWindow?.webContents.send("update-progress", progressObj) // Send progress to renderer
	})

	autoUpdater.on("update-downloaded", (info) => {
		console.log("PROD MODE: Update downloaded:", info)
		process.env.UPDATING = "false" // Reset flag, ready to install
		mainWindow?.webContents.send("update-downloaded", info) // Send event to renderer process
		// Optional: Prompt user immediately
		/*
        dialog.showMessageBox(mainWindow, {
            type: 'info',
            title: 'Update Ready',
            message: 'A new version has been downloaded. Restart the application to apply the update.',
            buttons: ['Restart', 'Later']
        }).then(result => {
            if (result.response === 0) { // Restart button
                autoUpdater.quitAndInstall();
            }
        });
        */
	})
}

// --- IPC Handlers ---

// IPC event handler for 'restart-app' to quit and install update (only relevant in prod)
ipcMain.on("restart-app", async () => {
	if (app.isPackaged) {
		console.log(
			"PROD MODE: Received restart-app command. Quitting and installing update."
		)
		autoUpdater.quitAndInstall()
	} else {
		console.log(
			"DEV MODE: Received restart-app command, but auto-updates are disabled. Doing nothing."
		)
	}
})

// IPC handler to get user profile (available in both modes)
ipcMain.handle("get-profile", async () => {
	// In dev mode, this might return empty/default data if auth is skipped
	// Consider returning mock data in dev if needed
	if (!app.isPackaged) {
		console.log(
			"DEV MODE: Handling get-profile. Returning potentially empty profile or mock data."
		)
		return { given_name: "John Doe", picture: null }
	}
	try {
		return await getProfile()
	} catch (error) {
		console.error("Error getting profile:", error)
		return null // Indicate failure
	}
})

// IPC handler to get private data (available in both modes)
ipcMain.handle("get-private-data", async () => {
	if (!app.isPackaged) {
		console.log("DEV MODE: Handling get-private-data.")
		// Consider returning mock data if needed
	}
	try {
		return await getPrivateData()
	} catch (error) {
		console.error("Error getting private data:", error)
		return null
	}
})

ipcMain.handle("get-beta-user-status", async () => {
	try {
		if (!app.isPackaged) {
			console.log(
				"DEV MODE: Handling fetch-pricing-plan. Returning 'pro' (or 'free' if preferred)."
			)
			return "true" // Return a default/mock value for development
		}
		const betaUserStatus = await getBetaUserStatusFromKeytar() // Get beta user status from Keytar
		return betaUserStatus
	} catch (error) {
		await console.log(`Error fetching beta user status: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

// IPC event handler for 'log-out'
ipcMain.on("log-out", () => {
	console.log("Log-out command received.")
	// Close all windows except potentially a dedicated logout window if needed
	BrowserWindow.getAllWindows().forEach((win) => {
		// Add logic here if you want to keep one window open (like the logout window itself)
		win.close()
	})
	if (app.isPackaged) {
		console.log("PROD MODE: Creating logout window after logout command.")
		createLogoutWindow() // Show logout confirmation/status only in production
	} else {
		console.log(
			"DEV MODE: Closing windows on logout command, not showing logout window."
		)
		// In dev, maybe just quit or reload the main window to simulate a fresh start
		app.quit() // Or mainWindow?.reload();
	}
})

// IPC handler to fetch pricing plan
ipcMain.handle("fetch-pricing-plan", async () => {
	if (!app.isPackaged) {
		console.log(
			"DEV MODE: Handling fetch-pricing-plan. Returning 'pro' (or 'free' if preferred)."
		)
		return "pro" // Return a default/mock value for development
	}
	try {
		const pricing = await getPricingFromKeytar()
		console.log("PROD MODE: Fetched pricing plan:", pricing)
		return pricing || "free" // Default to 'free' if not found
	} catch (error) {
		console.error(`PROD MODE: Error fetching pricing plan: ${error}`)
		return "free" // Default to 'free' on error
	}
})

// IPC handler to fetch pro credits
ipcMain.handle("fetch-pro-credits", async () => {
	if (!app.isPackaged) {
		console.log(
			"DEV MODE: Handling fetch-pro-credits. Returning mock value 999."
		)
		return 999 // Return a high/mock value for development
	}
	try {
		const credits = await getCreditsFromKeytar()
		console.log("PROD MODE: Fetched pro credits:", credits)
		return credits
	} catch (error) {
		console.error(`PROD MODE: Error fetching pro credits: ${error}`)
		return 0 // Default to 0 on error
	}
})

// IPC handler to decrement pro credits (careful in dev)
ipcMain.handle("decrement-pro-credits", async () => {
	if (!app.isPackaged) {
		console.log(
			"DEV MODE: Handling decrement-pro-credits. Simulating success without changing stored value."
		)
		// Don't actually decrement anything in dev mode to avoid persistence issues
		return true // Simulate success
	}
	try {
		let credits = await getCreditsFromKeytar()
		credits = Math.max(0, credits - 1) // Ensure credits don't go below 0
		await setCreditsInKeytar(credits)
		console.log(`PROD MODE: Decremented pro credits to ${credits}.`)
		return true
	} catch (error) {
		console.error(`PROD MODE: Error decrementing pro credits: ${error}`)
		return false
	}
})

// IPC handler to fetch referral code
ipcMain.handle("fetch-referral-code", async () => {
	if (!app.isPackaged) {
		console.log(
			"DEV MODE: Handling fetch-referral-code. Returning mock code 'DEVREFERRAL'."
		)
		return "DEVREFERRAL"
	}
	try {
		const referralCode = await getReferralCodeFromKeytar()
		console.log("PROD MODE: Fetched referral code:", referralCode)
		return referralCode
	} catch (error) {
		console.error(`PROD MODE: Error fetching referral code: ${error}`)
		return null
	}
})

// IPC handler to fetch referrer status
ipcMain.handle("fetch-referrer-status", async () => {
	if (!app.isPackaged) {
		console.log(
			"DEV MODE: Handling fetch-referrer-status. Returning mock status 'true'."
		)
		return true // Or false, depending on testing needs
	}
	try {
		const referrerStatus = await getReferrerStatusFromKeytar()
		console.log("PROD MODE: Fetched referrer status:", referrerStatus)
		return referrerStatus
	} catch (error) {
		console.error(`PROD MODE: Error fetching referrer status: ${error}`)
		return null
	}
})

// IPC handler to fetch beta user status
ipcMain.handle("fetch-beta-user-status", async () => {
	// This might be useful in DEV too, to toggle UI elements
	// Let's allow it but log clearly.
	let isDev = !app.isPackaged
	if (isDev) console.log("DEV MODE: Handling fetch-beta-user-status.")

	try {
		const betaUserStatus = await getBetaUserStatusFromKeytar()
		console.log(
			`${isDev ? "DEV" : "PROD"} MODE: Fetched beta user status:`,
			betaUserStatus
		)
		// In pure dev mode, maybe default to true/false if needed?
		// if (isDev && betaUserStatus === null) return true; // Example default for dev
		return betaUserStatus
	} catch (error) {
		console.error(
			`${isDev ? "DEV" : "PROD"} MODE: Error fetching beta user status: ${error}`
		)
		return null // Return null on error
	}
})

// IPC handler to set referrer using referral code
ipcMain.handle("set-referrer", async (_event, { referralCode }) => {
	const isDev = !app.isPackaged
	if (isDev) {
		console.log(
			`DEV MODE: Simulating set-referrer with code: ${referralCode}. Returning success.`
		)
		// Optionally, store mock status locally if needed for dev UI
		return {
			message: "DEV: Referrer status simulated successfully",
			status: 200
		}
	}
	try {
		const apiUrl = `${process.env.APP_SERVER_URL}/get-user-and-set-referrer-status`
		console.log(
			`PROD MODE: Sending referral code ${referralCode} to ${apiUrl}`
		)
		const response = await fetch(apiUrl, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ referral_code: referralCode }) // Assumes API expects this body
		})

		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`PROD MODE: Error from set-referrer API: ${response.status} ${response.statusText}`,
				errorText
			)
			throw new Error(`Error from API: ${response.statusText}`)
		}

		const result = await response.json()
		console.log("PROD MODE: Set referrer API call successful:", result)
		// We should probably update the status in Keytar here as well
		await fetchAndSetReferrerStatus() // Re-fetch and save the updated status
		return {
			message: result.message || "Referrer updated successfully",
			status: 200
		}
	} catch (error) {
		console.error(`PROD MODE: Error in set-referrer IPC handler: ${error}`)
		return { error: error.message, status: 500 } // Return error object
	}
})

// IPC handler to invert beta user status
ipcMain.handle("invert-beta-user-status", async () => {
	const isDev = !app.isPackaged
	if (isDev) {
		console.log("DEV MODE: Simulating invert-beta-user-status.")
		// Optionally toggle a mock status if needed
		return { message: "DEV: Beta status inversion simulated", status: 200 }
	}
	try {
		// URL might be different, adjust as needed. Assumed UTILS_SERVER_URL before.
		const apiUrl = `${process.env.APP_SERVER_URL}/get-user-and-invert-beta-user-status`
		const profile = await getProfile() // Needed for user_id

		if (!profile || !profile.sub) {
			throw new Error("User profile or ID not available.")
		}

		console.log(
			`PROD MODE: Inverting beta status for user: ${profile.sub} via ${apiUrl}`
		)
		const response = await fetch(apiUrl, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ user_id: profile.sub })
		})

		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`PROD MODE: Error from invert-beta API: ${response.status} ${response.statusText}`,
				errorText
			)
			throw new Error(`Error from API: ${response.statusText}`)
		}

		const result = await response.json()
		console.log(
			"PROD MODE: Invert beta status API call successful:",
			result
		)

		// Update local status in Keytar immediately for UI consistency
		const currentStatus = await getBetaUserStatusFromKeytar()
		const newStatus = !currentStatus
		await setBetaUserStatusInKeytar(newStatus)
		console.log(
			`PROD MODE: Updated local beta status in Keytar to: ${newStatus}`
		)

		return {
			message: result.message || "Beta status inverted successfully",
			status: 200
		}
	} catch (error) {
		console.error(
			`PROD MODE: Error in invert-beta-user-status IPC handler: ${error}`
		)
		return { error: error.message, status: 500 }
	}
})

// --- Database Handlers (Use FastAPI backend) ---

// Helper function to fetch user_id (needed for memory operations)
// Uses the same backend endpoint as get-user-data initially
async function getUserIdForMemoryOps() {
	try {
		// In production, fetch the real user ID
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/get-user-data`,
			{
				method: "POST", // Assuming this returns the user data structure
				headers: { "Content-Type": "application/json" }
			}
		)
		if (!response.ok) {
			const errorDetail = await response
				.json()
				.catch(() => ({ detail: "Failed to parse error response" }))
			console.error(
				`Error fetching user data for ID: ${response.status}`,
				errorDetail
			)
			throw new Error("Failed to fetch user data")
		}
		const result = await response.json()
		// Adjust path based on your actual DB structure from get-user-data
		const userId = result?.data?.personalInfo?.name || result?.data?.userId
		if (!userId) {
			console.error("User ID not found in fetched DB data:", result)
			throw new Error("User ID could not be determined from DB data")
		}
		return userId
	} catch (error) {
		console.error("Error fetching user ID for memory operations:", error)
		throw error // Propagate error
	}
}

// IPC handler to set data in user profile database (replaces existing keys)
ipcMain.handle("set-user-data", async (_event, args) => {
	const { data } = args
	console.log("IPC: set-user-data called with:", data)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/set-user-data`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ data: data })
			}
		)
		if (!response.ok) {
			const errorDetail = await response
				.json()
				.catch(() => ({ detail: "Failed to parse error response" }))
			console.error(
				`Error from set-user-data API: ${response.status}`,
				errorDetail
			)
			return {
				message: `Error storing data: ${errorDetail.detail || response.statusText}`,
				status: response.status
			}
		}
		const result = await response.json()
		console.log("IPC: set-user-data successful:", result)
		return result
	} catch (error) {
		console.error(`IPC Error: set-user-data failed: ${error}`)
		return {
			message: "Error storing data",
			status: 500,
			error: error.message
		}
	}
})

// IPC handler to add/merge data into user profile database
ipcMain.handle("add-db-data", async (_event, args) => {
	const { data } = args
	console.log("IPC: add-db-data called with:", data)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/add-db-data`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ data: data })
			}
		)
		if (!response.ok) {
			const errorDetail = await response
				.json()
				.catch(() => ({ detail: "Failed to parse error response" }))
			console.error(
				`Error from add-db-data API: ${response.status}`,
				errorDetail
			)
			return {
				message: `Error adding data: ${errorDetail.detail || response.statusText}`,
				status: response.status
			}
		}
		const result = await response.json()
		console.log("IPC: add-db-data successful:", result)
		return result
	} catch (error) {
		console.error(`IPC Error: add-db-data failed: ${error}`)
		return {
			message: "Error adding data",
			status: 500,
			error: error.message
		}
	}
})

// IPC handler to get all user profile database data
ipcMain.handle("get-user-data", async () => {
	console.log("IPC: get-user-data called")
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/get-user-data`,
			{
				method: "POST", // Still POST as per original code
				headers: { "Content-Type": "application/json" }
			}
		)
		if (!response.ok) {
			const errorDetail = await response
				.json()
				.catch(() => ({ detail: "Failed to parse error response" }))
			console.error(
				`Error from get-user-data API: ${response.status}`,
				errorDetail
			)
			return {
				message: `Error fetching data: ${errorDetail.detail || response.statusText}`,
				status: response.status
			}
		}
		const result = await response.json()
		console.log("IPC: get-user-data successful.") // Don't log the actual data unless debugging
		return { data: result.data, status: result.status }
	} catch (error) {
		console.error(`IPC Error: get-user-data failed: ${error}`)
		return {
			message: "Error fetching data",
			status: 500,
			error: error.message
		}
	}
})

// --- Chat Handlers ---

ipcMain.handle("fetch-chat-history", async () => {
	console.log("IPC: fetch-chat-history called")
	try {
		// Assuming chat history doesn't strictly require prod auth checks
		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/get-history`,
			{
				method: "GET" // Assuming GET is appropriate
			}
		)
		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`Error fetching chat history: ${response.status} ${response.statusText}`,
				errorText
			)
			throw new Error(
				`Failed to fetch chat history: Status ${response.status}`
			)
		}
		const data = await response.json()
		console.log("IPC: fetch-chat-history successful.")
		return { messages: data.messages, status: 200 }
	} catch (error) {
		console.error(`IPC Error: fetch-chat-history failed: ${error}`)
		return {
			message: "Error fetching chat history",
			status: 500,
			error: error.message
		}
	}
})

// Keep the scraper handler from the current code if needed
ipcMain.handle("scrape-linkedin", async (_event, { linkedInProfileUrl }) => {
	// ... (keep the scrape-linkedin handler as it was in the "Current code")
	console.log("IPC: scrape-linkedin called for URL:", linkedInProfileUrl)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/scrape-linkedin`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ url: linkedInProfileUrl })
			}
		)
		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`Error from scrape-linkedin API: ${response.status} ${response.statusText}`,
				errorText
			)
			return {
				message: "Error scraping LinkedIn profile",
				status: response.status,
				error: errorText
			}
		}
		const { profile } = await response.json()
		console.log("IPC: scrape-linkedin successful.")
		return {
			message: "LinkedIn profile scraped successfully",
			profile,
			status: 200
		}
	} catch (error) {
		console.error(`IPC Error: scrape-linkedin failed: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

// Modified send-message handler
ipcMain.handle("send-message", async (_event, { input }) => {
	console.log(
		"IPC: send-message called with input:",
		input.substring(0, 50) + "..."
	) // Log truncated input
	let pricing = "pro" // Default to 'pro' in dev
	let credits = 999 // Default high credits in dev

	try {
		// --- Start: Logic from "Current code" ---
		if (app.isPackaged) {
			// Fetch real values only in production
			pricing = (await getPricingFromKeytar()) || "free"
			credits = await getCreditsFromKeytar()
			console.log(
				`PROD MODE: Sending message with pricing=${pricing}, credits=${credits}`
			)
		} else {
			console.log(
				`DEV MODE: Sending message with mock pricing=${pricing}, credits=${credits}`
			)
		}
		// --- End: Logic from "Current code" ---

		const payload = {
			input,
			pricing,
			credits
		}
		// Using logging from "Current code"
		console.log("Sending payload to /chat:", JSON.stringify(payload))

		const response = await fetch(`${process.env.APP_SERVER_URL}/chat`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(payload)
		})

		// Using error handling from "Current code"
		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`Error from /chat endpoint: ${response.status} ${response.statusText}`,
				errorText
			)
			// Send error back to renderer
			mainWindow?.webContents.send("chat-error", {
				message: `Server error: ${response.status} ${response.statusText || ""}. ${errorText}`
			})
			throw new Error(`Error sending message: Status ${response.status}`)
		}

		// Using null body check from "Current code"
		if (!response.body) {
			throw new Error("Response body is null, cannot process stream.")
		}

		// --- Start: Stream Handling from "Original node" ---
		const readable = response.body
		const decoder = new TextDecoder("utf-8")
		let buffer = ""

		console.log("IPC: Starting to process chat stream (original method)...") // Added logging

		// Use optional chaining for mainWindow access
		if (!mainWindow) {
			console.error("mainWindow is not available to send messages.")
			throw new Error("Application window not found.")
		}

		for await (const chunk of readable) {
			buffer += decoder.decode(chunk, { stream: true })
			// Use '\n\n' if your server sends messages separated by double newlines
			// Use '\n' if separated by single newlines (as in original)
			const splitMessages = buffer.split("\n") // Assuming single newline separation
			buffer = splitMessages.pop() || "" // Keep incomplete line, handle empty pop result

			for (const msg of splitMessages) {
				if (msg.trim() === "") continue // Skip empty lines
				try {
					const parsedMessage = JSON.parse(msg)
					// console.log("Parsed stream chunk:", parsedMessage); // Optional verbose logging

					if (parsedMessage.type === "assistantStream") {
						mainWindow.webContents.send("message-stream", {
							// Use a consistent or generated ID if needed
							messageId:
								parsedMessage.messageId ||
								`stream-${Date.now()}-${Math.random()}`,
							token: parsedMessage.token || "" // Ensure token is always a string
						})

						// Decrement credits *only in production* if pro was used on a free plan
						// This check is from the original logic, applied conditionally based on app.isPackaged
						if (
							app.isPackaged && // Only decrement in production
							parsedMessage.done && // Check if the message indicates completion *and* pro usage
							parsedMessage.proUsed &&
							pricing === "free"
						) {
							console.log(
								"PROD MODE: Pro features used on free plan. Decrementing credit."
							)
							let currentCredits = await getCreditsFromKeytar()
							currentCredits -= 1
							await setCreditsInKeytar(
								Math.max(currentCredits, 0)
							)
							// Optionally send updated credits back to renderer
							// mainWindow.webContents.send("credits-updated", Math.max(currentCredits, 0));
						} else if (
							!app.isPackaged &&
							parsedMessage.done &&
							parsedMessage.proUsed &&
							pricing === "free"
						) {
							console.log(
								"DEV MODE: Pro feature used on free plan (simulated). Would decrement in prod."
							)
						}
					} else if (parsedMessage.type === "stream_end") {
						// Handle explicit end message if server sends one
						console.log(
							"Stream ended signal received:",
							parsedMessage
						)
						// Potentially handle final proUsed check here as well if 'done' isn't reliable
						if (
							app.isPackaged &&
							parsedMessage.proUsed &&
							pricing === "free"
						) {
							// Potentially decrement here if the 'done' flag wasn't in the last assistantStream chunk
							// Requires careful coordination with server logic. Sticking to original for now.
							console.log(
								"PROD MODE: Stream end signal indicated pro usage on free plan."
							)
						}
					}
					// Handle other message types if necessary
				} catch (parseError) {
					console.warn(
						// Use warn instead of log for parsing errors
						`Error parsing streamed JSON message: ${parseError}. Message: "${msg}"`
					)
				}
			}
		}

		// Process any remaining data in the buffer (less common with newline splitting)
		if (buffer.trim()) {
			console.warn(
				"Processing remaining buffer after stream loop:",
				buffer
			)
			try {
				const parsedMessage = JSON.parse(buffer)
				// Handle potential final message (check type carefully)
				if (
					parsedMessage.type === "assistantStream" ||
					parsedMessage.type === "assistantMessage"
				) {
					mainWindow.webContents.send("message-stream", {
						messageId:
							parsedMessage.messageId ||
							`final-${Date.now()}-${Math.random()}`,
						token:
							parsedMessage.token || parsedMessage.message || ""
					})
					// Final check for proUsed if applicable and not caught by 'done' flag earlier
					if (
						app.isPackaged &&
						parsedMessage.proUsed &&
						!parsedMessage.done && // Avoid double counting if 'done' flag already triggered decrement
						pricing === "free"
					) {
						console.warn(
							"PROD MODE: Final buffer chunk indicated pro usage on free plan. Decrementing credit."
						)
						let currentCredits = await getCreditsFromKeytar()
						currentCredits -= 1
						await setCreditsInKeytar(Math.max(currentCredits, 0))
						// mainWindow.webContents.send("credits-updated", Math.max(currentCredits, 0));
					}
				}
			} catch (parseError) {
				console.error(
					// Use error for final buffer parse failure
					`Error parsing final buffer message: ${parseError}. Buffer: "${buffer}"`
				)
			}
		}
		// --- End: Stream Handling from "Original node" ---

		console.log("IPC: Streaming complete.") // Added logging
		return { message: "Streaming complete", status: 200 }
	} catch (error) {
		// Using error handling from "Current code"
		console.error(`IPC Error: send-message handler failed: ${error}`)
		// Ensure error message is sent to renderer
		mainWindow?.webContents.send("chat-error", {
			message: `Failed to process message: ${error.message}`
		})
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

ipcMain.handle("scrape-reddit", async (_event, { redditProfileUrl }) => {
	console.log("IPC: scrape-reddit called for URL:", redditProfileUrl)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/scrape-reddit`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ url: redditProfileUrl })
			}
		)
		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`Error from scrape-reddit API: ${response.status} ${response.statusText}`,
				errorText
			)
			return {
				message: "Error scraping Reddit profile",
				status: response.status,
				error: errorText
			}
		}
		const { topics } = await response.json()
		console.log("IPC: scrape-reddit successful.")
		return {
			message: "Reddit profile scraped successfully",
			topics,
			status: 200
		}
	} catch (error) {
		console.error(`IPC Error: scrape-reddit failed: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

ipcMain.handle("scrape-twitter", async (_event, { twitterProfileUrl }) => {
	console.log("IPC: scrape-twitter called for URL:", twitterProfileUrl)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/scrape-twitter`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ url: twitterProfileUrl })
			}
		)
		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`Error from scrape-twitter API: ${response.status} ${response.statusText}`,
				errorText
			)
			return {
				message: "Error scraping Twitter profile",
				status: response.status,
				error: errorText
			}
		}
		const { topics } = await response.json()
		console.log("IPC: scrape-twitter successful.")
		return {
			message: "Twitter profile scraped successfully",
			topics,
			status: 200
		}
	} catch (error) {
		console.error(`IPC Error: scrape-twitter failed: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

// --- Graph Handlers ---

ipcMain.handle("build-personality", async () => {
	console.log("IPC: build-personality called")
	try {
		console.log("Calling /create-document...")
		const documentResponse = await fetch(
			`${process.env.APP_SERVER_URL}/create-document`,
			{ method: "POST" }
		)
		if (!documentResponse.ok) {
			const errorText = await documentResponse.text()
			console.error(
				`Error from /create-document: ${documentResponse.status}`,
				errorText
			)
			return {
				message: "Error creating document",
				status: documentResponse.status,
				error: errorText
			}
		}
		const { personality } = await documentResponse.json()
		console.log("Document created successfully. Personality:", personality)

		// Update local DB with personality *before* creating graph if graph depends on it
		console.log("Updating local DB with personality...")
		const addDataResponse = await fetch(
			`${process.env.APP_SERVER_URL}/add-db-data`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ data: { userData: { personality } } })
			}
		)
		if (!addDataResponse.ok) {
			const errorText = await addDataResponse.text()
			console.error(
				`Error from /add-db-data: ${addDataResponse.status}`,
				errorText
			)
			return {
				message: "Error updating local DB with personality",
				status: addDataResponse.status,
				error: errorText
			}
		}
		console.log("Local DB updated with personality.")

		console.log("Calling /initiate-long-term-memories...")
		const graphResponse = await fetch(
			`${process.env.APP_SERVER_URL}/initiate-long-term-memories`,
			{ method: "POST" }
		)
		if (!graphResponse.ok) {
			const errorText = await graphResponse.text()
			console.error(
				`Error from /initiate-long-term-memories: ${graphResponse.status}`,
				errorText
			)
			return {
				message: "Error creating graph",
				status: graphResponse.status,
				error: errorText
			}
		}
		console.log("Graph created successfully.")
		return {
			message: "Document and Graph created successfully",
			status: 200
		}
	} catch (error) {
		console.error(`IPC Error: build-personality failed: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

ipcMain.handle("reset-long-term-memories", async () => {
	console.log("IPC: reset-long-term-memories called")
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/initiate-long-term-memories`,
			{ method: "POST" }
		)
		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`Error from /initiate-long-term-memories (recreate): ${response.status}`,
				errorText
			)
			return {
				message: "Error recreating graph",
				status: response.status,
				error: errorText
			}
		}
		console.log("Graph recreated successfully.")
		return { message: "Graph recreated successfully", status: 200 }
	} catch (error) {
		console.error(`IPC Error: reset-long-term-memories failed: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

ipcMain.handle(
	"customize-long-term-memories",
	async (_event, { newGraphInfo }) => {
		console.log(
			"IPC: customize-long-term-memories called with info:",
			newGraphInfo.substring(0, 50) + "..."
		)
		let credits = 999 // Dev default
		let pricing = "pro" // Dev default

		try {
			if (app.isPackaged) {
				credits = await getCreditsFromKeytar()
				pricing = await getPricingFromKeytar()
				console.log(
					`PROD MODE: Customizing graph with credits=${credits}, pricing=${pricing}`
				)
			} else {
				console.log(
					`DEV MODE: Customizing graph with mock credits=${credits}, pricing=${pricing}`
				)
			}

			const response = await fetch(
				`${process.env.APP_SERVER_URL}/customize-long-term-memories`,
				{
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						information: newGraphInfo,
						credits: credits
					}) // Send current credits
				}
			)

			if (!response.ok) {
				const errorText = await response.text()
				console.error(
					`Error from /customize-long-term-memories: ${response.status}`,
					errorText
				)
				return {
					message: "Error customizing graph",
					status: response.status,
					error: errorText
				}
			}

			const result = await response.json() // Expecting { message: string, pro_used: boolean } from backend?
			console.log("Customize graph successful:", result)

			// Decrement credit only if in production, on free plan, and if the API indicates pro was used
			if (
				app.isPackaged &&
				pricing === "free" &&
				result?.pro_used === true
			) {
				console.log(
					"PROD MODE: Pro customization used on free plan. Decrementing credit."
				)
				await ipcMain.handle("decrement-pro-credits")
			} else if (pricing === "free" && result?.pro_used === true) {
				console.log(
					"DEV MODE: Pro customization used (simulated), would decrement credit in prod."
				)
			}

			return {
				message: result.message || "Graph customized successfully",
				status: 200
			}
		} catch (error) {
			console.error(
				`IPC Error: customize-long-term-memories failed: ${error}`
			)
			return { message: `Error: ${error.message}`, status: 500 }
		}
	}
)

ipcMain.handle("delete-subgraph", async (_event, { source_name }) => {
	console.log("IPC: delete-subgraph called for source:", source_name)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/delete-subgraph`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ source: source_name })
			}
		)
		// Assuming the API returns JSON with status, e.g., { status: 'success' } or { status: 'failure', error: '...' }
		const result = await response.json()
		if (!response.ok || result.status === "failure") {
			console.error(
				`Error from /delete-subgraph: ${response.status}`,
				result.error || "Unknown error"
			)
			return {
				status: "failure",
				error:
					result.error || `Server responded with ${response.status}`
			}
		}
		console.log("Delete subgraph successful:", result)
		return result // Return the full response from API (e.g., { status: 'success' })
	} catch (error) {
		console.error(`IPC Error: delete-subgraph failed: ${error}`)
		return { status: "failure", error: error.message }
	}
})

// --- Task Management Handlers ---

ipcMain.handle("fetch-tasks", async () => {
	console.log("IPC: fetch-tasks called")
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/fetch-tasks`
		)
		if (!response.ok) {
			const errorText = await response.text()
			console.error(`Error fetching tasks: ${response.status}`, errorText)
			throw new Error(`HTTP error! status: ${response.status}`)
		}
		const result = await response.json()
		console.log("IPC: fetch-tasks successful.")
		return result
	} catch (error) {
		console.error("IPC Error: fetch-tasks failed:", error)
		return { error: error.message }
	}
})

ipcMain.handle("add-task", async (_event, taskData) => {
	console.log("IPC: add-task called with:", taskData)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/add-task`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(taskData)
			}
		)
		if (!response.ok) {
			const errorText = await response.text()
			console.error(`Error adding task: ${response.status}`, errorText)
			throw new Error(`HTTP error! status: ${response.status}`)
		}
		const result = await response.json()
		console.log("IPC: add-task successful:", result)
		return result
	} catch (error) {
		console.error("IPC Error: add-task failed:", error)
		return { error: error.message }
	}
})

ipcMain.handle(
	"update-task",
	async (_event, { taskId, description, priority }) => {
		console.log(`IPC: update-task called for ID ${taskId} with:`, {
			description,
			priority
		})
		try {
			const response = await fetch(
				`${process.env.APP_SERVER_URL || "http://localhost:5000"}/update-task`,
				{
					method: "POST", // Kept POST as per original code
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						task_id: taskId,
						description,
						priority
					})
				}
			)
			if (!response.ok) {
				const errorText = await response.text()
				console.error(
					`Error updating task ${taskId}: ${response.status}`,
					errorText
				)
				throw new Error(`HTTP error! status: ${response.status}`)
			}
			const result = await response.json()
			console.log(`IPC: update-task for ID ${taskId} successful:`, result)
			return result
		} catch (error) {
			console.error(
				`IPC Error: update-task for ID ${taskId} failed:`,
				error
			)
			return { error: error.message }
		}
	}
)

ipcMain.handle("delete-task", async (_event, taskId) => {
	console.log("IPC: delete-task called for ID:", taskId)
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/delete-task`,
			{
				method: "POST", // Kept POST as per original code
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ task_id: taskId })
			}
		)
		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`Error deleting task ${taskId}: ${response.status}`,
				errorText
			)
			throw new Error(`HTTP error! status: ${response.status}`)
		}
		const result = await response.json()
		console.log(`IPC: delete-task for ID ${taskId} successful:`, result)
		return result
	} catch (error) {
		console.error(`IPC Error: delete-task for ID ${taskId} failed:`, error)
		return { error: error.message }
	}
})

// --- Memory Management Handlers (Short-Term / SQLite) ---

ipcMain.handle("fetch-short-term-memories", async (_event, { category }) => {
	console.log(
		`IPC: fetch-short-term-memories called for category: ${category}`
	)
	try {
		const userId = await getUserIdForMemoryOps() // Get real or mock user ID
		console.log(`Using User ID: ${userId} for fetching memories.`)

		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/get-short-term-memories`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ user_id: userId, category, limit: 10 }) // Added limit as per original
			}
		)

		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`Error fetching memories for ${userId}, category ${category}: ${response.status}`,
				errorText
			)
			throw new Error(
				`Failed to fetch memories: Status ${response.status}`
			)
		}

		const memories = await response.json()
		console.log(
			`IPC: fetch-short-term-memories for ${category} successful.`
		)
		return memories // Should be the list of memories
	} catch (error) {
		console.error("IPC Error: fetch-short-term-memories failed:", error)
		return [] // Return empty array on error as per original code
	}
})

ipcMain.handle("add-short-term-memory", async (_event, memoryData) => {
	console.log("IPC: add-short-term-memory called with:", memoryData)
	try {
		const userId = await getUserIdForMemoryOps()
		const requestBody = { user_id: userId, ...memoryData }
		console.log(`Adding memory for User ID: ${userId}`)

		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/add-short-term-memory`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(requestBody)
			}
		)

		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`Error adding memory for ${userId}: ${response.status}`,
				errorText
			)
			throw new Error(`HTTP error! status: ${response.status}`)
		}

		const result = await response.json()
		console.log("IPC: add-short-term-memory successful:", result)
		return result
	} catch (error) {
		console.error("IPC Error: add-short-term-memory failed:", error)
		return { error: error.message }
	}
})

ipcMain.handle("update-short-term-memory", async (_event, memoryData) => {
	// memoryData should include the memory ID (e.g., memory_id) and updated fields
	console.log("IPC: update-short-term-memory called with:", memoryData)
	try {
		const userId = await getUserIdForMemoryOps()
		const requestBody = { user_id: userId, ...memoryData }
		console.log(
			`Updating memory ID ${memoryData.memory_id} for User ID: ${userId}`
		) // Log ID if available

		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/update-short-term-memory`,
			{
				method: "POST", // Assuming POST, adjust if your API uses PUT/PATCH
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(requestBody)
			}
		)

		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`Error updating memory for ${userId}: ${response.status}`,
				errorText
			)
			throw new Error(`HTTP error! status: ${response.status}`)
		}

		const result = await response.json()
		console.log("IPC: update-short-term-memory successful:", result)
		return result
	} catch (error) {
		console.error("IPC Error: update-short-term-memory failed:", error)
		return { error: error.message }
	}
})

ipcMain.handle("delete-short-term-memory", async (_event, memoryData) => {
	// memoryData should include the ID of the memory to delete (e.g., memory_id)
	console.log("IPC: delete-short-term-memory called with:", memoryData)
	try {
		const userId = await getUserIdForMemoryOps()
		const requestBody = { user_id: userId, ...memoryData }
		console.log(
			`Deleting memory ID ${memoryData.memory_id} for User ID: ${userId}`
		) // Log ID if available

		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/delete-short-term-memory`,
			{
				method: "POST", // Assuming POST, adjust if your API uses DELETE with body or different structure
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(requestBody)
			}
		)

		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`Error deleting memory for ${userId}: ${response.status}`,
				errorText
			)
			throw new Error(`HTTP error! status: ${response.status}`)
		}

		const result = await response.json()
		console.log("IPC: delete-short-term-memory successful:", result)
		return result
	} catch (error) {
		console.error("IPC Error: delete-short-term-memory failed:", error)
		return { error: error.message }
	}
})

ipcMain.handle("clear-all-short-term-memories", async () => {
	console.log("IPC: clear-all-short-term-memories called")
	try {
		const userId = await getUserIdForMemoryOps()
		console.log(`Clearing all memories for User ID: ${userId}`)

		const response = await fetch(
			`${process.env.APP_SERVER_URL || "http://localhost:5000"}/clear-all-short-term-memories`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ user_id: userId })
			}
		)

		if (!response.ok) {
			const errorText = await response.text()
			console.error(
				`Error clearing memories for ${userId}: ${response.status}`,
				errorText
			)
			throw new Error(`HTTP error! status: ${response.status}`)
		}

		const result = await response.json()
		console.log("IPC: clear-all-short-term-memories successful:", result)
		return result
	} catch (error) {
		console.error("IPC Error: clear-all-short-term-memories failed:", error)
		return { error: error.message }
	}
})

ipcMain.handle("fetch-long-term-memories", async () => {
	console.log("IPC: fetch-long-term-memories called")
	try {
		// Ensure APP_SERVER_URL is defined in your environment variables
		const apiUrl = `${process.env.APP_SERVER_URL}/get-graph-data`
		if (!apiUrl) {
			throw new Error("APP_SERVER_URL environment variable is not set.")
		}

		console.log(`Fetching graph data from backend: ${apiUrl}`)
		const response = await fetch(apiUrl, {
			method: "POST", // Using POST as defined in FastAPI
			headers: {
				"Content-Type": "application/json"
				// Add any necessary authentication headers here if required by your backend
				// 'Authorization': `Bearer ${token}`
			},
			// No body needed for this specific request, but sending empty JSON
			// is often good practice for POST if the backend expects it.
			body: JSON.stringify({})
		})

		if (!response.ok) {
			let errorText = response.statusText
			try {
				// Try to get more detailed error from FastAPI response body
				const errorBody = await response.json()
				errorText = errorBody.detail || errorText
			} catch (e) {
				// Ignore if response body isn't JSON
			}
			console.error(
				`Error fetching graph data from backend: ${response.status} ${errorText}`
			)
			// Return an error object that the frontend can check
			return {
				error: `Failed to fetch graph data: ${errorText}`,
				status: response.status
			}
		}

		// Parse the JSON response which should contain { nodes, edges }
		const graphData = await response.json()
		console.log("IPC: fetch-long-term-memories successful.")
		// Directly return the data in the expected format
		return graphData // Should be { nodes: [...], edges: [...] }
	} catch (error) {
		console.error(`IPC Error: fetch-long-term-memories failed: ${error}`)
		// Return an error object
		return {
			error: `Network or processing error: ${error.message}`,
			status: 500
		}
	}
})

ipcMain.handle("get-notifications", async () => {
	console.log("IPC: get-notifications called")
	try {
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/get-notifications`
		)
		if (!response.ok) {
			const errorDetail = await response
				.json()
				.catch(() => ({ detail: "Failed to parse error response" }))
			console.error(
				`Error from get-notifications API: ${response.status}`,
				errorDetail
			)
			return {
				message: `Error fetching notifications: ${errorDetail.detail || response.statusText}`,
				status: response.status
			}
		}
		const result = await response.json()
		console.log("IPC: get-notifications successful.")
		return { notifications: result.notifications, status: 200 }
	} catch (error) {
		console.error(`IPC Error: get-notifications failed: ${error}`)
		return {
			message: "Error fetching notifications",
			status: 500,
			error: error.message
		}
	}
})

ipcMain.handle("get-task-approval-data", async (event, taskId) => {
	// Destructure taskId from the input object
	console.log(`IPC: get-task-approval-data called for taskId: ${taskId}`)
	if (!taskId) {
		console.error(
			"IPC Error: get-task-approval-data called without taskId."
		)
		return { status: 400, message: "Task ID is required." }
	}
	try {
		const apiUrl = `http://localhost:5000/get-task-approval-data` // Updated URL (no ID in path)
		console.log(
			`Fetching approval data from: ${apiUrl} with taskId: ${taskId}`
		)

		const response = await fetch(apiUrl, {
			method: "POST", // Changed to POST
			headers: {
				"Content-Type": "application/json" // Essential header for JSON body
			},
			body: JSON.stringify({ task_id: taskId }) // Send taskId in the request body
		})

		// Try to parse JSON regardless of ok status for potential error details
		let responseBody
		const contentType = response.headers.get("content-type")
		if (contentType && contentType.includes("application/json")) {
			responseBody = await response.json()
		} else {
			responseBody = { detail: await response.text() } // Fallback for non-JSON errors
		}

		if (!response.ok) {
			console.error(
				`Error from backend (${response.status}) for get-task-approval-data:`,
				responseBody
			)
			return {
				status: response.status,
				message: responseBody.detail || `HTTP error ${response.status}`
			}
		}

		console.log(`IPC: get-task-approval-data successful for ${taskId}.`)
		// Assuming the response body structure is now { approval_data: ... }
		return { status: 200, approval_data: responseBody.approval_data }
	} catch (error) {
		console.error(
			`IPC Error: Failed to fetch approval data for ${taskId}:`,
			error
		)
		return {
			status: 500,
			message: `Error fetching approval data: ${error.message}`
		}
	}
})

// Handler to approve a task - now sends ID in body
ipcMain.handle("approve-task", async (event, taskId) => {
	// Destructure taskId from the input object
	console.log(`IPC: approve-task called for taskId: ${taskId}`)
	if (!taskId) {
		console.error("IPC Error: approve-task called without taskId.")
		return { status: 400, message: "Task ID is required." }
	}
	try {
		const apiUrl = `http://localhost:5000/approve-task` // Updated URL (no ID in path)
		console.log(
			`Sending approval request to: ${apiUrl} for taskId: ${taskId}`
		)

		const response = await fetch(apiUrl, {
			method: "POST",
			headers: {
				"Content-Type": "application/json" // Essential header for JSON body
			},
			body: JSON.stringify({ task_id: taskId }) // Send taskId in the request body
		})

		// Try to parse JSON regardless of ok status
		let responseBody
		const contentType = response.headers.get("content-type")
		if (contentType && contentType.includes("application/json")) {
			responseBody = await response.json()
		} else {
			responseBody = {
				detail: await response.text(),
				message: await response.text()
			} // Fallback
		}

		if (!response.ok) {
			console.error(
				`Error from backend (${response.status}) for approve-task:`,
				responseBody
			)
			// Use 'detail' from FastAPI or 'message' as fallback
			return {
				status: response.status,
				message:
					responseBody.detail ||
					responseBody.message ||
					`HTTP error ${response.status}`
			}
		}

		console.log(`IPC: approve-task successful for ${taskId}.`)
		// Assuming the response body structure is now { message: ..., result: ... }
		return {
			status: 200,
			message: responseBody.message,
			result: responseBody.result
		}
	} catch (error) {
		console.error(`IPC Error: Failed to approve task ${taskId}:`, error)
		return {
			status: 500,
			message: `Error approving task: ${error.message}`
		}
	}
})

ipcMain.handle("get-data-sources", async () => {
	try {
		const response = await fetch("http://localhost:5000/get_data_sources")
		if (!response.ok) {
			throw new Error("Failed to fetch data sources")
		}
		const data = await response.json()
		return data
	} catch (error) {
		console.error("Error fetching data sources:", error)
		return { error: error.message }
	}
})

ipcMain.handle("set-data-source-enabled", async (event, { source, enabled }) => {
	console.log(
		`IPC: set-data-source-enabled called for source: ${source}, enabled: ${enabled}`
	)
	try {
		const response = await fetch(
			"http://localhost:5000/set_data_source_enabled",
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ source, enabled })
			}
		)
		if (!response.ok) {
			throw new Error("Failed to set data source enabled")
		}
		const data = await response.json()
		return data
	} catch (error) {
		console.error("Error setting data source enabled:", error)
		return { error: error.message }
	}
})

ipcMain.handle("clear-chat-history", async (event) => {
	console.log("[IPC Main] Received clear-chat-history request.")
	const targetUrl = `${process.env.APP_SERVER_URL || "http://localhost:5000"}/clear-chat-history`

	try {
		const response = await fetch(targetUrl, {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
				// Add any other necessary headers like Authorization if needed
			}
			// No body is needed for this specific endpoint based on the Python code
		})

		// Check if the fetch itself was successful (status code 2xx)
		if (!response.ok) {
			// Try to get error details from the backend response body
			let errorDetail = `Backend responded with status ${response.status}`
			try {
				const errorData = await response.json()
				errorDetail = errorData.detail || JSON.stringify(errorData) // FastAPI often uses 'detail' for HTTPException
			} catch (jsonError) {
				// If the error response is not JSON, use the status text
				errorDetail = response.statusText
			}
			console.error(
				`[IPC Main] Failed to clear chat history on backend: ${errorDetail}`
			)
			// Return an object that includes the status for the renderer to check
			return {
				status: response.status,
				error: `Failed to clear history: ${errorDetail}`
			}
		}

		// If the request was successful (e.g., 200 OK)
		const responseData = await response.json() // Even if successful, parse the response
		console.log("[IPC Main] Backend successfully cleared chat history.")
		return {
			status: response.status, // Should be 200
			message: responseData.message // Pass the success message along
		}
	} catch (error) {
		console.error(
			"[IPC Main] Network or fetch error calling clear-chat-history:",
			error
		)
		// Return an error structure consistent with other failures
		return {
			status: 500, // Indicate an internal/communication error
			error: `Failed to communicate with backend: ${error.message}`
		}
	}
})

// --- End of File ---
