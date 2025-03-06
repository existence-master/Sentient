import { app, BrowserWindow, ipcMain, dialog } from "electron"
import path, { dirname } from "path"
import { fileURLToPath } from "url"
import { v4 as uuid } from "uuid"
import fetch from "node-fetch"
import { getChatsDb, getUserProfileDb } from "../utils/db.js"
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
	setBetaUserStatusInKeytar,
	hasApiKey,
	setApiKey,
	deleteApiKey
} from "../utils/auth.js"
import { getPrivateData } from "../utils/api.js"
import { createAuthWindow, createLogoutWindow } from "./auth.js"
import { setupServer } from "../scripts/setupServer.js"
import { startAppServer } from "../scripts/appServer.js"
import { startNeo4jServer, stopNeo4jServer } from "../scripts/neo4j.js"
import dotenv from "dotenv"
import pkg from "electron-updater"
const { autoUpdater } = pkg
import serve from "electron-serve"
import fs from "fs"

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
let chatsDbPath
/**
 * Path to the user profile database file (JSON).
 * @type {string}
 */
let userProfileDbPath
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

// Define upload directory
const uploadDir = path.join(__dirname, '../uploads')

// Ensure upload directory exists
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir)
}


// Determine paths based on whether the application is packaged or not
if (app.isPackaged) {
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
	chatsDbPath = path.join(basePath, "chatsDb.json")
	userProfileDbPath = path.join(basePath, "userProfileDb.json")
	appOutDir = path.join(__dirname, "../out")
} else {
	basePath = __dirname
	dotenvPath = path.resolve(__dirname, "../.env")
	chatsDbPath = path.resolve(__dirname, "../../chatsDb.json")
	userProfileDbPath = path.resolve(__dirname, "../../userProfileDb.json")
	appOutDir = path.join(__dirname, "../out")
}

// Load environment variables from .env file
dotenv.config({ path: dotenvPath })

/**
 * Database instance for chats.
 * @type {lowdb.Low<lowdb.Data<any>>}
 */
const chatsDb = await getChatsDb(chatsDbPath)
/**
 * Database instance for user profile data.
 * @type {lowdb.Low<lowdb.Data<any>>}
 */
const userProfileDb = await getUserProfileDb(userProfileDbPath)

if (!userProfileDb.data.userData.selectedModel) {
	userProfileDb.data.userData.selectedModel = "llama3.2:3b"
	await userProfileDb.write()
}

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
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

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
			enableRemoteModule: false,
			devTools: true // Enable DevTools for development and debugging
		},
		backgroundColor: "#000000",
		autoHideMenuBar: true // Hides the default menu bar
	})

	// Load app URL based on environment (packaged or development)
	if (app.isPackaged) {
		appServe(mainWindow).then(() => mainWindow.loadURL("app://-"))
	} else {
		mainWindow.loadURL(process.env.ELECTRON_APP_URL)
		mainWindow.webContents.openDevTools() // Open DevTools in development mode
		mainWindow.webContents.on("did-fail-load", () => {
			mainWindow.webContents.reloadIgnoringCache() // Reload page if load fails
		})
	}

	// Handle window closing event to confirm exit
	let isExiting = false
	mainWindow.on("close", async (event) => {
		if (!isExiting) {
			event.preventDefault() // Prevent default close action
			isExiting = true

			// Show confirmation dialog before quitting
			const response = await dialog.showMessageBox(mainWindow, {
				type: "question",
				buttons: ["Yes", "No"],
				defaultId: 1,
				title: "Confirm Exit",
				message: "Are you sure you want to quit?"
			})

			// If user confirms exit, attempt to stop servers and quit the app
			if (response.response === 0) {
				try {
					// await stopNeo4jServer() // Commented out: Stop Neo4j server on app close (currently disabled)
					// await stopAppSubServers() // Commented out: Stop app sub-servers on app close (currently disabled)
					await dialog.showMessageBox(mainWindow, {
						type: "info",
						message: "Bye friend...",
						title: "Quitting Sentient (click ok to exit)"
					})
					setTimeout(() => app.quit(), 1000) // Quit app after a short delay
				} catch (error) {
					await console.log(
						`Error while stopping processes: ${error}`
					)
					dialog.showErrorBox(
						"Error",
						"Failed to stop some processes. Please try again."
					)
					isExiting = false // Reset exiting flag on error
				}
			} else {
				isExiting = false // Reset exiting flag if user cancels exit
			}
		}
	})
}

/**
 * Checks user information validity, including pricing status, referral code,
 * beta user status, credits, and Google credentials. If any check fails,
 * it triggers logout.
 * @async
 * @returns {Promise<void>}
 * @throws {Error} If any user info check fails, leading to logout.
 */
const checkUserInfo = async () => {
	try {
		await checkPricingStatus()
		await checkReferralCodeAndStatus()
		await checkBetaUserStatus()
		await resetCreditsIfNecessary()
		await checkGoogleCredentials()
		await console.log("User info checked.")
	} catch (error) {
		await console.log(`Error checking user info: ${error}`)
		BrowserWindow.getAllWindows().forEach((win) => win.close())
		createLogoutWindow() // Create logout window on user info check failure
	}
}

/**
 * Checks overall application validity by verifying authentication and user info.
 * If validity checks fail, redirects to the authentication window.
 * @async
 * @returns {Promise<void>}
 * @throws {Error} If authentication or user info is invalid, leading to auth window.
 */
export const checkValidity = async () => {
	try {
		await checkAuthStatus()
		await checkUserInfo()
	} catch (error) {
		await console.log(`Error checking validity: ${error}`)
		createAuthWindow() // Create auth window on validity check failure
	}
}

/**
 * Checks user authentication status by refreshing tokens.
 * Throws an error if token refresh fails, indicating user is not authenticated.
 * @async
 * @returns {Promise<void>}
 * @throws {Error} If refreshTokens fails, indicating authentication failure.
 */
const checkAuthStatus = async () => {
	try {
		await refreshTokens()
		await console.log("User is authenticated.")
	} catch (error) {
		await console.log(`Error in authentication: ${error}`)
		throw new Error("User is not authenticated.")
	}
}

/**
 * Checks and fetches/sets referral code and referrer status from Keytar and API.
 * @async
 * @returns {Promise<void>}
 * @throws {Error} If referral code or status check fails.
 */
const checkReferralCodeAndStatus = async () => {
	try {
		let referralCode = await getReferralCodeFromKeytar()
		if (!referralCode) {
			await console.log("Referral code not found.")
			referralCode = await fetchAndSetReferralCode() // Fetch and set referral code if not found
			await console.log("Referral code set")
		}

		let referrerStatus = await getReferrerStatusFromKeytar()
		if (referrerStatus === null || referrerStatus === false) {
			await console.log("Referrer status not found.")
			referrerStatus = await fetchAndSetReferrerStatus() // Fetch and set referrer status if not found
			await console.log("Referrer status set")
		}

		await console.log("Referral code and status checked.")
	} catch (error) {
		await console.log(`Error in referral code and status check: ${error}`)
		throw new Error("Referral code and status check failed.")
	}
}

/**
 * Checks and fetches/sets beta user status from Keytar and API.
 * @async
 * @returns {Promise<void>}
 * @throws {Error} If beta user status check fails.
 */
const checkBetaUserStatus = async () => {
	try {
		let betaUserStatus = await getBetaUserStatusFromKeytar()
		if (betaUserStatus === null || betaUserStatus === false) {
			await console.log("Beta user status not found.")
			betaUserStatus = await fetchAndSetBetaUserStatus() // Fetch and set beta user status if not found
			await console.log("Beta user status set")
		}

		await console.log("Beta user status checked.")
	} catch (error) {
		await console.log(`Error in beta user status check: ${error}`)
		throw new Error("Beta user status check failed.")
	}
}

/**
 * Checks pricing status and user session validity. If session is expired due to inactivity
 * (7 days), it triggers logout.
 * @async
 * @returns {Promise<void>}
 * @throws {Error} If pricing check fails or session expired, leading to logout.
 */
const checkPricingStatus = async () => {
	try {
		const pricing = await getPricingFromKeytar()
		const lastCheckin = await getCheckinFromKeytar()

		if (!pricing || !lastCheckin) {
			await console.log("Pricing or last checkin not found.")
			await fetchAndSetUserRole() // Fetch and set user role/pricing if not found
			await console.log("Pricing set")
			return
		}

		const currentTimestamp = Math.floor(Date.now() / 1000)

		// Check if last check-in was more than 7 days ago
		if (currentTimestamp - lastCheckin >= 7 * 24 * 60 * 60) {
			BrowserWindow.getAllWindows().forEach((win) => win.close())
			createLogoutWindow() // Create logout window if session expired
			await console.log("User session expired due to inactivity.")
			throw new Error("User session expired due to inactivity.")
		}

		await console.log("Pricing checked.")
	} catch (error) {
		await console.log(`Error in pricing check: ${error}`)
		throw new Error("Pricing check failed.")
	}
}

/**
 * Resets pro credits if the current date is different from the last check-in date.
 * Gives 10 credits to referrers and 5 to regular users.
 * @async
 * @returns {Promise<void>}
 * @throws {Error} If there's an error resetting pro credits.
 */
const resetCreditsIfNecessary = async () => {
	try {
		const currentDate = new Date().toDateString()
		const checkinTimestamp = await getCreditsCheckinFromKeytar()
		const checkinDate = checkinTimestamp
			? new Date(checkinTimestamp * 1000).toDateString()
			: null

		// Reset credits only if the check-in date is not today
		if (checkinDate !== currentDate) {
			await console.log("Resetting proCredits.")
			const referrer = getReferrerStatusFromKeytar()

			if (referrer === true)
				await setCreditsInKeytar(10) // 10 credits for referrers
			else await setCreditsInKeytar(5) // 5 credits for non-referrers

			await console.log("proCredits reset.")

			await setCreditsCheckinInKeytar() // Update credits check-in timestamp
			await console.log("proCredits checkin set.")
		}
	} catch (error) {
		await console.log(`Error resetting proCredits: ${error}`)
		throw new Error("Error checking proCredits")
	}
}

/**
 * Checks Google credentials by making a request to the authentication server.
 * @async
 * @returns {Promise<void>}
 * @throws {Error} If Google credentials check fails.
 */
const checkGoogleCredentials = async () => {
	try {
		const response = await fetch(
			"http://127.0.0.1:5007/authenticate-google" // URL to check Google auth status
		)
		const data = await response.json()

		if (data.success) {
			await console.log("Google credentials checked.")
		} else {
			throw new Error(data.error)
		}
	} catch (error) {
		await console.log(`Error checking Google credentials: ${error}`)
		throw new Error("Error checking Google credentials")
	}
}

/**
 * Acquires a single instance lock for the application. If another instance is already running,
 * this instance will quit. Otherwise, it sets up event listeners for second instances.
 */
const gotTheLock = app.requestSingleInstanceLock()

if (!gotTheLock) {
	app.quit() // Quit if another instance is already running
} else {
	app.on("second-instance", () => {
		// Focus on the main window if a second instance is attempted
		if (mainWindow) {
			if (mainWindow.isMinimized()) mainWindow.restore()
			mainWindow.focus()
		}
	})
}

/**
 * Initializes application servers (App server, Auth server, Utils server).
 * Sets a flag to indicate that the app server is loaded.
 * @async
 * @returns {Promise<void>}
 */
const initializeAppServer = async () => {
	// startAppServer() // Start the main app server (FastAPI) - currently disabled
	// startAuthServer() // Start the authentication server - currently disabled
	// startUtilsServer() // Start the utilities server - currently disabled
	process.env.APP_SERVER_LOADED = "true" // Set flag indicating app server is loaded
}

/**
 * Checks for application updates using Electron AutoUpdater.
 * Sets the feed URL based on whether the user is a beta user or not.
 * Then creates the main application window and initiates update checks.
 * @async
 * @returns {Promise<void>}
 */

const checkForUpdates = async () => {
	/**
	 *  Indicates whether the current user is a beta user.
	 *  If true, the application will check for pre-release (beta) updates.
	 *  If false, the application will check for stable updates.
	 */
	const betaUserStatus = true

	if (betaUserStatus) {
		// Configure autoUpdater to check for pre-release updates if the user is a beta user.
		autoUpdater.setFeedURL({
			provider: "github", // Specify the update provider as GitHub
			owner: "existence-master", // The owner of the GitHub repository
			repo: "Sentient" // The name of the GitHub repository
		})

		/**
		 * Allows autoUpdater to check for and download pre-release versions (e.g., beta, alpha).
		 * This is set to true for beta users to receive beta updates.
		 */
		autoUpdater.allowPrerelease = true
	} else {
		// Configure autoUpdater to check for stable releases if the user is not a beta user.
		autoUpdater.setFeedURL({
			provider: "github", // Specify the update provider as GitHub
			owner: "existence-master", // The owner of the GitHub repository
			repo: "Sentient" // The name of the GitHub repository
		})
		// autoUpdater.allowPrerelease is not set here, which defaults to false, ensuring only stable releases are checked.
	}

	/**
	 * Creates the main application window.
	 * This should be called after setting up the update feed to ensure the app window is ready regardless of update status.
	 */
	createAppWindow()

	/**
	 * Initiates the update check process and notifies the user if an update is available.
	 * This asynchronous call checks for updates based on the configured feed URL and displays a notification to the user.
	 */
	autoUpdater.checkForUpdatesAndNotify()
}

/**
 * Starts the application by checking for updates and then initializing app servers
 * and checking validity after a delay.
 * @async
 * @returns {Promise<void>}
 */
const startApp = async () => {
	await checkForUpdates() // Check for application updates
	setTimeout(async () => {
		if (process.env.UPDATING !== "true") {
			await initializeAppServer() // Initialize application servers
			console.log("App servers started successfully")
			await checkValidity() // Check application validity (auth and user info)
		}
	}, 10000) // Delay before starting servers and validity check
}

// Event listener when the app is ready to start
app.on("ready", startApp)

// Event listener for 'window-all-closed' event
app.on("window-all-closed", () => {
	if (process.platform !== "darwin") app.quit() // Quit app when all windows are closed, except on macOS
})

// AutoUpdater event: update available
autoUpdater.on("update-available", () => {
	process.env.UPDATING = "true" // Set flag indicating update is in progress
	mainWindow.webContents.send("update-available") // Send event to renderer process
})

// AutoUpdater event: update downloaded
autoUpdater.on("update-downloaded", () => {
	mainWindow.webContents.send("update-downloaded") // Send event to renderer process
})

// IPC event handler for 'restart-app' to quit and install update
ipcMain.on("restart-app", async () => {
	autoUpdater.quitAndInstall() // Quit and install the downloaded update
})

// IPC handler to get user profile
ipcMain.handle("get-profile", async () => {
	return getProfile()
})

// IPC handler to get private data
ipcMain.handle("get-private-data", async () => {
	return getPrivateData()
})

// IPC event handler for 'log-out' to close all windows and create logout window
ipcMain.on("log-out", () => {
	BrowserWindow.getAllWindows().forEach((win) => win.close()) // Close all browser windows
	createLogoutWindow() // Create logout window
})

// IPC handler to setup server
ipcMain.handle("setup-server", async () => {
	try {
		let response = await setupServer() // Run server setup script
		process.env.UPDATING = "false" // Reset updating flag after setup
		return { message: "Server setup successful", status: 200, response }
	} catch (error) {
		await console.log(`Error setting up server: ${error}`)
		return { message: "Error setting up server", status: 500 }
	}
})

// IPC handler to start Neo4j server
ipcMain.handle("start-neo4j", async () => {
	try {
		while (true) {
			if (process.env.NEO4J_SERVER_STARTED !== "true") {
				await startNeo4jServer() // Start Neo4j server
				process.env.NEO4J_SERVER_STARTED = "true" // Set flag when server start is attempted
			}

			const isReady = await pollServer(process.env.NEO4J_SERVER_URL) // Poll Neo4j server for readiness
			if (isReady) {
				console.log("Neo4j server started successfully")
				return {
					message: "Neo4j server started successfully",
					status: 200
				}
			} else {
				console.warn("Neo4j server is not ready. Restarting...")
				process.env.NEO4J_SERVER_STARTED = "false" // Reset flag if server is not ready, to attempt restart in next poll
			}
		}
	} catch (error) {
		await console.log(`Error in starting Neo4j server: ${error}`)
		return {
			message: `Error starting Neo4j server: ${error.message}`,
			status: 500
		}
	}
})

// IPC handler to stop Neo4j server
ipcMain.handle("stop-neo4j", async () => {
	try {
		await stopNeo4jServer() // Stop Neo4j server
		return { message: "Neo4j server stopped successfully", status: 200 }
	} catch (error) {
		await console.log(`Error in stopping Neo4j server: ${error}`)
		return {
			message: `Error stopping Neo4j server: ${error.message}`,
			status: 500
		}
	}
})

// IPC handler to fetch pricing plan
ipcMain.handle("fetch-pricing-plan", async () => {
	try {
		const pricing = await getPricingFromKeytar() // Get pricing plan from Keytar
		return pricing
	} catch (error) {
		await console.log(`Error fetching roles: ${error}`)
		return "free" // Default to 'free' if fetching fails
	}
})

// IPC handler to set referrer using referral code
ipcMain.handle("set-referrer", async (_event, { referralCode }) => {
	try {
		const apiUrl = `${process.env.APP_SERVER_URL}/get-user-and-set-referrer-status`
		// API call to set referrer status
		const response = await fetch(apiUrl, {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
			},
			body: JSON.stringify({ referral_code: referralCode })
		})

		if (!response.ok) {
			throw new Error(`Error from API: ${response.statusText}`)
		}

		await response.json()
		return { message: "Referrer updated successfully", status: 200 }
	} catch (error) {
		await console.log(`Error in sending referral code: ${error}`)
		return { error: error.message }
	}
})

// IPC handler to invert beta user status
ipcMain.handle("invert-beta-user-status", async () => {
	try {
		const apiUrl = `${process.env.UTILS_SERVER_URL}/get-user-and-invert-beta-user-status`
		const profile = await getProfile() // Get user profile

		console.log(`Inverting beta status for user: ${profile.sub}`)

		// API call to invert beta user status
		const response = await fetch(apiUrl, {
			method: "POST",
			headers: {
				"Content-Type": "application/json"
			},
			body: JSON.stringify({ user_id: profile.sub })
		})

		if (!response.ok) {
			throw new Error(`Error from API: ${response.statusText}`)
		}

		await response.json()

		const betaUserStatus = await getBetaUserStatusFromKeytar() // Get current beta user status

		console.log(`Beta status before inversion: ${betaUserStatus}`)
		const newBetaUserStatus = !betaUserStatus // Invert the status

		console.log(`Beta status after inversion: ${newBetaUserStatus}`)
		await setBetaUserStatusInKeytar(newBetaUserStatus) // Set the new beta user status in Keytar
		return { message: "Beta status inverted successfully", status: 200 }
	} catch (error) {
		await console.log(`Error in inverting beta status: ${error}`)
		return { error: error.message }
	}
})

// IPC handler to fetch pro credits
ipcMain.handle("fetch-pro-credits", async () => {
	try {
		const credits = await getCreditsFromKeytar() // Get pro credits from Keytar
		return credits
	} catch (error) {
		await console.log(`Error fetching credits: ${error}`)
		return 0 // Return 0 if fetching fails
	}
})

// IPC handler to fetch referral code
ipcMain.handle("fetch-referral-code", async () => {
	try {
		const referralCode = await getReferralCodeFromKeytar() // Get referral code from Keytar
		return referralCode
	} catch (error) {
		await console.log(`Error fetching referral code: ${error}`)
		return null // Return null if fetching fails
	}
})

// IPC handler to fetch referrer status
ipcMain.handle("fetch-referrer-status", async () => {
	try {
		const referrerStatus = await getReferrerStatusFromKeytar() // Get referrer status from Keytar
		return referrerStatus
	} catch (error) {
		await console.log(`Error fetching referral status: ${error}`)
		return null // Return null if fetching fails
	}
})

// IPC handler to fetch beta user status
ipcMain.handle("fetch-beta-user-status", async () => {
	try {
		const betaUserStatus = await getBetaUserStatusFromKeytar() // Get beta user status from Keytar
		return betaUserStatus
	} catch (error) {
		await console.log(`Error fetching beta user status: ${error}`)
		return null // Return null if fetching fails
	}
})

// IPC handler to decrement pro credits
ipcMain.handle("decrement-pro-credits", async () => {
	try {
		let credits = await getCreditsFromKeytar() // Get current credits
		credits -= 1 // Decrement credits
		await setCreditsInKeytar(Math.min(credits, 0)) // Set the decremented credits, minimum 0
		return true
	} catch (error) {
		await console.log(`Error decrementing credits: ${error}`)
		return false
	}
})

// IPC handler to set data in user profile database
ipcMain.handle("set-db-data", async (_event, args) => {
	/** @type {{ data: object }} */ const { data } = args // Type hinting for args
	try {
		userProfileDb.data.userData = {
			...userProfileDb.data.userData,
			...data // Merge existing user data with new data
		}
		await userProfileDb.write() // Write changes to database
		return { message: "Data stored successfully", status: 200 }
	} catch (error) {
		await console.log(`Error setting data: ${error}`)
		return { message: "Error storing data", status: 500 }
	}
})

// IPC handler to add data to user profile database, handles array and object merging
ipcMain.handle("add-db-data", async (_event, args) => {
	/** @type {{ data: object }} */ const { data } = args // Type hinting for args
	try {
		const existingData = userProfileDb.data.userData || {}
		// Iterate through the data to be added
		for (const [key, value] of Object.entries(data)) {
			if (Array.isArray(existingData[key]) && Array.isArray(value)) {
				// For arrays, merge and remove duplicates
				existingData[key] = [
					...new Set([...existingData[key], ...value])
				]
			} else if (
				typeof existingData[key] === "object" &&
				typeof value === "object"
			) {
				// For objects, merge properties
				existingData[key] = { ...existingData[key], ...value }
			} else {
				// For other types, simply overwrite or set new value
				existingData[key] = value
			}
		}
		userProfileDb.data.userData = existingData
		await userProfileDb.write() // Write changes to database
		return { message: "Data added successfully", status: 200 }
	} catch (error) {
		await console.log(`Error adding data: ${error}`)
		return { message: "Error adding data", status: 500 }
	}
})

// IPC handler to get all user profile database data
ipcMain.handle("get-db-data", async () => {
	try {
		return {
			data: userProfileDb.data.userData, // Return user data
			userProfileDbPath, // Return database path
			status: 200
		}
	} catch (error) {
		await console.log(`Error fetching data: ${error}`)
		return { message: "Error fetching data", status: 500 }
	}
})

// IPC handler to get all chats from database
ipcMain.handle("get-chats", async () => {
	try {
		return { data: { chats: chatsDb.data.chats }, status: 200 } // Return chats data
	} catch (error) {
		await console.log(`Error fetching data: ${error}`)
		return { message: "Error fetching data", status: 500 }
	}
})

/**
 * Polls a server URL until it's ready (responds with status 200).
 * @async
 * @param {string} url - The URL of the server to poll.
 * @param {number} [interval=10000] - Polling interval in milliseconds.
 * @returns {Promise<boolean>} True if server is ready, false otherwise (after timeout/errors).
 */
const pollServer = async (url, interval = 10000) => {
	while (true) {
		try {
			const response = await fetch(url) // Attempt to fetch from the URL
			if (response.ok) {
				return true // Server is ready if response is ok
			}
		} catch (error) {
			await console.log(`Error in polling: ${error}`)
		}
		await delay(interval) // Wait before next poll attempt
	}
}

// IPC handler to initiate FastAPI server and model
ipcMain.handle("initiate-fastapi-server", async () => {
	try {
		startAppServer() // Start FastAPI server

		while (true) {
			const isReady = await pollServer(`${process.env.APP_SERVER_URL}/`) // Poll server root URL
			if (isReady) {
				console.log("App server started successfully")
				if (process.env.APP_SERVER_INITIATED !== "true") {
					// Initiate FastAPI model if not already initiated
					const initiateResponse = await fetch(
						`${process.env.APP_SERVER_URL}/initiate`,
						{ method: "POST" }
					)

					if (initiateResponse.ok) {
						process.env.APP_SERVER_INITIATED = "true" // Set flag after successful initiation
						console.log("App server initiated successfully")
						return {
							message: "FastAPI server initiated successfully",
							status: 200
						}
					} else {
						console.log("Failed to initiate FastAPI model.")
					}
				} else {
					console.log("FastAPI server is already initiated.")
					return {
						message:
							"FastAPI server is ready and initiated already",
						status: 200
					}
				}
			} else {
				process.env.APP_SERVER_LOADED = "false" // Reset server loaded flag if not ready
			}
		}
	} catch (error) {
		await console.log(`Error starting FastAPI server: ${error.message}`)
		return {
			message: `Error starting FastAPI server: ${error.message}`,
			status: 500
		}
	}
})

// IPC handler to fetch chat history by chat ID
ipcMain.handle("fetch-chat-history", async (_event, { chatId }) => {
	/** @type {{ chatId: string }} */
	try {
		const chat = chatsDb.data.chats.find((c) => c.id === chatId) // Find chat by ID

		if (!chat) {
			return { message: "Chat not found", status: 404 }
		}

		return { messages: chat.chatHistory, status: 200 } // Return chat history
	} catch (error) {
		await console.log(`Error fetching chat history: ${error}`)
		return { message: "Error fetching chat history", status: 500 }
	}
})

// IPC handler to rename a chat
ipcMain.handle("rename-chat", async (_event, { id, newName }) => {
	/** @type {{ id: string, newName: string }} */
	try {
		const chat = chatsDb.data.chats.find((c) => c.id === id) // Find chat by ID

		if (!chat) {
			return { message: "Chat not found", status: 404 }
		}

		chat.title = newName // Update chat title

		await chatsDb.write() // Write changes to database

		return { message: "Chat renamed successfully", status: 200 }
	} catch (error) {
		await console.log(`Error renaming chat: ${error}`)
		return { message: "Error renaming chat", status: 500 }
	}
})

// IPC handler to delete a chat
ipcMain.handle("delete-chat", async (_event, { id }) => {
	/** @type {{ id: string }} */
	try {
		const chatIndex = chatsDb.data.chats.findIndex((c) => c.id === id) // Find chat index by ID

		if (chatIndex === -1) {
			return { message: "Chat not found", status: 404 }
		}

		chatsDb.data.chats.splice(chatIndex, 1) // Remove chat from array

		await chatsDb.write() // Write changes to database

		return { message: "Chat deleted successfully", status: 200 }
	} catch (error) {
		await console.log(`Error deleting chat: ${error}`)
		return { message: "Error deleting chat", status: 500 }
	}
})

// IPC handler to scrape LinkedIn profile
ipcMain.handle("scrape-linkedin", async (_event, { linkedInProfileUrl }) => {
	/** @type {{ linkedInProfileUrl: string }} */
	try {
		// API call to scrape LinkedIn profile
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/scrape-linkedin`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ url: linkedInProfileUrl })
			}
		)

		if (!response.ok) {
			return { message: "Error scraping LinkedIn profile", status: 500 }
		}

		const { profile } = await response.json() // Extract profile data from response

		return {
			message: "LinkedIn profile scraped successfully",
			profile,
			status: 200
		}
	} catch (error) {
		await console.log(`Error scraping LinkedIn profile: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

// IPC handler to send a message to the chat API and stream response
ipcMain.handle("send-message", async (_event, { chatId, input }) => {
	/** @type {{ chatId: string, input: string }} */
	try {
		const chat = chatsDb.data.chats.find((c) => c.id === chatId) // Find chat by ID

		if (!chat) {
			return { message: "Chat not found", status: 404 }
		}

		const pricing = (await getPricingFromKeytar()) || "free" // Get pricing plan or default to 'free'
		const credits = await getCreditsFromKeytar() // Get current pro credits

		// API call to send chat message
		const response = await fetch(`${process.env.APP_SERVER_URL}/chat`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				input: input,
				pricing: pricing,
				credits: credits,
				chat_id: chatId
			})
		})

		if (!response.ok) {
			return { message: "Error sending message", status: 500 }
		}

		const readable = response.body // Get readable stream from response body
		const decoder = new TextDecoder("utf-8") // Text decoder for stream

		let buffer = "" // Buffer to hold streamed data

		// Iterate over stream chunks
		for await (const chunk of readable) {
			buffer += decoder.decode(chunk, { stream: true }) // Decode chunk and append to buffer

			const splitMessages = buffer.split("\n") // Split buffer by newline to get individual messages
			buffer = splitMessages.pop() // Keep the last (possibly incomplete) message in buffer

			// Process each complete message
			for (const msg of splitMessages) {
				try {
					const parsedMessage = JSON.parse(msg) // Parse each message as JSON

					if (parsedMessage.type === "assistantStream") {
						// Handle assistant stream messages (tokens)
						let last = chat.chatHistory[chat.chatHistory.length - 1]
						if (!last || last.isUser || last.toolResult) {
							// Create new assistant message if last message is user's or tool result
							last = {
								id: uuid(),
								message: "",
								isUser: false,
								memoryUsed: parsedMessage.memoryUsed,
								agentsUsed: parsedMessage.agentsUsed,
								internetUsed: parsedMessage.internetUsed
							}
							chat.chatHistory.push(last) // Add new message to chat history
						}
						mainWindow.webContents.send("message-stream", {
							chatId,
							messageId: last.id,
							token: parsedMessage.token // Send message stream event to renderer
						})

						last.message += parsedMessage.token // Append token to last message content
						if (parsedMessage.done === true) {
							// Update message flags when streaming is done
							last.memoryUsed = parsedMessage.memoryUsed
							last.agentsUsed = parsedMessage.agentsUsed
							last.internetUsed = parsedMessage.internetUsed

							if (
								parsedMessage.proUsed === true &&
								pricing === "free"
							) {
								// Decrement credits if pro features used in free plan
								let credits = await getCreditsFromKeytar()
								credits -= 1
								await setCreditsInKeytar(Math.max(credits, 0))
							}
						}
						await chatsDb.write() // Write changes to database
					} else if (parsedMessage.type === "userMessage") {
						// Handle user messages
						const userMessage = {
							id: uuid(),
							message: parsedMessage.message,
							isUser: true,
							memoryUsed: false,
							agentsUsed: false,
							internetUsed: false
						}
						chat.chatHistory.push(userMessage) // Add user message to chat history
					} else if (parsedMessage.type === "assistantMessage") {
						// Handle complete assistant messages (non-stream)
						const assistantMessage = {
							id: uuid(),
							message: parsedMessage.message,
							isUser: false,
							memoryUsed: parsedMessage.memoryUsed,
							agentsUsed: parsedMessage.agentsUsed,
							internetUsed: parsedMessage.internetUsed
						}
						if (
							parsedMessage.proUsed === true &&
							pricing === "free"
						) {
							// Decrement credits if pro features used in free plan
							let credits = await getCreditsFromKeytar()
							credits -= 1
							await setCreditsInKeytar(Math.max(credits, 0))
						}
						chat.chatHistory.push(assistantMessage) // Add assistant message to chat history
					}
					if (parsedMessage.type === "toolResult") {
						// Handle tool result messages
						let toolMessage
						if (parsedMessage.tool_name === "search_inbox") {
							toolMessage = {
								id: uuid(),
								message: JSON.stringify({
									type: "toolResult",
									tool_name: parsedMessage.tool_name,
									emails: parsedMessage.result.email_data,
									gmail_search_url:
										parsedMessage.result.gmail_search_url
								}),
								isUser: false,
								memoryUsed: false,
								agentsUsed: true,
								internetUsed: false,
								toolResult: true // Flag as tool result message
							}
						} else {
							toolMessage = "Tool has no additional information."
						}
						chat.chatHistory.push(toolMessage) // Add tool result message to chat history
						await chatsDb.write() // Write changes to database
					} else if (
						parsedMessage.type === "intermediary-flow-update"
					) {
						// Send intermediary flow update to renderer
						mainWindow.webContents.send(
							"flow-updated",
							parsedMessage.message
						)
					} else if (parsedMessage.type === "intermediary-flow-end") {
						// Send intermediary flow end signal to renderer
						mainWindow.webContents.send("flow-ended")
					} else {
						// Send status updates to renderer
						mainWindow.webContents.send(
							"status-updated",
							parsedMessage.message
						)
					}
					await chatsDb.write() // Write changes to database after each message part
				} catch (parseError) {
					console.log(
						`Error parsing streamed message: ${parseError}: ${msg}`
					)
				}
			}
		}

		// Handle any remaining data in buffer after stream ends
		if (buffer.trim()) {
			try {
				const parsedMessage = JSON.parse(buffer)

				if (parsedMessage.type === "assistantMessage") {
					chat.chatHistory.push({
						id: uuid(),
						message: parsedMessage.message,
						isUser: false,
						memoryUsed: parsedMessage.memoryUsed,
						agentsUsed: parsedMessage.agentsUsed,
						internetUsed: parsedMessage.internetUsed
					})
					await chatsDb.write() // Write changes to database
				}
			} catch (parseError) {
				console.log(`Error parsing final buffer message: ${parseError}`)
			}
		}

		return { message: "Streaming complete", status: 200 }
	} catch (error) {
		console.log(`Error sending message: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

// IPC handler to create document and graph in the server
ipcMain.handle("create-document-and-graph", async () => {
	try {
		// API call to create document
		const documentResponse = await fetch(
			`${process.env.APP_SERVER_URL}/create-document`,
			{ method: "POST" }
		)

		if (!documentResponse.ok) {
			return { message: "Error creating document", status: 500 }
		}

		const { personality } = await documentResponse.json() // Extract personality from response
		userProfileDb.data.userData.personality = personality // Store personality in user profile

		// API call to create graph
		const graphResponse = await fetch(
			`${process.env.APP_SERVER_URL}/create-graph`,
			{ method: "POST" }
		)

		if (!graphResponse.ok) {
			return { message: "Error creating graph", status: 500 }
		}

		return { message: "Graph created successfully", status: 200 }
	} catch (error) {
		await console.log(`Error creating document and graph: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

// IPC handler to recreate the user graph
ipcMain.handle("recreate-graph", async () => {
	try {
		// API call to recreate graph
		const graphResponse = await fetch(
			`${process.env.APP_SERVER_URL}/create-graph`,
			{ method: "POST" }
		)

		if (!graphResponse.ok) {
			return { message: "Error recreating graph", status: 500 }
		}

		return { message: "Graph recreated successfully", status: 200 }
	} catch (error) {
		await console.log(`Error recreating graph: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

// IPC handler to reinitiate FastAPI server with a specific chat context
ipcMain.handle("reinitiate-fastapi-server", async (_event, { chatId }) => {
	/** @type {{ chatId: string }} */
	try {
		// API call to initiate FastAPI server
		let response = await fetch(`${process.env.APP_SERVER_URL}/initiate`, {
			method: "POST"
		})

		if (response.ok) {
			// API call to set chat context
			response = await fetch(`${process.env.APP_SERVER_URL}/set-chat`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ id: chatId })
			})

			if (response.ok) {
				return {
					message: "FastAPI server reinitiated successfully",
					status: 200
				}
			} else {
				await console.log(`Failed to reinitiate FastAPI server`)
				return {
					message: "Failed to reinitiate FastAPI server",
					status: response.status
				}
			}
		} else {
			await console.log(`Failed to reinitiate FastAPI server`)
			return {
				message: "Failed to reinitiate FastAPI server",
				status: response.status
			}
		}
	} catch (error) {
		await console.log(`Error reinitiating FastAPI server: ${error}`)
		return {
			message: `Error reinitiating FastAPI server: ${error.message}`,
			status: 500
		}
	}
})

// IPC handler to clear chat history for a specific chat
ipcMain.handle("clear-chat-history", async (_event, { chatId }) => {
	/** @type {{ chatId: string }} */
	try {
		const chat = chatsDb.data.chats.find((c) => c.id === chatId) // Find chat by ID

		if (!chat) {
			return { message: "Chat not found", status: 404 }
		}

		chat.chatHistory = [] // Clear chat history array

		await chatsDb.write() // Write changes to database
		return { message: "Chat history cleared successfully", status: 200 }
	} catch (error) {
		await console.log(`Error clearing chat history: ${error}`)
		return { message: "Error clearing chat history", status: 500 }
	}
})

// IPC handler to create a new chat
ipcMain.handle("create-chat", async (_event, { title }) => {
	/** @type {{ title: string }} */
	try {
		const chatId = `chat-${uuid()}` // Generate a unique chat ID
		chatsDb.data.chats.push({
			id: chatId,
			title,
			chatHistory: [] // Initialize with empty chat history
		})
		await chatsDb.write() // Write changes to database
		return { message: "Chat created successfully", chatId, status: 200 }
	} catch (error) {
		await console.log(`Error creating chat: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

// IPC handler to set the current chat context in the server
ipcMain.handle("set-chat", async (_event, { chatId }) => {
	/** @type {{ chatId: string }} */
	try {
		// API call to set chat context
		const response = await fetch(`${process.env.APP_SERVER_URL}/set-chat`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ id: chatId })
		})

		if (!response.ok) {
			const errorMessage = await response.text()
			return { message: errorMessage, status: response.status }
		}

		const data = await response.json()
		return { message: data.message, status: response.status }
	} catch (error) {
		await console.log(`Error setting chat: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

// IPC handler to customize user graph
ipcMain.handle("customize-graph", async (_event, { newGraphInfo }) => {
	/** @type {{ newGraphInfo: string }} */
	try {
		let credits = await getCreditsFromKeytar() // Get current pro credits
		const pricing = await getPricingFromKeytar() // Get pricing plan

		// API call to customize graph
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/customize-graph`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					information: newGraphInfo,
					credits: credits
				})
			}
		)

		if (!response.ok) {
			return {
				message: "Error customizing graph",
				error: `Server responded with status ${response.status}`,
				status: response.status
			}
		}

		if (pricing === "free") {
			// Decrement credits if customizing graph in free plan
			credits -= 1
			await setCreditsInKeytar(Math.max(credits, 0))
		}

		return {
			message: "Graph customized successfully",
			status: 200
		}
	} catch (error) {
		await console.log(`Error customizing graph: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

// IPC handler to scrape Reddit profile
ipcMain.handle("scrape-reddit", async (_event, { redditProfileUrl }) => {
	/** @type {{ redditProfileUrl: string }} */
	try {
		// API call to scrape Reddit profile
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/scrape-reddit`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ url: redditProfileUrl })
			}
		)

		if (!response.ok) {
			return { message: "Error scraping Reddit profile", status: 500 }
		}

		const { topics } = await response.json() // Extract topics from response

		return {
			message: "Reddit profile scraped successfully",
			topics,
			status: 200
		}
	} catch (error) {
		await console.log(`Error scraping Reddit profile: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

// IPC handler to scrape Twitter profile
ipcMain.handle("scrape-twitter", async (_event, { twitterProfileUrl }) => {
	/** @type {{ twitterProfileUrl: string }} */
	try {
		// API call to scrape Twitter profile
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/scrape-twitter`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ url: twitterProfileUrl })
			}
		)

		if (!response.ok) {
			return { message: "Error scraping Twitter profile", status: 500 }
		}

		const { topics } = await response.json() // Extract topics from response

		return {
			message: "Twitter profile scraped successfully",
			topics,
			status: 200
		}
	} catch (error) {
		await console.log(`Error scraping Twitter profile: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

// IPC handler to get beta user status
ipcMain.handle("get-beta-user-status", async () => {
	try {
		const betaUserStatus = await getBetaUserStatusFromKeytar() // Get beta user status from Keytar
		return betaUserStatus
	} catch (error) {
		await console.log(`Error fetching beta user status: ${error}`)
		return { message: `Error: ${error.message}`, status: 500 }
	}
})

// IPC handler to delete a subgraph from the knowledge graph
ipcMain.handle("delete-subgraph", async (event, { source_name }) => {
	/** @type {{ source_name: string }} */
	try {
		// API call to delete subgraph
		const response = await fetch(
			`${process.env.APP_SERVER_URL}/delete-subgraph`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ source: source_name })
			}
		)

		return response.data // Return response data from API
	} catch (error) {
		console.log(`Error deleting subgraph: ${error}`)
		return { status: "failure", error: error.message }
	}
})

ipcMain.handle("get-ollama-models", async () => {
	try {
		const response = await fetch("http://localhost:11434/api/tags")
		if (!response.ok) {
			throw new Error("Failed to fetch models from Ollama")
		}
		const data = await response.json()
		return data.models.map((model) => model.name)
	} catch (error) {
		console.error("Error fetching Ollama models:", error)
		return [] // Return empty array on failure
	}
})

// IPC handler to check if an API key exists for a provider
ipcMain.handle("check-api-key", async (event, provider) => {
	return await hasApiKey(provider)
})

// IPC handler to set an API key for a provider
ipcMain.handle("set-api-key", async (event, { provider, apiKey }) => {
	return await setApiKey(provider, apiKey)
})

ipcMain.handle("get-stored-providers", async () => {
	try {
		const response = await fetch(
			"http://localhost:5005/get-stored-providers"
		)
		if (!response.ok) {
			throw new Error("Failed to fetch stored providers")
		}
		return await response.json()
	} catch (error) {
		console.error("Error fetching stored providers:", error)
		throw error
	}
})

// IPC handler to delete an API key for a provider
ipcMain.handle("delete-api-key", async (event, provider) => {
	return await deleteApiKey(provider)
})

// Handle file uploads
ipcMain.handle("upload-files", async (event, files) => {
	try {
		const uploadedFiles = []
		for (const file of files) {
			const filePath = path.join(uploadDir, file.name)
			// Check for file name conflicts and skip if exists
			if (!fs.existsSync(filePath)) {
				fs.writeFileSync(filePath, file.data)
				uploadedFiles.push(file.name)
			}
		}
		// Send request to refresh RAG context
		await fetch("http://localhost:5000/refresh-rag-context", {
			method: "POST",
			headers: { "Content-Type": "application/json" }
		})
		return {
			status: 200,
			message: "Files uploaded successfully",
			files: uploadedFiles
		}
	} catch (error) {
		console.error(`Error uploading files: ${error}`)
		return {
			status: 500,
			message: "Error uploading files",
			error: error.message
		}
	}
})

// Handle file deletion
ipcMain.handle("delete-file", async (event, fileName) => {
	try {
		const filePath = path.join(uploadDir, fileName)
		if (fs.existsSync(filePath)) {
			fs.unlinkSync(filePath)
			// Send request to refresh RAG context
			await fetch("http://localhost:5000/refresh-rag-context", {
				method: "POST",
				headers: { "Content-Type": "application/json" }
			})
			return { status: 200, message: "File deleted successfully" }
		}
		return { status: 404, message: "File not found" }
	} catch (error) {
		console.error(`Error deleting file: ${error}`)
		return {
			status: 500,
			message: "Error deleting file",
			error: error.message
		}
	}
})

// Get list of uploaded files
ipcMain.handle('get-uploaded-files', async () => {
    try {
        const files = fs.readdirSync(uploadDir)
        return { status: 200, files }
    } catch (error) {
        console.error(`Error getting uploaded files: ${error}`)
        return { status: 500, message: 'Error getting uploaded files', error: error.message }
    }
})