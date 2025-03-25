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
	appOutDir = path.join(__dirname, "../out")
} else {
	basePath = __dirname
	dotenvPath = path.resolve(__dirname, "../.env")
	chatsDbPath = path.resolve(__dirname, "../../chatsDb.json")
	userProfileDbPath = path.resolve(__dirname, "../../userProfileDb.json")
	appOutDir = path.join(__dirname, "../out")
}

let ws = new WebSocket("ws://localhost:5000/ws") // Replace with your FastAPI server URL if different

ws.onopen = () => {
	console.log("WebSocket connection opened with FastAPI")
	// Optionally send a message to the server upon connection
	// ws.send('Electron client connected');
}

ws.onmessage = (event) => {
	const messageData = JSON.parse(event.data)
	console.log("WebSocket message received:", messageData)

	if (messageData.type === "task_completed") {
		const { task_id, description, result } = messageData
		new Notification({
			title: "Task Completed!",
			body: `Task "${description}" (ID: ${task_id}) completed successfully.\nResult: ${result.substring(0, 100)}...` // Limit result preview for notification
		}).show()
	} else if (messageData.type === "task_error") {
		const { task_id, description, error } = messageData
		new Notification({
			title: "Task Error!",
			body: `Task "${description}" (ID: ${task_id}) encountered an error.\nError: ${error}`
		}).show()
	}
}

ws.onclose = () => {
	console.log("WebSocket connection closed")
}

ws.onerror = (error) => {
	console.error("WebSocket error:", error)
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
			"http://127.0.0.1:5000/authenticate-google" // URL to check Google auth status
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

ipcMain.handle("fetch-chat-history", async () => {
	try {
		const response = await fetch(`http://localhost:5000/get-history`, {
			method: "GET"
		})
		if (!response.ok) throw new Error("Failed to fetch chat history")
		const data = await response.json()
		return { messages: data.messages, status: 200 }
	} catch (error) {
		console.log(`Error fetching chat history: ${error}`)
		return { message: "Error fetching chat history", status: 500 }
	}
})

ipcMain.handle("send-message", async (_event, { input }) => {
	try {
		const pricing = (await getPricingFromKeytar()) || "free"
		const credits = await getCreditsFromKeytar()
		console.log(
			"PAYLOAD:",
			JSON.stringify({
				input,
				pricing,
				credits,
				chat_id: "single_chat"
			})
		)
		const response = await fetch(`${process.env.APP_SERVER_URL}/chat`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				input,
				pricing,
				credits,
				chat_id: "single_chat"
			})
		})
		if (!response.ok) throw new Error("Error sending message")

		const readable = response.body
		const decoder = new TextDecoder("utf-8")
		let buffer = ""

		for await (const chunk of readable) {
			buffer += decoder.decode(chunk, { stream: true })
			const splitMessages = buffer.split("\n")
			buffer = splitMessages.pop()

			for (const msg of splitMessages) {
				try {
					const parsedMessage = JSON.parse(msg)
					if (parsedMessage.type === "assistantStream") {
						mainWindow.webContents.send("message-stream", {
							messageId: parsedMessage.messageId || Date.now(),
							token: parsedMessage.token
						})
						if (
							parsedMessage.done &&
							parsedMessage.proUsed &&
							pricing === "free"
						) {
							let credits = await getCreditsFromKeytar()
							credits -= 1
							await setCreditsInKeytar(Math.max(credits, 0))
						}
					}
				} catch (parseError) {
					console.log(
						`Error parsing streamed message: ${parseError}: ${msg}`
					)
				}
			}
		}

		if (buffer.trim()) {
			try {
				const parsedMessage = JSON.parse(buffer)
				if (parsedMessage.type === "assistantMessage") {
					mainWindow.webContents.send("message-stream", {
						messageId: Date.now(),
						token: parsedMessage.message
					})
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

ipcMain.handle("fetch-tasks", async () => {
	try {
		const response = await fetch("http://localhost:5000/fetch-tasks")
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`)
		}
		return await response.json()
	} catch (error) {
		console.error("Error fetching tasks:", error)
		return { error: error.message }
	}
})

ipcMain.handle("add-task", async (event, taskData) => {
	try {
		const response = await fetch("http://localhost:5000/add-task", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(taskData)
		})
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`)
		}
		return await response.json()
	} catch (error) {
		console.error("Error adding task:", error)
		return { error: error.message }
	}
})

ipcMain.handle(
	"update-task",
	async (event, { taskId, description, priority }) => {
		try {
			const response = await fetch(
				`http://localhost:5000/update-task`, // Changed to POST endpoint for update
				{
					method: "POST", // Changed method to POST
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ task_id: taskId, description, priority }) // Included taskId in the body
				}
			)
			if (!response.ok) {
				throw new Error(`HTTP error! status: ${response.status}`)
			}
			return await response.json()
		} catch (error) {
			console.error("Error updating task:", error)
			return { error: error.message }
		}
	}
)

ipcMain.handle("delete-task", async (event, taskId) => {
	try {
		const response = await fetch(`http://localhost:5000/delete-task`, {
			// Changed to POST endpoint for delete
			method: "POST", // Changed method to POST
			headers: { "Content-Type": "application/json" }, // Added Content-Type header as we are sending JSON body
			body: JSON.stringify({ task_id: taskId }) // Included taskId in the body
		})
		if (!response.ok) {
			throw new Error(`HTTP error! status: ${response.status}`)
		}
		return await response.json()
	} catch (error) {
		console.error("Error deleting task:", error)
		return { error: error.message }
	}
})