import { BrowserWindow } from "electron" // Importing BrowserWindow class from electron module
import {
	getAuthenticationURL, // Importing function to get authentication URL
	loadTokens, // Importing function to load tokens after authentication
	getLogOutUrl, // Importing function to get logout URL
	logout // Importing function to perform logout
} from "../utils/auth.js"
import { checkValidity } from "./index.js" // Importing function to check token validity

// Variable to hold the authentication window instance - win: BrowserWindow | null
let win = null

/**
 * Creates and displays a BrowserWindow for user authentication.
 *
 * This function is responsible for setting up and displaying an Electron BrowserWindow
 * that loads the authentication URL. It also handles the OAuth2 callback, token loading,
 * and window lifecycle management for the authentication process.
 *
 * @async
 * @function createAuthWindow
 * @returns {void}
 */
export function createAuthWindow() {
	destroyAuthWin() // Ensure no existing auth window

	win = new BrowserWindow({
		width: 2000, // Width of the authentication window
		height: 1500, // Height of the authentication window
		autoHideMenuBar: true, // Hides the menu bar
		webPreferences: {
			nodeIntegration: false, // Node.js integration disabled for security
			enableRemoteModule: false, // Remote module disabled for security
			devTools: false // DevTools disabled in production
		},
		backgroundColor: "#000000" // Background color of the window set to black
	})

	win.loadURL(getAuthenticationURL()) // Load authentication URL in the window

	const {
		session: { webRequest }
	} = win.webContents // Destructure webRequest from webContents session

	const filter = {
		urls: ["http://localhost/callback*"] // Filter for web requests to callback URL
	}

	// Listener for onBeforeRequest event for URLs matching the filter
	webRequest.onBeforeRequest(filter, async ({ url }) => {
		await loadTokens(url) // Load tokens from the callback URL
		await checkValidity() // Check validity of the loaded tokens
		return destroyAuthWin() // Destroy authentication window after token load and validity check
	})

	// Event listener for 'authenticated' event on the window (not standard Electron API, likely custom event)
	win.on("authenticated", () => {
		destroyAuthWin() // Destroy authentication window after successful authentication
	})

	// Event listener for 'closed' event on the window
	win.on("closed", () => {
		win = null // Dereference win variable when window is closed
	})
}

/**
 * Destroys the authentication window if it exists.
 *
 * This function checks if an authentication window instance exists and, if so, closes and dereferences it
 * to ensure proper resource cleanup. It's used to prevent multiple authentication windows from being open.
 *
 * @function destroyAuthWin
 * @returns {void}
 */
function destroyAuthWin() {
	if (!win) return // If win is null (no window), return early
	win.close() // Close the authentication window
	win = null // Dereference win variable
}

/**
 * Creates and displays a BrowserWindow for user logout.
 *
 * This function sets up a BrowserWindow to load the logout URL, effectively logging the user out
 * of the authentication provider. It automatically closes after a short delay, after attempting to logout.
 *
 * @async
 * @function createLogoutWindow
 * @returns {void}
 */
export function createLogoutWindow() {
	let logoutWindow = new BrowserWindow({
		width: 400, // Width of the logout window
		height: 300, // Height of the logout window
		show: false, // Initially hide the window
		backgroundColor: "#ffffff", // Background color of the window set to white
		autoHideMenuBar: true, // Hides the menu bar
		webPreferences: {
			nodeIntegration: false, // Node.js integration disabled for security
			contextIsolation: true, // Context isolation enabled for security
			enableRemoteModule: false, // Remote module disabled for security
			devTools: false // DevTools disabled in production
		}
	})

	logoutWindow.loadURL(getLogOutUrl()) // Load logout URL in the window

	// Event listener for 'ready-to-show' event, emitted when the window is ready to be displayed
	logoutWindow.once("ready-to-show", () => {
		logoutWindow.show() // Show the logout window once it is ready

		// Set timeout to attempt logout and then close the logout window
		setTimeout(async () => {
			try {
				await logout() // Attempt to perform logout
			} catch (error) {
				await console.log(`Error during logout: ${error}`) // Log error if logout fails
			}
		}, 5000) // 5 seconds delay before attempting logout and closing
	})

	// Event listener for 'closed' event on the logout window
	logoutWindow.on("closed", () => {
		logoutWindow = null // Dereference logoutWindow variable when window is closed
	})
}
