import { app } from "electron"
import path from "path"
import https from "https"
import fs from "fs"
import sudo from "sudo-prompt"
import { writeToLog } from "../utils/logger.js"
import { getBetaUserStatusFromKeytar } from "../utils/auth.js"

/**
 * Fetches the beta user status asynchronously.
 * This status is used to determine whether to check for pre-release server updates.
 */
const betaUserStatus = await getBetaUserStatusFromKeytar()
/**
 * URL for the GitHub repository's releases API endpoint.
 * Used to fetch release information including version and installer download URL.
 */
const GITHUB_REPO_URL =
	"https://api.github.com/repos/existence-master/Sentient/releases"

/**
 * Directory where the server application will be installed.
 * It's located in the user's AppData/Local/Programs directory, under the Sentient application path.
 */
const SERVER_DIR = path.join(
	app.getPath("home"),
	"AppData",
	"Local",
	"Programs",
	"Sentient",
	"server"
)
/**
 * Path for the temporary server installer executable.
 * Installer is downloaded to the temp directory before execution.
 */
const TEMP_EXE = path.join(app.getPath("temp"), "ServerInstaller.exe")
/**
 * Path to the file that stores the current server version.
 * This file is located within the server installation directory.
 */
const CURRENT_VERSION_FILE = path.join(SERVER_DIR, "version.txt")
/**
 * Path to the NSSM (Non-Sucking Service Manager) executable.
 * NSSM is used to install and manage the server as a Windows service.
 */
const NSSM_PATH = path.join(SERVER_DIR, "nssm.exe")

/**
 * Asynchronously checks for updates to the server application.
 *
 * It fetches the latest release information from GitHub and compares the version with the current installed version.
 *
 * @returns {Promise<object|null>} - Returns an object containing the latest version info and download URL if an update is available, otherwise null.
 *                                   The object has the structure: `{ version: string, downloadUrl: string }`.
 *                                   Returns null if no update is available or in case of an error.
 */
const checkForUpdates = async () => {
	try {
		const latestVersionInfo = await fetchLatestReleaseInfo()
		const currentVersion = await getCurrentVersion()
		return latestVersionInfo.version.replace(/^v/, "") !== currentVersion
			? latestVersionInfo
			: null
	} catch (err) {
		await writeToLog(`Error checking updates: ${err}`)
		return null
	}
}

/**
 * Fetches the latest release information from the GitHub API.
 *
 * Depending on the `betaUserStatus`, it either fetches the latest stable release or the latest pre-release.
 * For beta users, it retrieves a list of pre-releases and selects the most recent one based on published date.
 *
 * @returns {Promise<object>} - Returns a promise that resolves to an object containing the latest version and installer download URL.
 *                              The object has the structure: `{ version: string, downloadUrl: string }`.
 * @throws {Error} - Throws an error if the API request fails, if there are no pre-releases available for beta users,
 *                   or if required assets (version.txt, ServerInstaller.exe) are missing in the release.
 */
const fetchLatestReleaseInfo = () => {
	return new Promise((resolve, reject) => {
		/**
		 * API URL to fetch release information.
		 * If betaUserStatus is true, fetches a list of releases (up to 100) to find the latest prerelease.
		 * If betaUserStatus is false, fetches the 'latest' release, which is the latest stable release.
		 */
		const apiUrl = betaUserStatus
			? `${GITHUB_REPO_URL}?per_page=100`
			: `${GITHUB_REPO_URL}/latest`

		/**
		 * Handles the HTTP request to the GitHub API.
		 * Follows redirects and processes the response to extract release information.
		 *
		 * @param {string} url - The API URL to request.
		 */
		const handleRequest = (url) => {
			https
				.get(
					url,
					{ headers: { "User-Agent": "Electron-App" } },
					async (res) => {
						// Handle HTTP redirects (status codes 3xx)
						if (
							res.statusCode >= 300 &&
							res.statusCode < 400 &&
							res.headers.location
						) {
							handleRequest(res.headers.location) // Redirect to the new location
						} else if (res.statusCode === 200) {
							// Successful response (status code 200)
							let data = ""
							res.on("data", (chunk) => (data += chunk)) // Accumulate response data
							res.on("end", async () => {
								// Response fully received
								try {
									let release
									if (betaUserStatus) {
										// Process for beta users to find the latest pre-release
										const releases = JSON.parse(data)
										const prereleases = releases.filter(
											(r) => r.prerelease // Filter for pre-releases
										)
										if (prereleases.length === 0) {
											// No pre-releases found
											throw new Error(
												"No pre-releases available"
											)
										}
										// Sort pre-releases by published date to get the latest
										release = prereleases.sort(
											(a, b) =>
												new Date(b.published_at) -
												new Date(a.published_at)
										)[0]
									} else {
										// Process for stable users, data is directly the latest release info
										release = JSON.parse(data)
									}

									// Find the 'version.txt' and 'ServerInstaller.exe' assets in the release
									const versionAsset = release.assets.find(
										(asset) => asset.name === "version.txt"
									)
									const installerAsset = release.assets.find(
										(asset) =>
											asset.name.endsWith(
												"ServerInstaller.exe"
											)
									)

									if (!versionAsset || !installerAsset) {
										// Required assets are missing in the release
										throw new Error(
											"Missing required assets in release"
										)
									}

									// Fetch the content of 'version.txt' to get the version number
									const version = await fetchAssetContent(
										versionAsset.browser_download_url
									)

									resolve({
										version: version.trim(), // Resolve with version and installer URL
										downloadUrl:
											installerAsset.browser_download_url
									})
								} catch (err) {
									reject(
										new Error(
											`Error processing release data: ${err.message}`
										)
									)
								}
							})
						} else {
							reject(new Error(`HTTP error ${res.statusCode}`)) // Reject for non-200 status codes
						}
					}
				)
				.on("error", reject) // Handle request errors
		}

		handleRequest(apiUrl)
	})
}

/**
 * Fetches the content of a specific asset file from a given URL.
 * Used to download the 'version.txt' file from a GitHub release asset.
 *
 * @param {string} assetUrl - The URL of the asset to download.
 * @returns {Promise<string>} - Returns a promise that resolves to the content of the asset file as a string.
 * @throws {Error} - Throws an error if the HTTP request fails.
 */
const fetchAssetContent = (assetUrl) => {
	return new Promise((resolve, reject) => {
		/**
		 * Handles the HTTP request to download the asset content.
		 * Follows redirects and resolves with the content on successful download.
		 *
		 * @param {string} url - The URL of the asset to download.
		 */
		const handleRequest = (url) => {
			https
				.get(
					url,
					{ headers: { "User-Agent": "Electron-App" } },
					(response) => {
						// Handle HTTP redirects
						if (
							response.statusCode >= 300 &&
							response.statusCode < 400 &&
							response.headers.location
						) {
							writeToLog(
								`Redirecting to: ${response.headers.location}`
							)
							handleRequest(response.headers.location) // Redirect to new location
						} else if (response.statusCode === 200) {
							// Successful response
							let content = ""
							response.on("data", (chunk) => (content += chunk)) // Accumulate content
							response.on("end", () => resolve(content)) // Resolve with the content
						} else {
							reject(
								new Error(
									`Failed to fetch asset. HTTP status: ${response.statusCode}`
								)
							) // Reject for non-200 status
						}
					}
				)
				.on("error", reject) // Handle request errors
		}
		handleRequest(assetUrl)
	})
}

/**
 * Gets the currently installed server version from the version file.
 *
 * @returns {Promise<string>} - Returns a promise that resolves to the current version string.
 *                             Returns "0.0.0" if the version file does not exist or in case of an error.
 */
const getCurrentVersion = async () => {
	try {
		return fs.existsSync(CURRENT_VERSION_FILE)
			? fs.readFileSync(CURRENT_VERSION_FILE, "utf-8").trim() // Read version from file if exists
			: "0.0.0" // Default version if file not found
	} catch (err) {
		await writeToLog(`Error reading current version: ${err}`)
		return "0.0.0" // Default version in case of error
	}
}

/**
 * Updates the server application to the latest version.
 *
 * Downloads the new server installer, stops the current SentientService, runs the installer,
 * reinstalls the SentientService, and starts the updated service.
 *
 * @param {string} downloadUrl - The download URL for the new server installer.
 * @throws {Error} - Throws an error if any step in the update process fails.
 */
const updateServer = async (downloadUrl) => {
	try {
		const exePath = await downloadFile(downloadUrl, TEMP_EXE) // Download the installer
		writeToLog("Downloaded installer.")
		await stopSentientService() // Stop the SentientService
		writeToLog("Stopped SentientService.")
		await runInstaller(exePath) // Run the installer
		writeToLog("Ran installer.")
		await reinstallSentientService() // Reinstall the SentientService
		writeToLog("Reinstalled SentientService.")
		await startSentientService() // Start the SentientService
		writeToLog("Started SentientService.")
	} catch (err) {
		await writeToLog(`Error updating server: ${err}`)
		throw err // Re-throw error to be handled by the caller
	}
}

/**
 * Stops the Windows service 'SentientService' using NSSM.
 *
 * @returns {Promise<void>} - Returns a promise that resolves when the service is successfully stopped.
 * @throws {Error} - Throws an error if stopping the service fails.
 */
const stopSentientService = () => {
	return new Promise((resolve, reject) => {
		sudo.exec(
			`"${NSSM_PATH}" stop SentientService`, // Command to stop the service using NSSM
			{ name: "Sentient" }, // Application name for sudo prompt
			(error, stdout, stderr) => {
				if (error)
					return reject(
						new Error(
							`Failed to stop service: ${stderr || error.message}` // Reject promise on error
						)
					)
				writeToLog("SentientService stopped.")
				resolve() // Resolve promise on success
			}
		)
	})
}

/**
 * Reinstalls the 'SentientService' Windows service using NSSM.
 *
 * It first removes the existing service and then installs it again with the updated server executable path.
 * It also sets the service to auto-start and adds a description.
 *
 * @returns {Promise<void>} - Returns a promise that resolves when the service is successfully reinstalled.
 * @throws {Error} - Throws an error if any command in the reinstallation process fails.
 */
const reinstallSentientService = () => {
	return new Promise((resolve, reject) => {
		/**
		 * Array of NSSM commands to execute for reinstallation.
		 * 1. Remove the existing service (with confirmation)
		 * 2. Install the service, pointing to the updated server executable
		 * 3. Set the service to auto-start on system boot
		 * 4. Set a description for the service
		 */
		const commands = [
			`"${NSSM_PATH}" remove SentientService confirm`,
			`"${NSSM_PATH}" install SentientService "${path.join(
				SERVER_DIR,
				"app.exe" // Path to the server application executable
			)}"`,
			`"${NSSM_PATH}" set SentientService Start SERVICE_AUTO_START`,
			`"${NSSM_PATH}" set SentientService Description "Handles background tasks for the Sentient application."`
		]

		/**
		 * Recursively runs commands from the commands array.
		 * Executes each command in sequence and resolves when all commands are successfully executed.
		 *
		 * @param {number} [index=0] - The index of the command to execute in the commands array.
		 */
		const runCommands = (index = 0) => {
			if (index >= commands.length) return resolve() // Resolve when all commands are executed

			sudo.exec(
				commands[index], // Execute the command at the current index
				{ name: "Sentient" }, // Application name for sudo prompt
				(error, stdout, stderr) => {
					if (error)
						return reject(
							new Error(
								`Failed command ${commands[index]}: ${
									stderr || error.message // Reject promise if command fails
								}`
							)
						)
					writeToLog(`Executed: ${commands[index]}`)
					runCommands(index + 1) // Run the next command
				}
			)
		}

		runCommands() // Start executing commands from index 0
	})
}

/**
 * Starts the 'SentientService' Windows service using NSSM.
 *
 * @returns {Promise<void>} - Returns a promise that resolves when the service is successfully started.
 * @throws {Error} - Throws an error if starting the service fails.
 */
const startSentientService = () => {
	return new Promise((resolve, reject) => {
		sudo.exec(
			`"${NSSM_PATH}" start SentientService`, // Command to start the service using NSSM
			{ name: "Sentient" }, // Application name for sudo prompt
			(error, stdout, stderr) => {
				if (error)
					return reject(
						new Error(
							`Failed to start service: ${
								stderr || error.message // Reject promise if starting fails
							}`
						)
					)
				writeToLog("SentientService started.")
				resolve() // Resolve promise on success
			}
		)
	})
}

/**
 * Downloads a file from a given URL to a specified destination path.
 *
 * @param {string} fileUrl - The URL of the file to download.
 * @param {string} destination - The local file path where the downloaded file should be saved.
 * @returns {Promise<string>} - Returns a promise that resolves to the destination path of the downloaded file upon successful download.
 * @throws {Error} - Throws an error if the download fails or if there is an issue writing the file.
 */
const downloadFile = (fileUrl, destination) => {
	return new Promise((resolve, reject) => {
		/**
		 * Handles the HTTP request to download the file.
		 * Follows redirects and pipes the response to a file stream to save the downloaded file.
		 *
		 * @param {string} url - The URL of the file to download.
		 */
		const handleRequest = (url) => {
			https
				.get(url, (response) => {
					// Handle HTTP redirects
					if (
						response.statusCode >= 300 &&
						response.statusCode < 400 &&
						response.headers.location
					) {
						writeToLog(
							`Redirecting to: ${response.headers.location}`
						)
						handleRequest(response.headers.location) // Redirect to new location
					} else if (response.statusCode === 200) {
						// Successful response
						const file = fs.createWriteStream(destination) // Create write stream for destination file
						response.pipe(file) // Pipe the response stream to the file stream
						file.on("finish", () => {
							// File download finished
							file.close((err) => {
								if (err)
									reject(err) // Reject if closing file stream fails
								else resolve(destination) // Resolve with destination path on success
							})
						})
					} else {
						reject(
							new Error(
								`Failed to download file. HTTP status: ${response.statusCode}` // Reject for non-200 status
							)
						)
					}
				})
				.on("error", (err) => {
					fs.unlink(destination, () => reject(err)) // Delete partially downloaded file and reject on error
				})
		}
		handleRequest(fileUrl) // Start the download process
	})
}

/**
 * Runs the server installer executable.
 *
 * @param {string} exePath - The path to the server installer executable.
 * @returns {Promise<void>} - Returns a promise that resolves when the installer process completes.
 * @throws {Error} - Throws an error if the installer fails to run.
 */
const runInstaller = (exePath) => {
	return new Promise((resolve, reject) => {
		const options = { name: "Sentient", env: { PATH: process.env.PATH } } // Options for sudo execution, including PATH environment variable

		sudo.exec(`"${exePath}"`, options, (error, stdout, stderr) => {
			// Execute the installer with sudo privileges
			if (error) reject(new Error(`Installer failed: ${error.message}`)) // Reject promise if installer execution fails
			if (stderr) console.error(`Installer error: ${stderr}`) // Log installer errors to console
			if (stdout) console.log(`Installer: ${stdout}`) // Log installer output to console
			resolve() // Resolve promise when installer completes (regardless of internal installer success, assuming process started)
		})
	})
}

/**
 * Sets up the FastAPI server application.
 *
 * Checks if the server directory exists, checks for updates, and updates the server if a new version is available.
 *
 * @returns {Promise<{success: boolean, updateNeeded?: boolean, error?: string}>} - Returns a promise that resolves to an object
 *           indicating the success of the server setup and whether an update was needed.
 *           The object has the structure: `{ success: boolean, updateNeeded?: boolean, error?: string }`.
 */
const setupServer = async () => {
	try {
		writeToLog("Setting up FastAPI server...")

		if (!fs.existsSync(SERVER_DIR)) {
			// Create server directory if it does not exist
			fs.mkdirSync(SERVER_DIR, { recursive: true })
		}

		const updateInfo = await checkForUpdates() // Check for server updates

		writeToLog(`Update info: ${JSON.stringify(updateInfo)}`)

		if (updateInfo) {
			// Update server if a new version is available
			await updateServer(updateInfo.downloadUrl)
		}

		return { success: true, updateNeeded: !!updateInfo } // Return success status and updateNeeded flag
	} catch (err) {
		await writeToLog(`Error setting up FastAPI server: ${err}`)
		return { success: false, error: err.message } // Return failure status and error message
	}
}

export { setupServer }