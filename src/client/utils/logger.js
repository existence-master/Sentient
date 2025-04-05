/**
 * @file Logging utility module for writing messages to a log file.
 * This module defines a function `writeToLog` that appends timestamped messages to a log file.
 * The log file path is determined based on the operating system to ensure proper file storage locations.
 */

import fs from "fs"
import path from "path"
import { app } from "electron"

/**
 * Path to the log file. This is determined based on the operating system.
 * On Windows, it's in ProgramData/Sentient/logs/node-backend.log.
 * On Linux, it's in userData/logs/node-backend.log within the Electron app's data directory.
 * @type {string}
 * @private
 */
let logFilePath

// Determine the log file path based on the operating system
if (process.platform === "win32") {
	logFilePath = path.join(
		process.env.PROGRAMDATA, // Access the ProgramData directory in Windows
		"Sentient",
		"logs",
		"node-backend.log"
	)
} else if (process.platform === "linux") {
	logFilePath = path.join(
		app.getPath("userData"), // Get the userData path for the Electron app on Linux
		"logs",
		"node-backend.log"
	)
}

/**
 * Asynchronously writes a message to the log file, prepending a timestamp.
 * It creates the log directory and file if they do not exist.
 *
 * @async
 * @function writeToLog
 * @param {string} message - The message to be written to the log file.
 * @returns {Promise<void>} - A Promise that resolves when the message is successfully written to the log file.
 *                            If there is an error during file operations, it logs the error to the console.
 *
 * @example
 * await writeToLog('Application started successfully.');
 * @example
 * await writeToLog('User action: button clicked.');
 */
export const writeToLog = async (message) => {
	/**
	 * Current timestamp in ISO format, used for logging.
	 * @type {string}
	 */
	const timestamp = new Date().toISOString()
	/**
	 * Formatted log message including timestamp and the message content.
	 * @type {string}
	 */
	const logMessage = `${timestamp}: ${message}\n`

	try {
		// Ensure the directory for the log file exists. Create recursively if necessary.
		await fs.promises.mkdir(path.dirname(logFilePath), { recursive: true })
		// Check if the log file exists. If not, create it.
		await fs.promises
			.access(logFilePath, fs.constants.F_OK) // Check file access (existence)
			.catch(async () => {
				await fs.promises.writeFile(logFilePath, "") // If file doesn't exist, create it (empty)
			})

		// Append the log message to the log file.
		await fs.promises.appendFile(logFilePath, logMessage)
	} catch (error) {
		console.error("Error writing to log file:", error) // Log any errors during file writing process to console.
	}
}