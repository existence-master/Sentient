/**
 * @file Script to manage the Sentient application server service using NSSM (Non-Sucking Service Manager) on Windows.
 * This script defines functions to start and stop the application server, which is managed as a Windows service.
 * It utilizes NSSM to check the service status and control its lifecycle.
 */

import { app } from "electron"
import path from "path"
import fetch from "node-fetch"
import { exec } from "child_process"

/**
 * Directory where the server and NSSM are located within the packaged application.
 * This path is specific to the packaged application structure on Windows, assuming installation in the user's AppData directory.
 * @constant {string}
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
 * Path to the NSSM executable (nssm.exe).
 * NSSM is used to manage the application server as a Windows service.
 * @constant {string}
 */
const NSSM_PATH = path.join(SERVER_DIR, "nssm.exe")

/**
 * Starts the Sentient application server service if it is not already running.
 * This function uses NSSM to check the service status and, if necessary, starts the service.
 * It logs the status and any errors to the console.
 * @function startAppServer
 * @returns {void}
 */
const startAppServer = () => {
	// Execute NSSM command to get the status of the 'SentientService'
	exec(`"${NSSM_PATH}" status SentientService`, (err, stdout, stderr) => {
		if (err) {
			// If there's an error during service status check, log the error
			console.error(`Error checking service: ${stderr}`)
			return
		}

		// Check if the service status output includes 'SERVICE_RUNNING'
		if (stdout.includes("SERVICE_RUNNING")) {
			console.log("SentientService is already running")
		} else {
			// If the service is not running, proceed to start it
			console.log("Starting SentientService...")
			// Execute NSSM command to start the 'SentientService'
			exec(
				`"${NSSM_PATH}" start SentientService`,
				(err, stdout, stderr) => {
					if (err) {
						// If there's an error starting the service, log the error
						console.error(`Failed to start service: ${stderr}`)
					} else {
						// If the service starts successfully, log a success message
						console.log("SentientService started successfully")
					}
				}
			)
		}
	})
}

/**
 * Stops the sub-servers of the application by sending a POST request to a specific endpoint.
 * This is used to gracefully shut down the application's backend servers when the application closes.
 * It attempts to fetch a 'stop-servers-on-app-close' endpoint and logs the result or any errors.
 * @async
 * @function stopAppSubServers
 * @returns {Promise<void>}
 */
const stopAppSubServers = async () => {
	try {
		// Attempt to send a POST request to the server's stop endpoint
		const response = await fetch(
			"http://127.0.0.1:5000/stop-servers-on-app-close",
			{
				method: "POST",
				headers: {
					"Content-Type": "application/json"
				}
			}
		)

		// Check if the response status is not okay (e.g., 404, 500)
		if (!response.ok) {
			// If not okay, log an error message including the response text
			console.error("Failed to stop servers:", await response.text())
			return
		}

		// If the response is okay, parse the JSON body
		await response.json()
		// Log a success message indicating subservers were stopped
		console.log("Subservers stopped successfully")
	} catch (error) {
		// Catch any errors during the fetch operation (e.g., network errors)
		console.log(`Error stopping subservers: ${error}`)
	}
}

export { startAppServer, stopAppSubServers }
