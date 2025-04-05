/**
 * @file Script to start the utility server as a child process within the Electron application.
 * This script is responsible for launching the `utils.exe` server, which provides utility functions for the application.
 * It manages the server process and logs its output and errors to the console.
 */

import { app } from "electron"
import path from "path"
import { spawn } from "child_process"

/**
 * Directory where the server executables are located within the packaged application.
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
 * Path to the utility server executable (utils.exe).
 * This server provides utility functionalities for the application, such as data processing or background tasks.
 * @constant {string}
 */
const SERVER_EXE_PATH = path.join(SERVER_DIR, "utils.exe")

/**
 * Starts the utility server as a child process.
 *
 * This function spawns a new process running the `utils.exe` executable.
 * It sets the current working directory for the spawned process to the server directory
 * and inherits the environment variables from the current process.
 * Listeners are attached to the process's stdout, stderr, and close events to log output,
 * errors, and exit codes to the console.
 *
 * @function startUtilsServer
 * @returns {void}
 */
const startUtilsServer = () => {
	/**
	 * Spawns a new process for the utility server executable.
	 * @type {ChildProcess}
	 */
	const utilsProcess = spawn(SERVER_EXE_PATH, {
		cwd: SERVER_DIR, // Sets the current working directory for the spawned process to the server directory.
		env: { ...process.env } // Inherits the environment variables from the current process.
	})

	/**
	 * Listener for data on the standard output stream of the utility server process.
	 * Logs any data received on stdout to the console, prefixed with "FastAPI:".
	 */
	utilsProcess.stdout.on("data", (data) => {
		console.log(`FastAPI: ${data}`)
	})

	/**
	 * Listener for data on the standard error stream of the utility server process.
	 * Logs any data received on stderr to the console as an error, prefixed with "FastAPI error:".
	 */
	utilsProcess.stderr.on("data", (data) => {
		console.error(`FastAPI error: ${data}`)
	})

	/**
	 * Listener for the 'close' event of the utility server process.
	 * This event is emitted when the server process has finished and exited.
	 * Logs the exit code of the server process to the console, prefixed with "FastAPI exited with code".
	 */
	utilsProcess.on("close", (code) => {
		console.log(`FastAPI exited with code ${code}`)
	})
}

export { startUtilsServer }