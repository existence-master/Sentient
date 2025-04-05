/**
 * @file Script to start the authentication server as a child process within the Electron application.
 * This script is responsible for spawning the `auth.exe` server executable,
 * managing its process, and logging its output and errors.
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
 * Path to the authentication server executable (auth.exe).
 * This executable is responsible for handling authentication-related tasks for the application.
 * @constant {string}
 */
const SERVER_EXE_PATH = path.join(SERVER_DIR, "auth.exe")

/**
 * Starts the authentication server as a child process.
 *
 * This function spawns a new process running the `auth.exe` executable.
 * It configures the working directory and environment for the server process.
 * The function also sets up listeners for stdout, stderr, and close events
 * to log the server's output, errors, and exit status to the console.
 *
 * @function startAuthServer
 * @returns {void}
 */
const startAuthServer = () => {
	/**
	 * Spawns a new process for the authentication server executable.
	 * @type {ChildProcess}
	 */
	const appProcess = spawn(SERVER_EXE_PATH, {
		cwd: SERVER_DIR, // Sets the current working directory for the spawned process to the server directory.
		env: { ...process.env } // Inherits the environment variables from the current process.
	})

	/**
	 * Listener for data on the standard output stream of the authentication server process.
	 * Logs any data received on stdout to the console, prefixed with "Auth Server:".
	 */
	appProcess.stdout.on("data", (data) => {
		console.log(`Auth Server: ${data}`)
	})

	/**
	 * Listener for data on the standard error stream of the authentication server process.
	 * Logs any data received on stderr to the console as an error, prefixed with "Auth Server error:".
	 */
	appProcess.stderr.on("data", (data) => {
		console.error(`Auth Server error: ${data}`)
	})

	/**
	 * Listener for the 'close' event of the authentication server process.
	 * This event is emitted when the server process has finished and exited.
	 * Logs the exit code of the server process to the console.
	 */
	appProcess.on("close", (code) => {
		console.log(`Auth Server exited with code ${code}`)
	})
}

export { startAuthServer }