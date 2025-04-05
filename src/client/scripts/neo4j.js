/**
 * @file Script to manage the Neo4j server lifecycle (start and stop) using `sudo-prompt` for elevated privileges.
 * This script defines functions to start and stop the Neo4j database server, which requires administrator/sudo rights to execute Neo4j commands.
 * It uses `sudo-prompt` to request necessary permissions for these operations.
 */

import path from "path"
import sudo from "sudo-prompt"
import { app } from "electron"

/**
 * Asynchronously starts the Neo4j server.
 *
 * This function attempts to start the Neo4j server by executing the `neo4j start` command with elevated privileges using `sudo-prompt`.
 * It constructs the necessary paths to the Neo4j executable and sets up environment variables required for Neo4j to run correctly.
 *
 * @async
 * @returns {Promise<string>} A promise that resolves with a success message when the Neo4j server is started.
 * @throws {Error} If starting the Neo4j server fails, the promise is rejected with an error message.
 */
export const startNeo4jServer = async () => {
	try {
		// Construct the path to the Neo4j 'bin' directory within the application's installation.
		const neo4jBinPath = path.join(
			app.getPath("home"),
			"AppData",
			"Local",
			"Programs",
			"Sentient",
			"neo4j",
			"bin"
		)
		// Construct the full path to the Neo4j command executable.
		const neo4jCommand = path.join(neo4jBinPath, "neo4j")

		// Define options for the sudo-prompt execution.
		const options = {
			name: "Sentient", // Application name for the sudo prompt dialog.
			env: {
				NEO4J_HOME: path.join(
					// Set NEO4J_HOME environment variable, crucial for Neo4j to find its configuration and data.
					app.getPath("home"),
					"AppData",
					"Local",
					"Programs",
					"Sentient",
					"neo4j"
				),
				PATH: process.env.PATH // Inherit the system's PATH environment variable to ensure Neo4j can find dependencies.
			}
		}

		// Execute the 'neo4j start' command using sudo-prompt to request elevated privileges.
		await sudo.exec(neo4jCommand + " start", options)

		return "Neo4j server started successfully" // Return success message if the command executes without error.
	} catch (error) {
		await console.log(`Error starting Neo4j server: ${error}`) // Log error to console for debugging.
		throw new Error(`Failed to start Neo4j: ${error.message}`) // Throw an error to indicate Neo4j server start failure.
	}
}

/**
 * Asynchronously stops the Neo4j server.
 *
 * This function attempts to stop the Neo4j server by executing the `neo4j stop` command with elevated privileges using `sudo-prompt`.
 * It constructs the necessary paths to the Neo4j executable and sets up environment variables similar to the start function.
 *
 * @async
 * @returns {Promise<string>} A promise that resolves with a success message when the Neo4j server is stopped.
 * @throws {Error} If stopping the Neo4j server fails, the promise is rejected with an error message.
 */
export const stopNeo4jServer = async () => {
	try {
		// Construct the path to the Neo4j 'bin' directory within the application's installation.
		const neo4jBinPath = path.join(
			app.getPath("home"),
			"AppData",
			"Local",
			"Programs",
			"Sentient",
			"neo4j",
			"bin"
		)
		// Construct the full path to the Neo4j command executable.
		const neo4jCommand = path.join(neo4jBinPath, "neo4j")

		// Define options for the sudo-prompt execution, similar to startNeo4jServer.
		const options = {
			name: "Sentient", // Application name for the sudo prompt dialog.
			env: {
				NEO4J_HOME: path.join(
					// Set NEO4J_HOME environment variable, crucial for Neo4j commands.
					app.getPath("home"),
					"AppData",
					"Local",
					"Programs",
					"Sentient",
					"neo4j"
				),
				PATH: process.env.PATH // Inherit the system's PATH environment variable.
			}
		}

		// Execute the 'neo4j stop' command using sudo-prompt to request elevated privileges.
		await sudo.exec(neo4jCommand + " stop", options)

		return "Neo4j server stopped successfully" // Return success message if the command executes without error.
	} catch (error) {
		await console.log(`Error stopping Neo4J server: ${error}`) // Log error to console for debugging.
		throw new Error(`Failed to stop Neo4j: ${error.message}`) // Throw an error to indicate Neo4j server stop failure.
	}
}
