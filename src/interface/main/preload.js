/**
 * @file Preload script for the Electron renderer process.
 * This script bridges the gap between the secure renderer process and the main process,
 * exposing a limited set of Electron APIs to the window.
 * It uses `contextBridge` to safely expose functions to the `window.electron` object
 * in the renderer, enabling communication with the main process for specific tasks.
 */

const { contextBridge, ipcRenderer } = require("electron")

contextBridge.exposeInMainWorld("electron", {
	/**
	 * Listens for an event from the main process on a specific channel.
	 * Provides a way for the renderer process to react to events emitted by the main process.
	 *
	 * @param {string} channel - The name of the IPC channel to listen on.
	 * @param {Function} callback - The function to be called when an event is received on the channel.
	 *                              The callback function receives `event` and `...args` as parameters,
	 *                              where `...args` are the data sent from the main process.
	 *
	 * @example
	 * // In preload.js
	 * electron.on('app-version', (event, version) => {
	 *   console.log('App version:', version);
	 * });
	 *
	 * // In main.js
	 * mainWindow.webContents.send('app-version', app.getVersion());
	 */
	on: (channel, callback) => {
		ipcRenderer.on(channel, callback)
	},
	/**
	 * Sends a message to the main process on a specific channel.
	 * This is a one-way asynchronous communication from the renderer to the main process.
	 *
	 * @param {string} channel - The name of the IPC channel to send the message to.
	 * @param {...any} args -  Arguments to send with the message. These arguments will be received by the main process.
	 *
	 * @example
	 * // In preload.js
	 * electron.send('open-file-dialog');
	 *
	 * // In main.js
	 * ipcMain.on('open-file-dialog', (event) => { ... });
	 */
	send: (channel, ...args) => {
		ipcRenderer.send(channel, ...args)
	},
	/**
	 * Invokes a method in the main process and waits for the result.
	 * This is a two-way synchronous communication from the renderer to the main process, returning a Promise
	 * that resolves with the value returned by the main process handler.
	 *
	 * @param {string} channel - The name of the IPC channel to invoke.
	 * @param {...any} args - Arguments to send with the invocation. These arguments will be received by the main process handler.
	 * @returns {Promise<any>} A Promise that resolves with the value returned by the main process handler.
	 *
	 * @example
	 * // In preload.js
	 * async function getAppDataPath() {
	 *   const appDataPath = await electron.invoke('get-app-data-path');
	 *   console.log('App data path:', appDataPath);
	 * }
	 *
	 * // In main.js
	 * ipcMain.handle('get-app-data-path', async (event) => {
	 *   return app.getPath('appData');
	 * });
	 */
	invoke: (channel, ...args) => {
		return ipcRenderer.invoke(channel, ...args)
	},
	/**
	 * Invokes the 'get-profile' method in the main process to retrieve user profile information.
	 *
	 * @returns {Promise<any>} A Promise that resolves with the user profile data.
	 */
	getProfile: () => ipcRenderer.invoke("get-profile"),
	/**
	 * Sends a 'log-out' event to the main process to initiate the logout procedure.
	 * This function does not return any value.
	 */
	logOut: () => ipcRenderer.send("log-out"),
	/**
	 * Invokes the 'get-private-data' method in the main process to retrieve private user data.
	 *
	 * @returns {Promise<any>} A Promise that resolves with the private user data.
	 */
	getPrivateData: () => ipcRenderer.invoke("get-private-data"),
	/**
	 * Listens for the 'update-available' event from the main process, indicating a new update is available.
	 *
	 * @param {Function} callback - The function to be called when the 'update-available' event is received.
	 *                              The callback function receives `event` and `info` as parameters,
	 *                              where `info` is the update information sent from the main process.
	 */
	onUpdateAvailable: (callback) =>
		ipcRenderer.on("update-available", callback),
	/**
	 * Listens for the 'update-downloaded' event from the main process, indicating a new update has been downloaded.
	 *
	 * @param {Function} callback - The function to be called when the 'update-downloaded' event is received.
	 *                              The callback function receives `event` and `info` as parameters,
	 *                              where `info` is the update information sent from the main process.
	 */
	onUpdateDownloaded: (callback) =>
		ipcRenderer.on("update-downloaded", callback),

	onUpdateError: (callback) => ipcRenderer.on("update-error", callback),
	/**
	 * Sends a 'restart-app' event to the main process to request application restart and update installation.
	 * This function does not return any value.
	 */
	restartApp: () => ipcRenderer.send("restart-app"),
	/**
	 * Listens for the 'status-updated' event from the main process, typically used to display status messages in the UI.
	 *
	 * @param {Function} callback - The function to be called when the 'status-updated' event is received.
	 *                              The callback function receives `event` and `message` as parameters,
	 *                              where `message` is the status message string sent from the main process.
	 */
	onStatusUpdated: (callback) => ipcRenderer.on("status-updated", callback),
	/**
	 * Listens for the 'flow-updated' event from the main process, used to receive updates about intermediary flow steps.
	 *
	 * @param {Function} callback - The function to be called when the 'flow-updated' event is received.
	 *                              The callback function receives `event` and `message` as parameters,
	 *                              where `message` is the flow update message string from the main process.
	 */
	onFlowUpdated: (callback) => ipcRenderer.on("flow-updated", callback),
	/**
	 * Listens for the 'flow-ended' event from the main process, indicating the end of an intermediary flow.
	 *
	 * @param {Function} callback - The function to be called when the 'flow-ended' event is received.
	 *                              The callback function receives the `event` as a parameter.
	 */
	onFlowEnded: (callback) => ipcRenderer.on("flow-ended", callback),
	/**
	 * Sets up a listener for the 'message-stream' event from the main process.
	 * This is specifically designed to handle streaming messages, likely for chat functionalities.
	 * It first removes all existing listeners for 'message-stream' to ensure only one listener is active.
	 *
	 * @param {Function} callback - The function to be called when a 'message-stream' event is received.
	 *                              The callback function receives `data` as a parameter, which is the data
	 *                              sent from the main process containing streaming message information.
	 *                              It is expected that `data` is an object containing chat message details.
	 *
	 * @example
	 * // In renderer.js
	 * electron.onMessageStream((data) => {
	 *   console.log('Stream data received:', data);
	 *   // Handle the streamed message data, e.g., append to chat UI
	 * });
	 *
	 * // In main.js (sending stream data example)
	 * mainWindow.webContents.send('message-stream', { chatId: 'someChatId', token: 'Streamed token data' });
	 */
	onMessageStream: (callback) => {
		ipcRenderer.removeAllListeners("message-stream") // Ensure only one listener is active to avoid duplicate handling
		ipcRenderer.on("message-stream", (event, data) => {
			callback(data)
		})
	},

	onUpdateProgress: (callback) => {
		// Define the listener function so we can potentially remove it later if needed
		const listener = (event, progressObj) => callback(progressObj)
		ipcRenderer.on("update-progress", listener)

		// Return a cleanup function (optional but good practice)
		// This allows the frontend to remove the listener if the component unmounts
		// Note: Requires careful handling in the frontend useEffect cleanup
		return () => {
			ipcRenderer.removeListener("update-progress", listener)
			console.log("Removed update-progress listener") // For debugging
		}
	}
})