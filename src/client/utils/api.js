/**
 * @file API utility functions for fetching private data.
 * This module contains functions for making authenticated requests to fetch private data from the server.
 * It utilizes access tokens for authorization.
 */

import { getAccessToken } from "./auth.js"

/**
 * Asynchronously fetches private data from the server.
 *
 * This function retrieves an access token and uses it to make an authenticated request
 * to the `/private` endpoint of the server. It checks if the response is successful
 * and throws an error if the HTTP status code indicates failure.
 *
 * @async
 * @function getPrivateData
 * @returns {Promise<any>} A promise that resolves with the JSON response containing private data.
 * @throws {Error} If the HTTP response status is not ok (i.e., not in the 200-299 range).
 *
 * @example
 * try {
 *   const privateData = await getPrivateData();
 *   console.log('Private data:', privateData);
 * } catch (error) {
 *   console.error('Failed to fetch private data:', error);
 * }
 */
async function getPrivateData() {
	/**
	 * Retrieves the access token.
	 * @type {string | null}
	 */
	const accessToken = getAccessToken()
	/**
	 * Fetches data from the private endpoint with the access token in the Authorization header.
	 * @type {Response}
	 */
	const response = await fetch("http://localhost:3000/private", {
		headers: {
			Authorization: `Bearer ${accessToken}` // Include access token in the Authorization header as a Bearer token.
		}
	})

	/**
	 * Checks if the response was successful.
	 * If the response status code is not in the 200-299 range, it indicates an error.
	 */
	if (!response.ok) {
		throw new Error(`HTTP error! Status: ${response.status}`) // Throw an error with the HTTP status code if the request was not successful.
	}

	/**
	 * Parses the JSON response body.
	 * @returns {Promise<any>} A promise that resolves with the parsed JSON data.
	 */
	return await response.json() // Parse and return the JSON response body.
}

export { getPrivateData }