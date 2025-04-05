import React from "react"
/**
 * IconGoogleSheets Component - Renders the Google Sheets Icon as an SVG.
 *
 * This component is designed to display the Google Sheets icon using SVG paths.
 * It utilizes predefined path data to accurately represent the Google Sheets logo,
 * ensuring a visually consistent and scalable icon for use in the application.
 * The component is implemented as a functional React component, making it efficient
 * and suitable for modern React applications.
 *
 * The component accepts and applies any props passed to it to the root SVG element.
 * This allows for easy customization of the icon's appearance and behavior,
 * such as size, color, and event handling, via standard SVG attributes and React props.
 *
 * @param {SVGProps<SVGSVGElement>} props - React props to be passed to the SVG element.
 *                                         This includes standard SVG attributes and React event handlers for customization.
 *
 * @returns {React.ReactNode} - JSX element representing the Google Sheets icon as an SVG.
 *
 * @example
 * // Usage example within a React component:
 * <IconGoogleSheets
 *   width={30}
 *   height={30}
 *   fill="#388E3C" // Example fill color, can be customized
 *   className="sheets-icon"
 *   onClick={() => console.log('Google Sheets icon clicked')}
 * />
 */
const IconGoogleSheets = (props) => (
	<svg
		xmlns="http://www.w3.org/2000/svg" // Specifies the namespace for SVG
		width={25} // Sets the width of the SVG icon to 25 pixels
		height={25} // Sets the height of the SVG icon to 25 pixels
		viewBox="0 0 50 50" // Defines the view box for scaling the icon
		{...props} // Spreads any additional props to the SVG element for customization
	>
		{/* Path for the main body of the Google Sheets icon in green */}
		<path
			fill="#43a047" // Fill color set to a shade of green, resembling Google Sheets' color
			d="M37 45H11a3 3 0 0 1-3-3V6a3 3 0 0 1 3-3h19l10 10v29a3 3 0 0 1-3 3z" // Path data defining the shape of the document
		/>
		{/* Path for the folded corner at the top right in light green */}
		<path fill="#c8e6c9" d="M40 13H30V3z" />
		{/* Path for the shadow effect under the folded corner in dark green */}
		<path fill="#2e7d32" d="m30 13 10 10V13z" />
		{/* Path elements for the spreadsheet grid lines in very light green */}
		<path
			fill="#e8f5e9" // Fill color set to a very light green, for the grid lines
			d="M31 23H15v14h18V23h-2zm-14 2h4v2h-4v-2zm0 4h4v2h-4v-2zm0 4h4v2h-4v-2zm14 2h-8v-2h8v2zm0-4h-8v-2h8v2zm0-4h-8v-2h8v2z" // Path data for the spreadsheet grid
		/>
	</svg>
)

export default IconGoogleSheets
