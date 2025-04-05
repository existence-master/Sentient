import React from "react"
/**
 * IconGoogleDocs Component - Renders the Google Docs Icon as an SVG.
 *
 * This component generates the Google Docs icon using SVG paths. It is constructed
 * using predefined path data to accurately represent the Google Docs logo.
 * The component is built as a functional React component, designed to be
 * scalable and maintain visual consistency across different sizes and contexts.
 *
 * It accepts and applies any props passed to it directly to the root SVG element.
 * This enables users to customize the icon's appearance and behavior through
 * standard SVG attributes such as style, className, and event handling.
 *
 * @param {SVGProps<SVGSVGElement>} props - React props to be passed to the SVG element.
 *                                         This includes standard SVG attributes and event handlers,
 *                                         allowing for customization of the icon.
 *
 * @returns {React.ReactNode} - JSX element representing the Google Docs icon as an SVG.
 *
 * @example
 * // Usage example within a React component:
 * <IconGoogleDocs
 *   width={30}
 *   height={30}
 *   fill="#4285F4"
 *   className="docs-icon"
 *   onClick={() => console.log('Google Docs icon clicked')}
 * />
 */
const IconGoogleDocs = (props) => (
	<svg
		xmlns="http://www.w3.org/2000/svg" // Specifies the namespace for SVG
		width={25} // Sets the width of the SVG icon to 25 pixels
		height={25} // Sets the height of the SVG icon to 25 pixels
		viewBox="0 0 50 50" // Defines the view box, allowing the icon to scale responsively
		{...props} // Spreads any props passed to the component onto the SVG element
	>
		{/* Path element for the main body of the Google Docs icon in blue */}
		<path
			fill="#2196f3" // Fill color set to a shade of blue, resembling Google Docs' color
			d="M37 45H11a3 3 0 0 1-3-3V6a3 3 0 0 1 3-3h19l10 10v29a3 3 0 0 1-3 3z" // Path data defining the shape
		/>
		{/* Path element for the folded corner at the top right in light blue */}
		<path fill="#bbdefb" d="M40 13H30V3z" />
		{/* Path element for the shadow effect under the folded corner in dark blue */}
		<path fill="#1565c0" d="m30 13 10 10V13z" />
		{/* Path elements for the document lines in very light blue, representing text lines */}
		<path
			fill="#e3f2fd" // Fill color set to a very light blue, for the lines inside the document
			d="M15 23h18v2H15zm0 4h18v2H15zm0 4h18v2H15zm0 4h10v2H15z" // Path data for four horizontal lines
		/>
	</svg>
)

export default IconGoogleDocs
