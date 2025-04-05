/**
 * IconGoogleSlides Component - Renders the Google Slides Icon as an SVG.
 *
 * This component is designed to display the Google Slides icon using SVG paths.
 * It uses a collection of predefined path data to accurately represent the Google Slides logo,
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
 * @returns {React.ReactNode} - JSX element representing the Google Slides icon as an SVG.
 *
 * @example
 * // Usage example within a React component:
 * <IconGoogleSlides
 *   width={30}
 *   height={30}
 *   fill="#FBBC05" // Example fill color, can be customized to match Google Slides theme
 *   className="slides-icon"
 *   onClick={() => console.log('Google Slides icon clicked')}
 * />
 */
import React from "react"

const IconGoogleSlides = (props) => (
	<svg
		xmlns="http://www.w3.org/2000/svg" // Specifies the namespace for SVG
		width={25} // Sets the width of the SVG icon to 25 pixels
		height={25} // Sets the height of the SVG icon to 25 pixels
		viewBox="0 0 50 50" // Defines the view box for scaling the icon
		{...props} // Spreads any additional props to the SVG element for customization
	>
		{/* Path for the main body of the Google Slides icon in yellow */}
		<path
			fill="#ffc107" // Fill color set to a shade of yellow, resembling Google Slides' color
			d="M37 45H11a3 3 0 0 1-3-3V6a3 3 0 0 1 3-3h19l10 10v29a3 3 0 0 1-3 3z" // Path data defining the shape of the slide document
		/>
		{/* Path for the folded corner at the top right in light yellow */}
		<path fill="#ffecb3" d="M40 13H30V3z" />
		{/* Path for the shadow effect under the folded corner in orange */}
		<path fill="#ffa000" d="m30 13 10 10V13z" />
		{/* Path for the white rectangle representing the slide content area */}
		<path
			fill="#fff" // Fill color set to white, for the content area of the slide
			d="M30 22H18c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V24c0-1.1-.9-2-2-2zm0 4v8H18v-8h12z" // Path data for the white rectangle and inner area
		/>
	</svg>
)

export default IconGoogleSlides
