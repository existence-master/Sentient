import React from "react"
/**
 * IconGoogleDrive Component - Renders the Google Drive Icon as an SVG.
 *
 * This component is responsible for rendering the Google Drive icon using SVG paths.
 * It utilizes a collection of predefined path data to accurately and efficiently
 * draw the Google Drive logo. The component is designed as a functional React
 * component, ensuring it's lightweight and performant for UI rendering.
 *
 * The component accepts and applies any additional properties (props) directly
 * to the root SVG element. This allows for easy customization of the icon's
 * attributes such as size, color, and event handlers through standard SVG and
 * React props.
 *
 * @param {SVGProps<SVGSVGElement>} props - React props to be passed to the SVG element.
 *                                         Includes standard SVG attributes and React event handlers for customization.
 *
 * @returns {React.ReactNode} - JSX element representing the Google Drive icon as an SVG.
 *
 * @example
 * // Usage example within a React component:
 * <IconGoogleDrive
 *   width={40}
 *   height={40}
 *   fill="#3f8efc"
 *   className="drive-icon"
 *   onClick={() => console.log('Drive icon clicked')}
 * />
 */
const IconGoogleDrive = (props) => (
	<svg
		xmlns="http://www.w3.org/2000/svg" // Specifies the namespace for SVG
		width={25} // Sets the width of the SVG icon to 25 pixels
		height={25} // Sets the height of the SVG icon to 25 pixels
		viewBox="0 0 50 50" // Defines the view box for scaling the icon
		{...props} // Spreads any additional props to the SVG element for customization
	>
		{/* Path for the blue trapezoid shape at the top of the Drive icon */}
		<path
			fill="#1e88e5" // Blue color fill for the path
			d="M38.59 39c-.535.93-.298 1.68-1.195 2.197A6.004 6.004 0 0 1 34.39 42H13.61a6.015 6.015 0 0 1-3.004-.802C9.708 40.681 9.945 39.93 9.41 39l7.67-9h13.84l7.67 9z"
		/>
		{/* Path for the yellow trapezoid shape in the middle of the Drive icon */}
		<path
			fill="#fbc02d" // Yellow color fill for the path
			d="M27.463 6.999c1.073-.002 2.104-.716 3.001-.198a6.008 6.008 0 0 1 2.197 2.201l10.39 17.996c.537.93.807 1.967.808 3.002.001 1.037-1.267 2.073-1.806 3.001l-11.127-3.005-6.924-11.993 3.461-11.004z"
		/>
		{/* Path for the red trapezoid shape on the right side of the Drive icon */}
		<path
			fill="#e53935" // Red color fill for the path
			d="M43.86 30c0 1.04-.27 2.07-.81 3l-3.67 6.35a6.05 6.05 0 0 1-1.99 1.85L30.92 30h12.94z"
		/>
		{/* Path for the green trapezoid shape on the left side of the Drive icon */}
		<path
			fill="#4caf50" // Green color fill for the path
			d="M5.947 33.001c-.538-.928-1.806-1.964-1.806-3a6.011 6.011 0 0 1 .808-3.004l10.39-17.996a6.01 6.01 0 0 1 2.196-2.2c.897-.519 1.929.195 3.002.197l3.459 11.009-6.922 11.989-11.127 3.005z"
		/>
		{/* Path for the darker blue shading on the left side of the blue trapezoid */}
		<path
			fill="#1565c0" // Darker blue color fill for shading
			d="m17.08 30-6.47 11.2a6.05 6.05 0 0 1-1.99-1.85L4.95 33c-.54-.93-.81-1.96-.81-3h12.94z"
		/>
		{/* Path for the darker green shading on the top of the green trapezoid */}
		<path
			fill="#2e7d32" // Darker green color fill for shading
			d="M30.46 6.8 24 18 17.53 6.8c.78-.45 1.66-.73 2.6-.79L27.46 6c1.08 0 2.11.28 3 .8z"
		/>
	</svg>
)

export default IconGoogleDrive
