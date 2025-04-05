import React from "react"
/**
 * IconGoogleCalendar Component - Renders the Google Calendar Icon as an SVG.
 *
 * This component is designed to display the Google Calendar icon using SVG paths.
 * It utilizes predefined path data to construct the icon, ensuring a consistent
 * and scalable representation of the Google Calendar logo in the application.
 *
 * The component is created as a functional component in React, accepting and
 * spreading any provided props to the root SVG element. This allows for
 * customization of the icon through standard SVG attributes like className,
 * style, and event handlers.
 *
 * @param {SVGProps<SVGSVGElement>} props - React props to be passed to the SVG element.
 *                                         Includes standard SVG attributes and React event handlers.
 *
 * @returns {React.ReactNode} - JSX element representing the Google Calendar icon as an SVG.
 *
 * @example
 * // Usage example within a React component:
 * <IconGoogleCalendar
 *   width={30}
 *   height={30}
 *   fill="#007BFF"
 *   className="custom-icon-class"
 *   onClick={() => alert('Calendar icon clicked')}
 * />
 */
const IconGoogleCalendar = (props) => (
	<svg
		xmlns="http://www.w3.org/2000/svg" // Specifies the namespace for SVG
		width={25} // Sets the width of the SVG icon
		height={25} // Sets the height of the SVG icon
		viewBox="0 0 50 50" // Defines the view box for the SVG, allowing scaling
		{...props} // Spreads any additional props passed to the component to the SVG element
	>
		{/* Path for the white square background of the calendar icon */}
		<path fill="#fff" d="M13 13h22v22H13z" />

		{/* Path for the blue calendar page curl and number '31' representation */}
		<path
			fill="#1e88e5" // Blue color fill for the path
			d="m25.68 20.92 1.008 1.44 1.584-1.152v8.352H30V18.616h-1.44zM22.943 23.745c.625-.574 1.013-1.37 1.013-2.249 0-1.747-1.533-3.168-3.417-3.168-1.602 0-2.972 1.009-3.33 2.453l1.657.421c.165-.664.868-1.146 1.673-1.146.942 0 1.709.646 1.709 1.44 0 .794-.767 1.44-1.709 1.44h-.997v1.728h.997c1.081 0 1.993.751 1.993 1.64 0 .904-.866 1.64-1.931 1.64-.962 0-1.784-.61-1.914-1.418L17 26.802c.262 1.636 1.81 2.87 3.6 2.87 2.007 0 3.64-1.511 3.64-3.368 0-1.023-.504-1.941-1.297-2.559z"
		/>

		{/* Path for the yellow bottom banner of the calendar icon */}
		<path fill="#fbc02d" d="M34 42H14l-1-4 1-4h20l1 4z" />

		{/* Path for the green vertical bar on the right side of the calendar */}
		<path fill="#4caf50" d="m38 35 4-1V14l-4-1-4 1v20z" />

		{/* Path for the blue main body of the calendar with top left corner cut */}
		<path
			fill="#1e88e5"
			d="m34 14 1-4-1-4H9a3 3 0 0 0-3 3v25l4 1 4-1V14h20z"
		/>

		{/* Path for the red triangle in the top right corner */}
		<path fill="#e53935" d="M34 34v8l8-8z" />

		{/* Path for the darker blue shading on the top right and bottom left corners */}
		<path
			fill="#1565c0"
			d="M39 6h-5v8h8V9a3 3 0 0 0-3-3zM9 42h5v-8H6v5a3 3 0 0 0 3 3z"
		/>
	</svg>
)

export default IconGoogleCalendar
