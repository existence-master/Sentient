import React from "react"
/**
 * IconGoogleMail Component - Renders the Google Mail (Gmail) Icon as an SVG.
 *
 * This component is designed to render the Google Mail icon using SVG paths.
 * It uses a collection of predefined path data to accurately represent the Gmail logo,
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
 * @returns {React.ReactNode} - JSX element representing the Google Mail icon as an SVG.
 *
 * @example
 * // Usage example within a React component:
 * <IconGoogleMail
 *   width={30}
 *   height={30}
 *   fill="#D44638"
 *   className="gmail-icon"
 *   onClick={() => console.log('Gmail icon clicked')}
 * />
 */
const IconGoogleMail = (props) => (
	<svg
		xmlns="http://www.w3.org/2000/svg" // Specifies the namespace for SVG
		width={25} // Sets the width of the SVG icon to 25 pixels
		height={25} // Sets the height of the SVG icon to 25 pixels
		viewBox="0 0 50 50" // Defines the view box for scaling the icon
		{...props} // Spreads any additional props to the SVG element for customization
	>
		{/* Path for the green right side shape of the Gmail envelope */}
		<path
			fill="#4caf50"
			d="m45 16.2-5 2.75-5 4.75V40h7a3 3 0 0 0 3-3V16.2z"
		/>
		{/* Path for the blue left side shape of the Gmail envelope */}
		<path
			fill="#1e88e5" // Blue color fill for the path
			d="m3 16.2 3.614 1.71L13 23.7V40H6a3 3 0 0 1-3-3V16.2z"
		/>
		{/* Path for the red central shape of the Gmail envelope, forming the 'M' */}
		<path
			fill="#e53935" // Red color fill for the path
			d="m35 11.2-11 8.25-11-8.25-1 5.8 1 6.7 11 8.25 11-8.25 1-6.7z"
		/>
		{/* Path for the darker red shading on the bottom left corner */}
		<path
			fill="#c62828" // Darker red color fill for shading
			d="M3 12.298V16.2l10 7.5V11.2L9.876 8.859A4.298 4.298 0 0 0 3 12.298z"
		/>
		{/* Path for the yellow shading on the bottom right corner */}
		<path
			fill="#fbc02d" // Yellow color fill for shading
			d="M45 12.298V16.2l-10 7.5V11.2l3.124-2.341A4.298 4.298 0 0 1 45 12.298z"
		/>
	</svg>
)

export default IconGoogleMail
