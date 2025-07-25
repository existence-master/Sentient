// src/client/components/ProIcon.js
import { Tooltip } from "react-tooltip" // Corrected import path
import { IconStar } from "@tabler/icons-react" // Corrected import path
import React from "react"

/**
 * ProIcon Component - Displays a "Pro" icon with a tooltip indicating feature availability for Pro users only.
 *
 * This component is used to denote features that are exclusive to Pro users. It renders a star icon
 * and integrates a tooltip that appears on hover, explaining that the feature is available only
 * to users with a Pro subscription and directs them to the Settings page for upgrade options.
 *
 * @returns {React.ReactNode} - The ProIcon component UI, which includes a star icon and a tooltip.
 */
const ProIcon = () => (
	<div className="flex items-center">
		{/* Container div for icon and tooltip, using flex to align items vertically center */}
		<span
			data-tooltip-id="pro-feature" // Unique ID for the tooltip, used to associate Tooltip component
			data-tooltip-content="This feature is only available for Pro users. You can upgrade anytime from the Settings page."
			className="text-[var(--color-accent-orange)] ml-2 cursor-pointer"
			// Styling for the span element: yellow-400 text color, margin left for spacing, cursor pointer to indicate interactivity
		>
			<IconStar />{" "}
			{/* IconStar component to visually represent a Pro feature */}
		</span>
		{/* Tooltip component from react-tooltip, providing hover information */}
		<Tooltip id="pro-feature" place="right" type="dark" effect="float" />
		{/* Configuration for the tooltip: id to connect with the span, placement on the right of the icon, dark theme, float effect */}
	</div>
)

export default ProIcon
