import React, { useState } from "react"
import PropTypes from "prop-types"
import { IconCalendarEvent, IconUser, IconChevronDown, IconChevronUp, IconExternalLink } from "@tabler/icons-react"

/**
 * ToolResultCard Component - Displays a tool result (e.g., event details) in a styled card.
 *
 * @param {object} props - Component props.
 * @param {string} props.title - Main title of the result (e.g., event name).
 * @param {string} [props.subtitle] - Subtitle or secondary info (e.g., date/time).
 * @param {object|array} [props.details] - Additional details (object or array of key-value pairs).
 * @param {array} [props.attendees] - List of attendees (array of strings or objects).
 * @param {string} [props.externalLink] - Optional external link for more info.
 * @returns {React.ReactNode}
 */
const ToolResultCard = ({ title, subtitle, details, attendees, externalLink }) => {
  const [expanded, setExpanded] = useState(false)

  const handleToggle = () => setExpanded((prev) => !prev)

  return (
    <div className="w-full bg-[var(--color-primary-surface)] rounded-lg p-4 mb-4 border border-[var(--color-accent-blue)]">
      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <IconCalendarEvent className="w-6 h-6 text-[var(--color-accent-blue)]" />
        <h3 className="text-lg font-semibold text-[var(--color-text-primary)]">{title}</h3>
        {externalLink && (
          <a
            href={externalLink}
            target="_blank"
            rel="noopener noreferrer"
            className="ml-auto text-[var(--color-accent-blue)] hover:underline flex items-center gap-1"
            aria-label="Open in external app"
          >
            <IconExternalLink className="w-4 h-4" />
            <span className="text-xs">Open</span>
          </a>
        )}
      </div>
      {subtitle && <div className="text-[var(--color-text-secondary)] text-sm mb-2">{subtitle}</div>}
      {/* Expand/collapse details */}
      <button
        onClick={handleToggle}
        className="flex items-center gap-1 text-[var(--color-accent-blue)] hover:text-white text-sm mb-2 focus:outline-none"
        aria-expanded={expanded}
        aria-controls="tool-result-details"
      >
        {expanded ? <IconChevronUp className="w-4 h-4" /> : <IconChevronDown className="w-4 h-4" />}
        <span>{expanded ? "Hide details" : "Show details"}</span>
      </button>
      {expanded && (
        <div id="tool-result-details" className="mt-2 border-t border-[var(--color-primary-surface-elevated)] pt-2">
          {/* Render details as key-value pairs if object, or as list if array */}
          {details && typeof details === "object" && !Array.isArray(details) && (
            <ul className="mb-2">
              {Object.entries(details).map(([key, value]) => (
                <li key={key} className="text-[var(--color-text-primary)] text-sm flex gap-2">
                  <span className="font-medium">{key}:</span> <span>{String(value)}</span>
                </li>
              ))}
            </ul>
          )}
          {details && Array.isArray(details) && (
            <ul className="mb-2">
              {details.map((item, idx) => (
                <li key={idx} className="text-[var(--color-text-primary)] text-sm">{String(item)}</li>
              ))}
            </ul>
          )}
          {/* Attendees */}
          {attendees && attendees.length > 0 && (
            <div className="mt-2">
              <div className="flex items-center gap-2 mb-1">
                <IconUser className="w-4 h-4 text-[var(--color-accent-blue)]" />
                <span className="font-medium text-[var(--color-text-primary)]">Attendees:</span>
              </div>
              <ul className="ml-6 list-disc">
                {attendees.map((att, idx) => (
                  <li key={idx} className="text-[var(--color-text-secondary)] text-sm">
                    {typeof att === "string" ? att : att.name || JSON.stringify(att)}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

ToolResultCard.propTypes = {
  title: PropTypes.string.isRequired,
  subtitle: PropTypes.string,
  details: PropTypes.oneOfType([
    PropTypes.object,
    PropTypes.array
  ]),
  attendees: PropTypes.arrayOf(
    PropTypes.oneOfType([
      PropTypes.string,
      PropTypes.object
    ])
  ),
  externalLink: PropTypes.string
}

export default ToolResultCard 