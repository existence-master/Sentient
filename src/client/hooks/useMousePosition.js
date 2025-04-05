import { useEffect } from "react" // Importing useEffect hook from React

/**
 * useMousePosition Hook - Tracks mouse position relative to a specified React ref element.
 *
 * This hook provides functionality to track the mouse cursor's position within the bounds of a DOM element,
 * specified by a React ref. It uses event listeners for 'mousemove' and 'touchmove' to capture cursor/touch
 * coordinates and updates a callback function with the relative position of the mouse inside the ref element.
 *
 * @param {React.RefObject<HTMLElement>} ref - React ref object attached to the DOM element to track mouse position within.
 * @param {function} callback - Callback function to be executed on mouse move, receives an object with x and y coordinates.
 * @returns {void}
 */
export function useMousePosition(
	ref, // React ref object attached to the target HTML element - ref: React.RefObject<HTMLElement>
	callback // Callback function to be called on mouse move events - callback: (position: { x: number, y: number }) => void
) {
	/**
	 * useEffect hook to set up and tear down mouse and touch event listeners.
	 *
	 * This hook is responsible for attaching event listeners for 'mousemove' and 'touchmove' events to the DOM element
	 * referenced by `ref`. It calculates the mouse position relative to the element's bounds and invokes the `callback`
	 * function with the x and y coordinates each time the mouse moves or touch moves within the element.
	 * It also sets up a cleanup function to remove the event listeners when the component unmounts or dependencies change,
	 * preventing memory leaks.
	 */
	useEffect(() => {
		/**
		 * Handles mousemove events to calculate and provide mouse position relative to the ref element.
		 *
		 * Retrieves mouse coordinates from the event, calculates the position relative to the bounding rectangle
		 * of the ref element, and invokes the callback function with these relative coordinates.
		 * @param {MouseEvent} event - The mousemove event object.
		 * @returns {void}
		 */
		const handleMouseMove = (event) => {
			const { clientX, clientY } = event // Mouse position relative to the viewport
			const { top, left } = ref.current?.getBoundingClientRect() || {
				// Get bounding rectangle of the ref element, default to {top: 0, left: 0} if ref.current is null
				top: 0,
				left: 0
			}

			callback?.({ x: clientX - left, y: clientY - top }) // Call callback function with mouse position relative to the ref element
		}

		/**
		 * Handles touchmove events to calculate and provide touch position relative to the ref element.
		 *
		 * Similar to `handleMouseMove`, but for touch events. It retrieves touch coordinates, calculates
		 * the position relative to the ref element, and invokes the callback function.
		 * @param {TouchEvent} event - The touchmove event object.
		 * @returns {void}
		 */
		const handleTouchMove = (event) => {
			const { clientX, clientY } = event.touches[0] // Touch position relative to the viewport (using first touch point)
			const { top, left } = ref.current?.getBoundingClientRect() || {
				// Get bounding rectangle of the ref element, default to {top: 0, left: 0} if ref.current is null
				top: 0,
				left: 0
			}

			callback?.({ x: clientX - left, y: clientY - top }) // Call callback function with touch position relative to the ref element
		}

		ref.current?.addEventListener("mousemove", handleMouseMove) // Add mousemove event listener to ref element
		ref.current?.addEventListener("touchmove", handleTouchMove) // Add touchmove event listener to ref element

		const nodeRef = ref.current // Store current ref in a variable for cleanup function

		// Cleanup function to remove event listeners when component unmounts or dependencies change
		return () => {
			nodeRef?.removeEventListener("mousemove", handleMouseMove) // Remove mousemove event listener
			nodeRef?.removeEventListener("touchmove", handleTouchMove) // Remove touchmove event listener
		}
	}, [ref, callback]) // Effect dependencies: ref, callback - effect runs when ref or callback changes
}