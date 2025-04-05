/**
 * @file Utility function for combining class names, especially for Tailwind CSS.
 * This module exports a function `cn` that merges Tailwind CSS classes and handles conditional class names,
 * ensuring that the final class string is optimized and conflict-free. It uses `clsx` for conditional classing
 * and `tailwind-merge` to resolve Tailwind class conflicts.
 */

import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"

/**
 * Utility function to conditionally join class names and merge Tailwind CSS class conflicts.
 * It uses `clsx` for efficiently handling conditional class names and `twMerge` to ensure
 * that Tailwind class conflicts are resolved correctly, with the last utility class taking precedence.
 *
 * @function cn
 * @param {...any} inputs - Any number of class name arguments, which can be strings, objects, or arrays
 *                          that `clsx` can handle. These are the class names to be combined and merged.
 * @returns {string} A single string of class names, with Tailwind CSS conflicts resolved, ready to be applied to HTML elements.
 *
 * @example
 * // Basic usage
 * cn("bg-red-500", "text-white", "p-4");
 * // Returns: "bg-red-500 text-white p-4"
 *
 * @example
 * // Conditional classes
 * cn("p-4", { "bg-blue-500": true, "text-white": false }, ["font-bold"]);
 * // Returns: "p-4 bg-blue-500 font-bold"
 *
 * @example
 * // Tailwind class merging - text-lg will override text-sm
 * cn("text-sm", "text-lg", "p-4");
 * // Returns: "text-lg p-4"
 */
export function cn(...inputs) {
	return twMerge(clsx(inputs))
}