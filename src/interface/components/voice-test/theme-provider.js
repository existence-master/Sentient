"use client"

import { createContext, useContext, useEffect, useState } from "react"
import React from "react"

const initialState = {
	theme: "system",
	setTheme: () => null
}

const ThemeProviderContext = createContext(initialState)

export function ThemeProvider({
	children,
	defaultTheme = "dark",
	storageKey = "theme",
	attribute = "class",
	enableSystem = true,
	disableTransitionOnChange = false,
	...props
}) {
	const [theme, setTheme] = useState(defaultTheme)

	useEffect(() => {
		const savedTheme = localStorage.getItem(storageKey)

		if (savedTheme) {
			setTheme(savedTheme)
		} else if (defaultTheme === "system" && enableSystem) {
			const systemTheme = window.matchMedia(
				"(prefers-color-scheme: dark)"
			).matches
				? "dark"
				: "light"
			setTheme(systemTheme)
		}
	}, [defaultTheme, storageKey, enableSystem])

	useEffect(() => {
		const root = window.document.documentElement

		if (disableTransitionOnChange) {
			root.classList.add("no-transitions")

			// Force a reflow
			window.getComputedStyle(root).getPropertyValue("opacity")

			setTimeout(() => {
				root.classList.remove("no-transitions")
			}, 0)
		}

		root.classList.remove("light", "dark")

		if (theme === "system" && enableSystem) {
			const systemTheme = window.matchMedia(
				"(prefers-color-scheme: dark)"
			).matches
				? "dark"
				: "light"
			root.classList.add(systemTheme)
		} else {
			root.classList.add(theme)
		}

		localStorage.setItem(storageKey, theme)
	}, [theme, storageKey, enableSystem, disableTransitionOnChange])

	const value = {
		theme,
		setTheme: (theme) => {
			setTheme(theme)
		}
	}

	return (
		<ThemeProviderContext.Provider {...props} value={value}>
			{children}
		</ThemeProviderContext.Provider>
	)
}

export const useTheme = () => {
	const context = useContext(ThemeProviderContext)

	if (context === undefined)
		throw new Error("useTheme must be used within a ThemeProvider")

	return context
}
