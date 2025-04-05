/** @type {import('tailwindcss').Config} */
module.exports = {
	content: [
		"./pages/**/*.{js,ts,jsx,tsx,mdx}",
		"./components/**/*.{js,ts,jsx,tsx,mdx}",
		"./app/**/*.{js,ts,jsx,tsx,mdx}",
		"./src/**/*.{ts,tsx}"
	],
	darkMode: "class",
	theme: {
		extend: {
			animation: {
				shimmer: "shimmer 2s linear infinite",
				sparkle: "sparkle 2s ease-in-out infinite",
				rotate: "rotate 10s linear infinite",
				first: "moveVertical 30s ease infinite",
				second: "moveInCircle 20s reverse infinite",
				third: "moveInCircle 40s linear infinite",
				fourth: "moveHorizontal 40s ease infinite",
				fifth: "moveInCircle 20s ease infinite",
				meteor: "meteor var(--duration) var(--delay) ease-in-out infinite"
			},
			keyframes: {
				meteor: {
					"0%": { transform: "translateY(-20%) translateX(-50%)" },
					"100%": { transform: "translateY(300%) translateX(-50%)" }
				},
				shimmer: {
					from: {
						backgroundPosition: "0 0"
					},
					to: {
						backgroundPosition: "-200% 0"
					},
					moveHorizontal: {
						"0%": {
							transform: "translateX(-50%) translateY(-10%)"
						},
						"50%": {
							transform: "translateX(50%) translateY(10%)"
						},
						"100%": {
							transform: "translateX(-50%) translateY(-10%)"
						}
					},
					moveInCircle: {
						"0%": {
							transform: "rotate(0deg)"
						},
						"50%": {
							transform: "rotate(180deg)"
						},
						"100%": {
							transform: "rotate(360deg)"
						}
					},
					moveVertical: {
						"0%": {
							transform: "translateY(-50%)"
						},
						"50%": {
							transform: "translateY(50%)"
						},
						"100%": {
							transform: "translateY(-50%)"
						}
					}
				},
				sparkle: {
					"0%, 100%": { opacity: "0.75", scale: "0.9" },
					"50%": { opacity: "1", scale: "1" }
				},
				rotate: {
					"0%": { transform: "rotate(0deg) scale(10)" },
					"100%": { transform: "rotate(-360deg) scale(10)" }
				}
			},
			fontFamily: {
				Poppins: ["Poppins", "sans-serif"],
				Montserrat: ["Montserrat", "sans-serif"],
				Quicksand: ["Quicksand", "sans-serif"]
			},
			colors: {
				lightblue: "#00B2FE",
				darkblue: "#005CFE",
				matteblack: "#212121",
				smokeblack: "#1A1A1A"
			},
			screens: {
				xs: "128px",
				"2xs": "430px",
				sm: "640px",
				md: "768px",
				lg: "1024px",
				xl: "1280px",
				"2xl": "1536px"
			},
			boxShadow: {
				glow: "0 0 20px rgba(65, 105, 225, 0.7), 0 0 40px rgba(0, 123, 255, 0.5), 0 0 60px rgba(30, 144, 255, 0.3)",
				glow2: "0 0 20px rgba(0, 191, 255, 0.7), 0 0 40px rgba(70, 130, 180, 0.5), 0 0 60px rgba(100, 149, 237, 0.3)"
			},
			filter: {
				"blur-20": "blur(20px)",
				"blur-25": "blur(25px)"
			},
			brightness: {
				150: "1.5"
			}
		}
	},
	plugins: []
}
