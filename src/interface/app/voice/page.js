import React from "react"
import { BackgroundCircleProvider } from "@components/voice-test/background-circle-provider"
import { ThemeToggle } from "@components/voice-test/ui/theme-toggle"
import { ResetChat } from "@components/voice-test/ui/reset-chat"
export default function Home() {
	return (
		<div className="flex flex-col items-center justify-center h-screen">
			<BackgroundCircleProvider />
			<div className="absolute top-4 right-4 z-10">
				<ThemeToggle />
			</div>
			<div className="absolute bottom-4 right-4 z-10">
				<ResetChat />
			</div>
		</div>
	)
}
