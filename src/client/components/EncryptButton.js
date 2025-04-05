import { useRef, useState } from "react" // Importing React hooks: useRef, useState
import { motion } from "framer-motion" // Importing motion component from framer-motion library
import { IconEdit, IconTrash } from "@tabler/icons-react" // Importing icons for edit and trash actions from tabler-icons-react library

/**
 * EncryptButton Component - A button with text scrambling animation on hover and edit/delete actions.
 *
 * This component renders a button that, on mouse hover, scrambles its text with random characters
 * before reverting to the original text on mouse leave. It also includes edit and delete icons,
 * triggering respective onClick events. The text scrambling effect is purely visual and for UX enhancement.
 *
 * @param {object} props - Component props.
 * @param {React.ReactNode} props.children - The text to be displayed on the button and scrambled during hover.
 * @param {function} props.onClick - Handler function to be called when the button (text area) is clicked.
 * @param {string} props.className - Optional CSS class names to apply to the button.
 * @param {function} props.onEdit - Handler function called when the edit icon is clicked.
 * @param {function} props.onDelete - Handler function called when the delete icon is clicked.
 * @returns {React.ReactNode} - The EncryptButton component UI.
 */
const EncryptButton = ({ children, onClick, className, onEdit, onDelete }) => {
	const TARGET_TEXT = children[0] // Original text to be displayed and scrambled - TARGET_TEXT: string
	const CYCLES_PER_LETTER = 2 // Number of scramble cycles per letter - CYCLES_PER_LETTER: number
	const SHUFFLE_TIME = 50 // Time interval for each character shuffle in milliseconds - SHUFFLE_TIME: number
	const CHARS = "!@#$%^&*():{};|,.<>/?" // Character set for scrambling effect - CHARS: string

	const intervalRef = useRef(null) // useRef to hold the interval ID for text scrambling - intervalRef: React.RefObject<number | null>
	// State to manage the displayed text, initially set to the original text - text: string
	const [text, setText] = useState(TARGET_TEXT)

	/**
	 * Scrambles the button text character by character.
	 *
	 * Initiates an interval that replaces characters in the button text with random characters from `CHARS` set,
	 * creating a scrambling effect. The scrambling progresses from left to right, CYCLES_PER_LETTER times per character,
	 * until the entire text is scrambled and then calls `stopScramble` to revert to the original text.
	 *
	 * @function scramble
	 * @returns {void}
	 */
	const scramble = () => {
		let pos = 0 // Current position in the text scrambling process - pos: number

		// Set interval to update text with scrambled characters
		intervalRef.current = setInterval(() => {
			// Create scrambled text by mapping over each character of TARGET_TEXT
			const scrambled = TARGET_TEXT.split("")
				.map((char, index) => {
					// Characters up to current position are kept original
					if (pos / CYCLES_PER_LETTER > index) {
						return char // Return original character
					}

					// For characters beyond current position, replace with random characters
					const randomCharIndex = Math.floor(
						Math.random() * CHARS.length
					) // Get random index from CHARS
					const randomChar = CHARS[randomCharIndex] // Get random character from CHARS

					return randomChar // Return random character for scrambling effect
				})
				.join("") // Join characters back into a string

			setText(scrambled) // Update text state with scrambled text
			pos++ // Increment position to move to next character in next interval

			// Stop scrambling when position exceeds total characters times cycles per letter
			if (pos >= TARGET_TEXT.length * CYCLES_PER_LETTER) {
				stopScramble() // Call stopScramble to clear interval and reset text
			}
		}, SHUFFLE_TIME) // Interval timing for scramble effect
	}

	/**
	 * Stops the text scramble animation and resets the text to the original.
	 *
	 * Clears the interval set by `scramble` to stop the text scrambling animation.
	 * Resets the `text` state to `TARGET_TEXT`, displaying the original button text.
	 *
	 * @function stopScramble
	 * @returns {void}
	 */
	const stopScramble = () => {
		clearInterval(intervalRef.current || undefined) // Clear the interval to stop scrambling
		setText(TARGET_TEXT) // Reset text state to original TARGET_TEXT
	}

	return (
		<motion.button
			whileHover={{ scale: 1.025 }} // সামান্য স্কেল বৃদ্ধি যখন হোভার করা হয়
			whileTap={{ scale: 0.975 }} // সামান্য স্কেল কমানো যখন ট্যাপ করা হয়
			onMouseEnter={scramble} // কল স্ক্র্যাম্বল ফাংশন যখন মাউস প্রবেশ করে
			onMouseLeave={stopScramble} // কল স্টপস্ক্র্যাম্বল ফাংশন যখন মাউস ছেড়ে যায়
			onClick={onClick} // কল অনক্লিক হ্যান্ডলার যখন বাটন ক্লিক করা হয়
			className={`relative flex items-center justify-between group overflow-hidden rounded-lg border-[1px] border-black bg-white px-4 py-2 font-mono font-medium uppercase text-black transition-colors hover:text-white ${className}`}
			// বাটন স্টাইলিং: আপেক্ষিক পজিশনিং, ফ্লেক্স ডিসপ্লে, ওভারফ্লো লুকানো, বর্ডার, ব্যাকগ্রাউন্ড, প্যাডিং, ফন্ট, টেক্সট ট্রান্সফর্মেশন, টেক্সট কালার, ট্রানজিশন, এবং ডাইনামিক ক্লাস
		>
			<span>{text}</span>
			{/* বাটন টেক্সট, যা স্ক্র্যাম্বলিং ইফেক্ট দেখায় */}

			<div className="flex items-center gap-2 z-10">
				{/* আইকনগুলির জন্য ধারক: ফ্লেক্স ডিসপ্লে, আইটেম সেন্টার, গ্যাপ, জেড-ইনডেক্স */}
				<IconEdit
					className="w-5 h-5 transition-colors cursor-pointer" // আইকন স্টাইলিং: প্রস্থ, উচ্চতা, ট্রানজিশন, কার্সর
					onClick={() => onEdit()} // অনক্লিক হ্যান্ডলার - সম্পাদনা ফাংশন কল করে
				/>
				<IconTrash
					className="w-5 h-5 transition-colors cursor-pointer" // আইকন স্টাইলিং: প্রস্থ, উচ্চতা, ট্রানজিশন, কার্সর
					onClick={() => onDelete()} // অনক্লিক হ্যান্ডলার - ডিলিট ফাংশন কল করে
				/>
			</div>
		</motion.button>
	)
}

export default EncryptButton