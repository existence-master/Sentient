import { useUserStore } from "./app-stores"
import { act } from "@testing-library/react"

// Mocking fetch is handled in jest.setup.js

describe("useUserStore", () => {
	beforeEach(() => {
		fetch.mockClear()
		// Reset store to initial state before each test
		act(() => {
			useUserStore.setState(useUserStore.getInitialState())
		})
	})

	it("should have a correct initial state", () => {
		const state = useUserStore.getState()
		expect(state.user).toBeNull()
		expect(state.onboardingComplete).toBe(false)
		expect(state.isLoading).toBe(true) // This is the initial state
		expect(state.error).toBeNull()
	})

	it("should fetch user data and update the store on success", async () => {
		const mockUserData = {
			data: {
				pricing: "pro",
				onboardingComplete: true
			}
		}
		const mockProfileData = {
			name: "Test User",
			email: "test@example.com"
		}

		fetch
			.mockResolvedValueOnce({
				ok: true,
				json: async () => mockUserData
			})
			.mockResolvedValueOnce({
				ok: true,
				json: async () => mockProfileData
			})

		const { fetchUserData } = useUserStore.getState()

		await act(async () => {
			await fetchUserData()
		})

		const state = useUserStore.getState()

		expect(state.isLoading).toBe(false)
		expect(state.error).toBeNull()
		expect(state.onboardingComplete).toBe(true)
		expect(state.user).toEqual({ ...mockUserData.data, ...mockProfileData })
	})

	it("should handle fetch errors and update the store", async () => {
		const errorMessage = "Could not verify user status."
		fetch.mockResolvedValueOnce({
			ok: false
		})

		const { fetchUserData } = useUserStore.getState()

		await act(async () => {
			await fetchUserData()
		})

		const state = useUserStore.getState()

		expect(state.isLoading).toBe(false)
		expect(state.user).toBeNull()
		expect(state.error).toBe(errorMessage)
	})

	it("should handle network errors during fetch", async () => {
		const networkError = "Network request failed"
		fetch.mockRejectedValueOnce(new Error(networkError))

		const { fetchUserData } = useUserStore.getState()

		await act(async () => {
			await fetchUserData()
		})

		const state = useUserStore.getState()

		expect(state.isLoading).toBe(false)
		expect(state.user).toBeNull()
		expect(state.error).toBe(networkError)
	})
})
