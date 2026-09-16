import { QueryClient } from '@tanstack/react-query'
import { isApiError } from './api'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      refetchOnWindowFocus: false,
      retry: (failureCount, error) => {
        // Don't retry client errors (404 = endpoint not built yet, 401, 422...).
        if (isApiError(error) && error.status >= 400 && error.status < 500) return false
        return failureCount < 2
      },
      retryDelay: (attempt) => Math.min(4000, 400 * 2 ** attempt)
    },
    mutations: { retry: false }
  }
})
