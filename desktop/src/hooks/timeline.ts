/** React Query hooks for the activity timeline (§21, #147). Live updates: lib/events.ts refetches ['timeline']. */
import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import type { TimelineFilters } from '@/lib/types'
import { qk } from './queryKeys'

const PAGE = 50

/** What Sentient did, newest first, one page at a time (`fetchNextPage` for older ones). */
export function useTimeline(filters: TimelineFilters) {
  return useInfiniteQuery({
    queryKey: qk.timeline.past(filters),
    queryFn: ({ pageParam }) => api.timeline.past(filters, pageParam, PAGE),
    initialPageParam: null as string | null,
    getNextPageParam: (last) => last.next
  })
}

/** What needs you now and what is scheduled in the coming week, plus paused routines. */
export function useUpcoming(filters: TimelineFilters) {
  return useQuery({ queryKey: qk.timeline.upcoming(filters), queryFn: () => api.timeline.upcoming(filters) })
}

export function useTimelineApps() {
  return useQuery({ queryKey: qk.timeline.apps, queryFn: api.timeline.apps, staleTime: 60_000 })
}
