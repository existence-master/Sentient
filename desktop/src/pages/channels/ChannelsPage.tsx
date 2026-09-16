import { Navigate } from 'react-router'

/** `/channels` is an alias for the "Messaging apps" tab of Devices. */
export function ChannelsPage() {
  return <Navigate to="/devices/messaging" replace />
}
