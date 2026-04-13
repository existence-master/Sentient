import os
import asyncio
import json
import datetime, inspect
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from fastmcp.prompts.prompt import Message
from fastmcp.utilities.logging import configure_logging, get_logger
from composio import Composio
from main.config import COMPOSIO_API_KEY

# Local imports
from . import auth
from . import prompts

# --- Standardized Logging Setup ---
configure_logging(level="INFO")
logger = get_logger(__name__)

composio = Composio(api_key=COMPOSIO_API_KEY)

# Conditionally load .env for local development
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)

# --- Server Initialization ---
mcp = FastMCP(
    name="GCalServer",
    instructions="Provides a comprehensive suite of tools to manage Google Calendar, including creating, updating, deleting, and searching for events and calendars."
)

# --- Prompt Registration ---
@mcp.resource("prompt://gcal-agent-system")
def get_gcal_system_prompt() -> str:
    """Provides the system prompt for the GCal agent."""
    return prompts.gcal_agent_system_prompt

# --- Helper for Tool Execution ---
async def _execute_tool(ctx: Context, action_name: str, **kwargs) -> Dict[str, Any]:
    """Helper to handle auth and execution for all tools using Composio."""
    tool_name = inspect.stack()[1].function
    logger.info(f"Executing tool: {tool_name} with parameters: {kwargs}")
    try:
        user_id = auth.get_user_id_from_context(ctx)
        connection_id = await auth.get_composio_connection_id(user_id, "gcalendar")

        # NEW: Fetch user info including privacy filters
        user_info = await auth.get_user_info(user_id)
        privacy_filters = user_info.get("privacy_filters", {})
        keyword_filters = privacy_filters.get("keywords", [])
        email_filters = [email.lower() for email in privacy_filters.get("emails", [])]

        # Composio's execute method is synchronous, so we use asyncio.to_thread
        # Filter out None values from kwargs before passing to Composio
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        result = await asyncio.to_thread(
            composio.tools.execute,
            action_name,
            arguments=filtered_kwargs,
            connected_account_id=connection_id,
            dangerously_skip_version_check=True
        )

        # NEW: Apply privacy filters if the action is fetching events
        if action_name in ["GOOGLECALENDAR_EVENTS_LIST", "GOOGLECALENDAR_FIND_EVENT", "GOOGLECALENDAR_EVENTS_INSTANCES", "GOOGLECALENDAR_SYNC_EVENTS"]:
            if result.get("successful") and result.get("data"):
                events = result["data"].get("items", [])
                filtered_events = []
                for event in events:
                    content_to_check = (event.get("summary", "") + " " + event.get("description", "")).lower()

                    # Keyword check
                    if any(word.lower() in content_to_check for word in keyword_filters):
                        logger.info(f"Filtering event '{event.get('summary')}' due to keyword match.")
                        continue

                    # Attendee email check
                    if email_filters:
                        attendees = event.get("attendees", [])
                        if attendees:
                            attendee_emails = {attendee.get("email", "").lower() for attendee in attendees if attendee.get("email")}
                            if any(blocked_email in attendee_emails for blocked_email in email_filters):
                                logger.info(f"Filtering event '{event.get('summary')}' due to attendee email match.")
                                continue

                    filtered_events.append(event)

                result["data"]["items"] = filtered_events
                logger.info(f"Applied privacy filters. Kept {len(filtered_events)} out of {len(events)} events.")

        # FIX: Ensure the result from the SDK is JSON serializable before returning.
        # This converts Pydantic models, datetimes, etc., into a clean dictionary.
        try:
            serializable_result = json.loads(json.dumps(result, default=str))
            return {"status": "success", "result": serializable_result}
        except TypeError as e:
            logger.error(f"Failed to serialize result for action '{action_name}': {e}", exc_info=True)
            # Fallback to a string representation if direct serialization fails
            return {"status": "success", "result": str(result)}

    except Exception as e:
        logger.error(f"Tool execution failed for action '{action_name}': {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

# --- Tool Definitions ---

@mcp.tool()
async def delete_calendar(ctx: Context, calendarId: str) -> Dict:
    """Deletes a secondary calendar."""
    params = {"calendar_id": calendarId}
    return await _execute_tool(ctx, "GOOGLECALENDAR_CALENDARS_DELETE", **params)

@mcp.tool()
async def update_calendar(ctx: Context, calendarId: str, summary: str, description: Optional[str] = None, location: Optional[str] = None, timeZone: Optional[str] = None) -> Dict:
    """Updates metadata for a calendar."""
    params = {"calendarId": calendarId, "summary": summary, "description": description, "location": location, "timeZone": timeZone}
    return await _execute_tool(ctx, "GOOGLECALENDAR_CALENDARS_UPDATE", **params)

@mcp.tool()
async def insert_calendar_into_list(ctx: Context, id: str, background_color: Optional[str] = None, color_id: Optional[str] = None, color_rgb_format: Optional[bool] = None, default_reminders: Optional[List[Dict]] = None, foreground_color: Optional[str] = None, hidden: Optional[bool] = None, notification_settings: Optional[Dict] = None, selected: Optional[bool] = None, summary_override: Optional[str] = None) -> Dict:
    """Inserts an existing calendar into the user's calendar list."""
    params = {k: v for k, v in locals().items() if k != 'ctx' and v is not None}
    return await _execute_tool(ctx, "GOOGLECALENDAR_CALENDAR_LIST_INSERT", **params)

@mcp.tool()
async def create_event(ctx: Context, start_datetime: str, summary: Optional[str] = None, attendees: Optional[List[Dict]] = None, calendar_id: str = "primary", create_meeting_room: Optional[bool] = None, description: Optional[str] = None, eventType: str = "default", event_duration_hour: Optional[int] = None, event_duration_minutes: int = 30, guestsCanInviteOthers: Optional[bool] = None, guestsCanSeeOtherGuests: Optional[bool] = None, guests_can_modify: Optional[bool] = None, location: Optional[str] = None, recurrence: Optional[List[str]] = None, send_updates: Optional[bool] = None, timezone: Optional[str] = None, transparency: str = "opaque", visibility: str = "default") -> Dict:
    """Creates an event on a google calendar."""
    # WORKAROUND for Composio inconsistency: convert attendees to list of strings
    attendees_to_send = attendees
    if attendees and isinstance(attendees, list):
        processed_attendees = []
        for attendee in attendees:
            if isinstance(attendee, dict) and 'email' in attendee:
                processed_attendees.append(attendee['email'])
            elif isinstance(attendee, str):
                processed_attendees.append(attendee)
        attendees_to_send = processed_attendees

    params = {"start_datetime": start_datetime, "summary": summary, "attendees": attendees_to_send, "calendar_id": calendar_id, "create_meeting_room": create_meeting_room, "description": description, "eventType": eventType, "event_duration_hour": event_duration_hour, "event_duration_minutes": event_duration_minutes, "guestsCanInviteOthers": guestsCanInviteOthers, "guestsCanSeeOtherGuests": guestsCanSeeOtherGuests, "guests_can_modify": guests_can_modify, "location": location, "recurrence": recurrence, "send_updates": send_updates, "timezone": timezone, "transparency": transparency, "visibility": visibility}
    return await _execute_tool(ctx, "GOOGLECALENDAR_CREATE_EVENT", **params)

@mcp.tool()
async def delete_event(ctx: Context, event_id: str, calendar_id: str = "primary") -> Dict:
    """Deletes a specified event by `event id` from a google calendar."""
    params = {"event_id": event_id, "calendar_id": calendar_id}
    return await _execute_tool(ctx, "GOOGLECALENDAR_DELETE_EVENT", **params)

@mcp.tool()
async def duplicate_calendar(ctx: Context, summary: str) -> Dict:
    """Creates a new, empty google calendar with the specified title (summary)."""
    params = {"summary": summary}
    return await _execute_tool(ctx, "GOOGLECALENDAR_DUPLICATE_CALENDAR", **params)

@mcp.tool()
async def get_event_instances(ctx: Context, eventId: str, calendarId: str = "primary", maxAttendees: Optional[int] = None, maxResults: Optional[int] = None, originalStart: Optional[str] = None, pageToken: Optional[str] = None, showDeleted: Optional[bool] = None, timeMax: Optional[str] = None, timeMin: Optional[str] = None, timeZone: Optional[str] = None) -> Dict:
    """Returns instances of the specified recurring event."""
    params = {"calendarId": calendarId, "eventId": eventId, "maxAttendees": maxAttendees, "maxResults": maxResults, "originalStart": originalStart, "pageToken": pageToken, "showDeleted": showDeleted, "timeMax": timeMax, "timeMin": timeMin, "timeZone": timeZone}
    return await _execute_tool(ctx, "GOOGLECALENDAR_EVENTS_INSTANCES", **params)

@mcp.tool()
async def list_events(
    ctx: Context,
    calendarId: str = "primary",
    alwaysIncludeEmail: Optional[bool] = None,
    eventTypes: Optional[str] = None,
    iCalUID: Optional[str] = None,
    maxAttendees: Optional[int] = None,
    maxResults: Optional[int] = None,
    orderBy: Optional[str] = None,
    pageToken: Optional[str] = None,
    privateExtendedProperty: Optional[str] = None,
    q: Optional[str] = None,
    sharedExtendedProperty: Optional[str] = None,
    showDeleted: Optional[bool] = None,
    showHiddenInvitations: Optional[bool] = None,
    singleEvents: Optional[bool] = None,
    syncToken: Optional[str] = None,
    timeMax: Optional[str] = None,
    timeMin: Optional[str] = None,
    timeZone: Optional[str] = None,
    updatedMin: Optional[str] = None
) -> Dict:
    """Returns events on the specified calendar."""
    params = {
        "calendarId": calendarId, "alwaysIncludeEmail": alwaysIncludeEmail, "eventTypes": eventTypes,
        "iCalUID": iCalUID, "maxAttendees": maxAttendees, "maxResults": maxResults, "orderBy": orderBy,
        "pageToken": pageToken, "privateExtendedProperty": privateExtendedProperty, "q": q,
        "sharedExtendedProperty": sharedExtendedProperty, "showDeleted": showDeleted,
        "showHiddenInvitations": showHiddenInvitations, "singleEvents": singleEvents,
        "syncToken": syncToken, "timeMax": timeMax, "timeMin": timeMin, "timeZone": timeZone,
        "updatedMin": updatedMin
    }

    # If no time range is specified at all, default to the next 7 days to avoid outdated defaults.
    if params.get('timeMin') is None and params.get('timeMax') is None and params.get('q') is None:
        now = datetime.datetime.now(datetime.timezone.utc)
        future = now + datetime.timedelta(days=7)
        params['timeMin'] = now.isoformat()
        params['timeMax'] = future.isoformat()
        logger.info(f"Defaulting timeMin and timeMax to the next 7 days.")
    
    # If only timeMin is provided, set timeMax to 30 days after timeMin to prevent errors.
    elif params.get('timeMin') and params.get('timeMax') is None:
        time_min_dt = datetime.datetime.fromisoformat(str(params['timeMin']).replace("Z", "+00:00"))
        time_max_dt = time_min_dt + datetime.timedelta(days=30)
        params['timeMax'] = time_max_dt.isoformat()
        logger.info(f"Automatically setting timeMax because timeMin was provided.")

    return await _execute_tool(ctx, "GOOGLECALENDAR_EVENTS_LIST", **params)

@mcp.tool()
async def find_event(ctx: Context, calendar_id: str = "primary", event_types: Optional[List[str]] = None, max_results: Optional[int] = None, order_by: Optional[str] = None, page_token: Optional[str] = None, query: Optional[str] = None, show_deleted: Optional[bool] = None, single_events: bool = True, timeMax: Optional[str] = None, timeMin: Optional[str] = None, updated_min: Optional[str] = None) -> Dict:
    """Finds events in a specified google calendar using text query, time ranges, and event types."""
    params = {"calendar_id": calendar_id, "event_types": event_types, "max_results": max_results, "order_by": order_by, "page_token": page_token, "query": query, "show_deleted": show_deleted, "single_events": single_events, "timeMax": timeMax, "timeMin": timeMin, "updated_min": updated_min}
    return await _execute_tool(ctx, "GOOGLECALENDAR_FIND_EVENT", **params)

@mcp.tool()
async def free_busy_query(ctx: Context, timeMax: str, timeMin: str, items: List[Dict], calendarExpansionMax: Optional[int] = None, groupExpansionMax: Optional[int] = None, timeZone: Optional[str] = None) -> Dict:
    """Returns free/busy information for a set of calendars."""
    params = {"timeMax": timeMax, "timeMin": timeMin, "items": items, "calendarExpansionMax": calendarExpansionMax, "groupExpansionMax": groupExpansionMax, "timeZone": timeZone}
    return await _execute_tool(ctx, "GOOGLECALENDAR_FREE_BUSY_QUERY", **params)

@mcp.tool()
async def get_calendar(ctx: Context, calendar_id: str = "primary") -> Dict:
    """Retrieves a specific google calendar."""
    params = {"calendar_id": calendar_id}
    return await _execute_tool(ctx, "GOOGLECALENDAR_GET_CALENDAR", **params)

@mcp.tool()
async def patch_calendar(ctx: Context, calendar_id: str, summary: str, description: Optional[str] = None, location: Optional[str] = None, timezone: Optional[str] = None) -> Dict:
    """Partially updates (patches) an existing google calendar."""
    params = {"calendar_id": calendar_id, "summary": summary, "description": description, "location": location, "timezone": timezone}
    return await _execute_tool(ctx, "GOOGLECALENDAR_PATCH_CALENDAR", **params)

@mcp.tool()
async def patch_event(ctx: Context, calendar_id: str, event_id: str, attendees: Optional[List[Dict]] = None, conference_data_version: Optional[int] = None, description: Optional[str] = None, end_time: Optional[str] = None, location: Optional[str] = None, max_attendees: Optional[int] = None, rsvp_response: Optional[str] = None, send_updates: Optional[str] = None, start_time: Optional[str] = None, summary: Optional[str] = None, supports_attachments: Optional[bool] = None, timezone: Optional[str] = None) -> Dict:
    """Updates specified fields of an existing event in a google calendar using patch semantics."""
    # WORKAROUND for Composio inconsistency: convert attendees to list of strings
    attendees_to_send = attendees
    if attendees and isinstance(attendees, list):
        processed_attendees = []
        for attendee in attendees:
            if isinstance(attendee, dict) and 'email' in attendee:
                processed_attendees.append(attendee['email'])
            elif isinstance(attendee, str):
                processed_attendees.append(attendee)
        attendees_to_send = processed_attendees

    params = {"calendar_id": calendar_id, "event_id": event_id, "attendees": attendees_to_send, "conference_data_version": conference_data_version, "description": description, "end_time": end_time, "location": location, "max_attendees": max_attendees, "rsvp_response": rsvp_response, "send_updates": send_updates, "start_time": start_time, "summary": summary, "supports_attachments": supports_attachments, "timezone": timezone}
    return await _execute_tool(ctx, "GOOGLECALENDAR_PATCH_EVENT", **params)

@mcp.tool()
async def update_event(ctx: Context, event_id: str, start_datetime: str, attendees: Optional[List[Dict]] = None, calendar_id: str = "primary", create_meeting_room: Optional[bool] = None, description: Optional[str] = None, eventType: str = "default", event_duration_hour: Optional[int] = None, event_duration_minutes: int = 30, guestsCanInviteOthers: Optional[bool] = None, guestsCanSeeOtherGuests: Optional[bool] = None, guests_can_modify: Optional[bool] = None, location: Optional[str] = None, recurrence: Optional[List[str]] = None, send_updates: Optional[bool] = None, summary: Optional[str] = None, timezone: Optional[str] = None, transparency: str = "opaque", visibility: str = "default") -> Dict:
    """Updates an existing event by `event id` in a google calendar."""
    # WORKAROUND for Composio inconsistency: convert attendees to list of strings
    attendees_to_send = attendees
    if attendees and isinstance(attendees, list):
        processed_attendees = []
        for attendee in attendees:
            if isinstance(attendee, dict) and 'email' in attendee:
                processed_attendees.append(attendee['email'])
            elif isinstance(attendee, str):
                processed_attendees.append(attendee)
        attendees_to_send = processed_attendees

    params = {"event_id": event_id, "start_datetime": start_datetime, "attendees": attendees_to_send, "calendar_id": calendar_id, "create_meeting_room": create_meeting_room, "description": description, "eventType": eventType, "event_duration_hour": event_duration_hour, "event_duration_minutes": event_duration_minutes, "guestsCanInviteOthers": guestsCanInviteOthers, "guestsCanSeeOtherGuests": guestsCanSeeOtherGuests, "guests_can_modify": guests_can_modify, "location": location, "recurrence": recurrence, "send_updates": send_updates, "summary": summary, "timezone": timezone, "transparency": transparency, "visibility": visibility}
    return await _execute_tool(ctx, "GOOGLECALENDAR_UPDATE_EVENT", **params)

@mcp.tool()
async def update_calendar_list_entry(ctx: Context, calendar_id: str, backgroundColor: Optional[str] = None, colorId: Optional[str] = None, colorRgbFormat: Optional[bool] = None, defaultReminders: Optional[List[Dict]] = None, foregroundColor: Optional[str] = None, hidden: Optional[bool] = None, notificationSettings: Optional[Dict] = None, selected: Optional[bool] = None, summaryOverride: Optional[str] = None) -> Dict:
    """Updates an existing entry on the user's calendar list."""
    params = {"calendar_id": calendar_id, "backgroundColor": backgroundColor, "colorId": colorId, "colorRgbFormat": colorRgbFormat, "defaultReminders": defaultReminders, "foregroundColor": foregroundColor, "hidden": hidden, "notificationSettings": notificationSettings, "selected": selected, "summaryOverride": summaryOverride}
    return await _execute_tool(ctx, "GOOGLECALENDAR_CALENDAR_LIST_UPDATE", **params)

@mcp.tool()
async def move_event(ctx: Context, calendar_id: str, destination: str, event_id: str, send_updates: Optional[str] = None) -> Dict:
    """Moves an event to another calendar."""
    params = {"calendar_id": calendar_id, "destination": destination, "event_id": event_id, "send_updates": send_updates}
    return await _execute_tool(ctx, "GOOGLECALENDAR_EVENTS_MOVE", **params)

@mcp.tool()
async def find_free_slots(ctx: Context, items: List[str] = ["primary"], time_max: Optional[str] = None, time_min: Optional[str] = None, timezone: str = "UTC", calendar_expansion_max: int = 50, group_expansion_max: int = 100) -> Dict:
    """Finds free/busy time slots in google calendars."""
    params = {"items": items, "time_max": time_max, "time_min": time_min, "timezone": timezone, "calendar_expansion_max": calendar_expansion_max, "group_expansion_max": group_expansion_max}
    return await _execute_tool(ctx, "GOOGLECALENDAR_FIND_FREE_SLOTS", **params)

@mcp.tool()
async def get_current_date_time(ctx: Context, timezone: Optional[int] = None) -> Dict:
    """
    Gets the current date and time in UTC as an RFC3339 formatted string, which is the format required by other calendar tools.
    """
    logger.info(f"Executing tool: get_current_date_time with timezone='{timezone}'")
    try:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        # The Google Calendar API requires RFC3339 format, which is what isoformat() produces with a 'Z'
        # We remove microseconds for better compatibility with various parsers.
        current_datetime_str = now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
        return {
            "status": "success",
            "result": {"current_datetime": current_datetime_str}
        }
    except Exception as e:
        logger.error(f"Tool get_current_date_time failed: {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def list_calendars(ctx: Context, max_results: int = 10, min_access_role: Optional[str] = None, page_token: Optional[str] = None, show_deleted: Optional[bool] = None, show_hidden: Optional[bool] = None, sync_token: Optional[str] = None) -> Dict:
    """Retrieves calendars from the user's google calendar list."""
    params = {"max_results": max_results, "min_access_role": min_access_role, "page_token": page_token, "show_deleted": show_deleted, "show_hidden": show_hidden, "sync_token": sync_token}
    return await _execute_tool(ctx, "GOOGLECALENDAR_LIST_CALENDARS", **params)

@mcp.tool()
async def remove_attendee_from_event(ctx: Context, attendee_email: str, event_id: str, calendar_id: str = "primary") -> Dict:
    """Removes an attendee from a specified event in a google calendar."""
    params = {"attendee_email": attendee_email, "event_id": event_id, "calendar_id": calendar_id}
    return await _execute_tool(ctx, "GOOGLECALENDAR_REMOVE_ATTENDEE", **params)

# --- Server Execution ---
if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", 9002))

    print(f"Starting GCal MCP Server on http://{host}:{port}")
    mcp.run(transport="sse", host=host, port=port)