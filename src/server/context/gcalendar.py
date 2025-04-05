from server.context.base import BaseContextEngine
from server.agents.functions import authenticate_calendar
from server.context.runnables import get_gcalendar_context_runnable
from datetime import datetime, timedelta
from dateutil import parser
import asyncio

class GCalendarContextEngine(BaseContextEngine):
    """Context Engine for processing Google Calendar data, tracking recent events."""

    def __init__(self, *args, **kwargs):
        print("GCalendarContextEngine.__init__ started")
        super().__init__(*args, **kwargs)
        print("GCalendarContextEngine.__init__ - calling authenticate_gcalendar()")
        self.gcalendar_service = authenticate_calendar()
        print("GCalendarContextEngine.__init__ - gcalendar_service authenticated")
        self.category = "gcalendar"
        print(f"GCalendarContextEngine.__init__ - category set to: {self.category}")
        # Initialize context structure if it doesn't exist
        if self.category not in self.context:
             self.context[self.category] = {"last_event_ids": [], "last_fetched": None}
             print(f"GCalendarContextEngine.__init__ - initialized context for category '{self.category}'")
        print("GCalendarContextEngine.__init__ finished")

    async def start(self):
        """Start the engine, running periodically every hour."""
        print("GCalendarContextEngine.start started")
        while True:
            print("GCalendarContextEngine.start - running engine iteration")
            try:
                await self.run_engine()
                print("GCalendarContextEngine.start - engine iteration finished successfully")
            except Exception as e:
                # Catch exceptions during the run_engine cycle to prevent the loop from stopping
                print(f"GCalendarContextEngine.start - ERROR during engine iteration: {e}")
                import traceback
                traceback.print_exc() # Log the full traceback for debugging
            finally:
                # Ensure sleep happens even if there's an error
                print("GCalendarContextEngine.start - sleeping for 3600 seconds")
                await asyncio.sleep(10800)  # Check every hour

    async def fetch_new_data(self):
        """
        Fetch upcoming events from Google Calendar for the next 24 hours.
        Compares fetched events with the last known events to return only new ones.
        Updates the context with the IDs of the latest 5 fetched events.
        """
        print("GCalendarContextEngine.fetch_new_data started")

        # 1. Load Previous Event IDs from context
        print("GCalendarContextEngine.fetch_new_data - loading previous event IDs from context")
        gcalendar_context = self.context.get(self.category, {"last_event_ids": [], "last_fetched": None})
        # Use a set for efficient checking of previously seen IDs
        last_event_ids_set = set(gcalendar_context.get("last_event_ids", []))
        print(f"GCalendarContextEngine.fetch_new_data - loaded {len(last_event_ids_set)} previous event IDs.")

        # 2. Fetch Current Events
        now = datetime.utcnow()
        now_iso = now.isoformat() + "Z"
        time_max_iso = (now + timedelta(days=1)).isoformat() + "Z"
        print(f"GCalendarContextEngine.fetch_new_data - fetching events from {now_iso} to {time_max_iso}")
        try:
            events_result = self.gcalendar_service.events().list(
                calendarId="primary",
                timeMin=now_iso,
                timeMax=time_max_iso,
                maxResults=10, # Fetch a few more than 5 just in case some are filtered out later
                singleEvents=True,
                orderBy="startTime"
            ).execute()
            fetched_events = events_result.get("items", [])
            print(f"GCalendarContextEngine.fetch_new_data - fetched {len(fetched_events)} events from API")
        except Exception as e:
            print(f"GCalendarContextEngine.fetch_new_data - ERROR fetching events from Google Calendar API: {e}")
            # Decide how to handle API errors: return empty, raise exception, etc.
            # Returning empty prevents crashing the loop but might miss updates.
            fetched_events = [] # Return empty list on API error for now

        # 3. Identify New Events
        new_events = []
        current_event_ids = [] # Store IDs from this fetch in order
        for event in fetched_events:
            event_id = event.get("id")
            if not event_id:
                print("GCalendarContextEngine.fetch_new_data - WARNING: Event found without an ID, skipping.")
                continue # Skip events without an ID

            current_event_ids.append(event_id) # Keep track of all valid IDs fetched this time

            if event_id not in last_event_ids_set:
                print(f"GCalendarContextEngine.fetch_new_data - Found NEW event: ID={event_id}, Summary='{event.get('summary', 'N/A')}'")
                new_events.append(event)
            # else: # Optional: log if you want to see old events being filtered out
            #     print(f"GCalendarContextEngine.fetch_new_data - Found OLD event (already seen): ID={event_id}")

        print(f"GCalendarContextEngine.fetch_new_data - identified {len(new_events)} new events out of {len(fetched_events)} fetched")

        # 4. Update Stored Event IDs (store the last 5 IDs from the *current* fetch)
        # We store the IDs from the current fetch to handle deletions/changes correctly.
        # The `current_event_ids` are already ordered by start time due to `orderBy="startTime"`
        updated_last_event_ids = current_event_ids[-5:] # Get the IDs of the last up to 5 events fetched
        print(f"GCalendarContextEngine.fetch_new_data - updating context with last {len(updated_last_event_ids)} event IDs from this fetch: {updated_last_event_ids}")

        # 5. Update Context and Save
        self.context[self.category] = {
            "last_fetched": now_iso,
            "last_event_ids": updated_last_event_ids # Store the list of the latest IDs
        }
        print("GCalendarContextEngine.fetch_new_data - saving updated context")
        await self.save_context() # Assuming BaseContextEngine provides this async method

        # 6. Return New Events
        print(f"GCalendarContextEngine.fetch_new_data - returning {len(new_events)} new events for processing")
        print("GCalendarContextEngine.fetch_new_data finished")
        return new_events # Only return events that weren't in the last known set

    async def process_new_data(self, new_events):
        """Process new calendar events into summaries."""
        print("GCalendarContextEngine.process_new_data started")
        if not new_events:
            print("GCalendarContextEngine.process_new_data - no new events to process")
            print("GCalendarContextEngine.process_new_data finished")
            return "" # Return empty string if no new events

        print(f"GCalendarContextEngine.process_new_data - processing {len(new_events)} new events")
        summaries = []
        for event in new_events:
            event_summary = event.get("summary", "No title")
            try:
                # Handle both date and dateTime formats
                start_str = event["start"].get("dateTime", event["start"].get("date"))
                if not start_str:
                     print(f"GCalendarContextEngine.process_new_data - WARNING: Event ID {event.get('id')} has no start time, skipping.")
                     continue
                start_time = parser.parse(start_str)
                # Format differently based on whether it's a full day event
                if "date" in event["start"] and "dateTime" not in event["start"]:
                     start_time_str = start_time.strftime("%Y-%m-%d (All day)")
                else:
                     start_time_str = start_time.strftime("%Y-%m-%d %H:%M")

                summary = f"You have an event '{event_summary}' at {start_time_str}"
                summaries.append(summary)
                print(f"GCalendarContextEngine.process_new_data - generated summary: {summary}")
            except Exception as e:
                print(f"GCalendarContextEngine.process_new_data - ERROR processing event {event.get('id')}: {e}")
                # Optionally skip this event or add an error message

        joined_summaries = "\n".join(summaries)
        print(f"GCalendarContextEngine.process_new_data - joined summaries: {joined_summaries}")
        print("GCalendarContextEngine.process_new_data finished")
        return joined_summaries

    async def get_runnable(self):
        """Return the GCalendar-specific runnable."""
        print("GCalendarContextEngine.get_runnable started")
        runnable = get_gcalendar_context_runnable()
        print(f"GCalendarContextEngine.get_runnable - returning runnable type: {type(runnable)}")
        print("GCalendarContextEngine.get_runnable finished")
        return runnable

    async def get_category(self):
        """Return the memory category for GCalendar."""
        print("GCalendarContextEngine.get_category started")
        print(f"GCalendarContextEngine.get_category - returning category: {self.category}")
        print("GCalendarContextEngine.get_category finished")
        return self.category