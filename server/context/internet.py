from server.context.base import BaseContextEngine
from server.context.runnables import get_internet_search_context_runnable
from server.common.runnables import *
from server.memory.runnables import *
from server.chat.functions import get_reframed_internet_query, get_search_results, get_search_summary
from datetime import datetime
import random

class InternetSearchContextEngine(BaseContextEngine):
    """Context Engine for processing internet search data based on user interests."""

    def __init__(self, *args, **kwargs):
        print("InternetSearchContextEngine.__init__ started")
        super().__init__(*args, **kwargs)
        self.category = "internet_search"
        print(f"InternetSearchContextEngine.__init__ - category set to: {self.category}")
        print("InternetSearchContextEngine.__init__ finished")
    
    async def start(self):
        """Start the engine, running periodically every hour."""
        print("BaseContextEngine.start started")
        while True:
            print("BaseContextEngine.start - running engine iteration")
            await self.run_engine()
            print("BaseContextEngine.start - engine iteration finished, sleeping for 3600 seconds")
            await asyncio.sleep(10800)  # Check every hour

    async def extract_interests(self, context):
        """
        Extract user interests from the provided context using an LLM runnable.
        
        Args:
            context (str): Unstructured text containing user interests.
        
        Returns:
            list: List of extracted interests (e.g., ['hiking', 'photography']).
        """
        runnable = get_interest_extraction_runnable()
        try:
            # Invoke the runnable with the context
            interests = runnable.invoke({"context": context})
            # Validate the output format
            if isinstance(interests, list) and all(isinstance(item, str) for item in interests):
                return interests
            else:
                print("Invalid interest format returned by LLM")
                return []
        except Exception as e:
            print(f"Error extracting interests: {e}")
            return []  # Fallback to empty list on error

    async def fetch_new_data(self):
        """Fetch new search results based on user interests."""
        # Retrieve user context from memory backend
        user_context = await self.memory_backend.retrieve_memory(self.user_id, "What are the user's interests?", "Long Term")
        
        print("USER CONTEXT", user_context)
        
        # Extract interests using the LLM-based method
        interests = await self.extract_interests(user_context)
        
        # Generate a search query based on interests
        if interests:
            selected_interest = random.choice(interests)  # Pick one interest randomly
            query = f"latest news in {selected_interest}"
        else:
            query = "latest news in technology"  # Fallback if no interests are found
        
        # Reframe the query and fetch results
        search_results = get_search_results(query)
        
        # Update context with timestamp
        self.context["internet_search"] = {
            "last_updated": datetime.utcnow().isoformat() + "Z",
            "search_results": search_results
        }
        await self.save_context()
        return search_results

    async def process_new_data(self, search_results):
        """Process search results into a summary with article links."""
        print("InternetSearchContextEngine.process_new_data started")
        internet_summary_runnable = get_internet_summary_runnable()
        summary = get_search_summary(internet_summary_runnable, search_results)
        # Enhance the summary with links (assuming search_results contains URLs)
        formatted_summary = f"Here’s a summary of recent articles:\n{summary}"
        print(f"InternetSearchContextEngine.process_new_data - generated summary: {formatted_summary}")
        print("InternetSearchContextEngine.process_new_data finished")
        return formatted_summary

    async def get_runnable(self):
        """Return the internet search-specific runnable."""
        print("InternetSearchContextEngine.get_runnable started")
        runnable = get_internet_search_context_runnable()
        print(f"InternetSearchContextEngine.get_runnable - returning runnable: {runnable}")
        print("InternetSearchContextEngine.get_runnable finished")
        return runnable

    async def get_category(self):
        """Return the memory category for internet search."""
        print("InternetSearchContextEngine.get_category started")
        print(f"InternetSearchContextEngine.get_category - returning category: {self.category}")
        print("InternetSearchContextEngine.get_category finished")
        return self.category