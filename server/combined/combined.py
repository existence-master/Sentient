# Combine planner and executor here.

from server.planner.plan import generate_plan_text
from server.executor.executor import execute_plan_with_bot
import re

def combined_action_execution(action_items, available_tools, context_sources, user_id):
    """
    Combines planning and execution of action items using available tools and context sources.
    """
    print("\nAction Items:\n" + "\n".join(action_items))
    
    # Generate plan based on action items
    plan = generate_plan_text(action_items, available_tools, context_sources)
    
    extracted_plan = re.sub(r"<think>.*?</think>", "", plan[0]["content"], flags=re.DOTALL).strip()
    
    print("\nExecuting Plan:\n" + extracted_plan)
    
    # Execute the generated plan
    execution_results = execute_plan_with_bot(extracted_plan, user_id)
        
    return {
        'plan': plan,
        'execution_results': execution_results
    }

# Example usage (for testing locally)
# if __name__ == "__main__":
#     action_items = ["The user needs to prepare a presentation for the Quarterly Report meeting."]
#     available_tools = ["gmail", "gcalendar", "gdrive", "gdocs", "gslides", "gsheet", "get_short_term_memories", "get_long_term_memories", "notion", "web_search"]
#     context_sources = ["short_term_memories (short-term information about the user)", "long_term_memories (long-term information about the user)", "gdrive (user's Google Drive)", "notion (user's Notion workspace)"]
#     result = combined_action_execution(action_items, available_tools, context_sources, "testuser")
#     print(result)