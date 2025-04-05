from typing import Any, Callable, Dict, Optional
import importlib
import re 
from html import unescape
import asyncio
import json

def get_function_from_agents(function_name: str) -> Optional[Callable]:
    """
    Dynamically import and retrieve a function from the 'agents' module.

    This function attempts to import the 'agents' module and then retrieve a specific function
    from it by name. This allows for late-binding of function calls, especially useful
    in plugin or agent-based architectures where function availability might be dynamic.

    Args:
        function_name (str): The name of the function to retrieve from the 'agents' module.

    Returns:
        Optional[Callable]: The function if found and successfully imported, otherwise None.
    """
    try:
        agents_module = importlib.import_module("agents")
        return getattr(agents_module, function_name, None)
    except ModuleNotFoundError:
        return None


def clean_key(key: str) -> str:
    """
    Clean a string key by removing parenthetical parts and extra whitespace.

    This function uses regular expressions to remove any text within parentheses,
    as well as the parentheses themselves, from the input string. It also strips
    leading and trailing whitespace to produce a clean key string. This is useful
    for processing keys that might contain descriptions or annotations in parentheses.

    Args:
        key (str): The input string key to be cleaned.

    Returns:
        str: The cleaned string key with parenthetical parts removed and whitespace trimmed.
    """
    return re.sub(r'\s*\(.*?\)\s*', '', key).strip()


def clean_description(description: Optional[str]) -> str:
    """
    Clean an HTML description by removing HTML tags and unescaping HTML entities.

    This function processes an HTML description string to remove any HTML tags present
    and unescape HTML character entities (like & to &). This results in a plain
    text description that is more readable and easier to process. If the input description
    is None, it returns an empty string.

    Args:
        description (Optional[str]): The HTML description string to be cleaned, or None.

    Returns:
        str: The cleaned plain text description. Returns an empty string if the input is None or empty.
    """
    if not description:
        return ""
    clean_text = re.sub(r'<.*?>', '', description) # Remove HTML tags
    clean_text = unescape(clean_text) # Unescape HTML entities
    return clean_text

class TimeoutException(Exception):
    """
    Custom exception raised when a timeout occurs.

    This exception can be used to signal that a process or operation has exceeded
    a predefined time limit. It's useful for implementing watchdog timers or
    handling operations that should not run indefinitely.
    """
    pass

def watchdog(timeout_sec: int):
    """
    Function to raise a TimeoutError after a specified number of seconds (not actually asynchronous).

    Note: This function as written will immediately raise a TimeoutError when called.
    It is not designed to run in the background and trigger a timeout after a delay.
    For asynchronous timeouts, consider using asyncio.wait_for.

    Args:
        timeout_sec (int): The timeout period in seconds. This parameter is not used to introduce a delay;
                           it's just included in the error message.

    Raises:
        TimeoutError: Always raised immediately when this function is called, indicating a timeout.
    """
    raise TimeoutError("Query timed out after {} seconds.".format(timeout_sec))


async def parse_and_execute_tool_calls(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a tool call dictionary, dynamically find and execute the function, and return the result.

    This asynchronous function takes a dictionary representing a tool call, extracts the function name
    and parameters, dynamically retrieves the function using `get_function_from_agents`, and then
    executes the function with the provided parameters. It handles both synchronous and asynchronous
    functions and includes comprehensive error handling for various potential issues like invalid
    JSON format, undefined functions, and type errors during execution.

    Args:
        tool_call (Dict[str, Any]): A dictionary containing the tool call details, expected to have
                                     'tool_name' and 'parameters' keys. 'parameters' should be a dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing the result of the function call. In case of success,
                         it returns the function's result directly. In case of failure, it returns a
                         dictionary with 'status': 'failure' and 'error': error message.
    """
    try:
        function_name = tool_call.get("tool_name") # Extract function name from tool call
        params_dict = tool_call.get("parameters", {}) # Extract parameters from tool call, default to empty dict if not present

        if not function_name or not isinstance(params_dict, dict):
            raise ValueError("Invalid tool call format.") # Raise ValueError if tool call format is invalid

        function_to_call = get_function_from_agents(function_name) # Dynamically get function from agents module
        if function_to_call:
            if asyncio.iscoroutinefunction(function_to_call):
                result = await function_to_call(**params_dict) # Await execution for async functions
            else:
                result = function_to_call(**params_dict) # Execute sync functions directly

            return result # Return result of the function call
        else:
            raise NameError(f"Function '{function_name}' is not defined.") # Raise NameError if function is not found

    except json.JSONDecodeError as je:
        print(f"JSONDecodeError: {je}")
        return {"status": "failure", "error": "Invalid JSON format."} # Handle JSON decode errors
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return {"status": "failure", "error": str(ve)} # Handle ValueErrors
    except NameError as ne:
        print(f"NameError: {ne}")
        return {"status": "failure", "error": str(ne)} # Handle NameErrors (function not found)
    except TypeError as te:
        print(f"TypeError: {te}")
        return {"status": "failure", "error": str(te)} # Handle TypeErrors during function execution
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"status": "failure", "error": str(e)} # Catch-all for unexpected errors


def extract_and_fix_json(json_string: str) -> Any:
    """
    Extract and attempt to fix JSON from a string that may contain malformed JSON.

    This function searches for JSON objects or arrays within a string, attempts to parse them,
    and if parsing fails, tries to fix common JSON syntax errors. It first sanitizes the input
    string by removing invalid characters, then uses regular expressions to find potential JSON
    objects or arrays. If initial parsing fails for all found JSON candidates, it applies
    syntax fixes and attempts to parse again.

    Args:
        json_string (str): The input string that might contain JSON.

    Returns:
        Any: The parsed JSON object (dict or list) if successful, otherwise raises a ValueError.

    Raises:
        ValueError: If JSON parsing fails even after attempting to fix syntax errors.
    """
    try:
        sanitized_string = sanitize_invalid_characters(json_string) # Remove invalid control characters

        json_matches = re.findall(r"(\{.*\}|\[.*\])", sanitized_string, re.DOTALL) # Find all potential JSON objects or arrays
        if not json_matches:
            raise ValueError("No JSON object found in the input string.") # Raise error if no JSON-like structure is found

        for match in json_matches:
            try:
                return json.loads(match) # Attempt to load each match as JSON
            except json.JSONDecodeError as e:
                print(f"Error in fixing JSON: {str(e)}")
                continue # If parsing fails, continue to the next match

        fixed_json_string = fix_json_syntax(sanitized_string) # Apply syntax fixes to the whole sanitized string
        return json.loads(fixed_json_string) # Attempt to parse the fixed JSON string

    except Exception as e:
        print(f"Error parsing JSON: {str(e)}")
        raise ValueError(f"Failed to parse JSON: {e}") # Raise ValueError if parsing fails after all attempts


def fix_json_syntax(json_string: str) -> str:
    """
    Attempt to fix common JSON syntax issues in a string, such as unbalanced brackets.

    This function tries to correct common syntax errors in JSON strings, specifically focusing
    on ensuring balanced curly braces '{}' and square brackets '[]'. It counts the number of
    opening and closing brackets of each type and adds or removes closing brackets as needed
    to balance them. It also removes any characters before the first bracket and after the last bracket
    to isolate the JSON structure.

    Args:
        json_string (str): The potentially malformed JSON string.

    Returns:
        str: The string with attempted JSON syntax fixes applied. Note that this does not guarantee
             valid JSON, but increases the chances of successful parsing for common errors.
    """
    json_string = re.sub(r"^[^{\[]*", "", json_string) # Remove anything before the start of JSON object/array
    json_string = re.sub(r"[^}\]]*$", "", json_string) # Remove anything after the end of JSON object/array

    open_braces = json_string.count("{")
    close_braces = json_string.count("}")
    open_brackets = json_string.count("[")
    close_brackets = json_string.count("]")

    # Balance curly braces
    if open_braces > close_braces:
        json_string += "}" * (open_braces - close_braces) # Add missing closing braces
    elif close_braces > open_braces:
        json_string = "{" * (close_braces - open_braces) + json_string # Add missing opening braces

    # Balance square brackets
    if open_brackets > close_brackets:
        json_string += "]" * (open_brackets - close_brackets) # Add missing closing brackets
    elif close_brackets > open_brackets:
        json_string = "[" * (close_brackets - open_brackets) + json_string # Add missing opening brackets

    return json_string # Return the syntax-fixed JSON string


def sanitize_invalid_characters(json_string: str) -> str:
    """
    Remove invalid control characters from a JSON string to ensure it is parseable.

    JSON strings must not contain control characters (U+0000 through U+001F). This function
    uses a regular expression to find and remove these invalid characters from the input string,
    making it more likely to be successfully parsed as JSON.

    Args:
        json_string (str): The input JSON string that might contain invalid control characters.

    Returns:
        str: The sanitized JSON string with invalid control characters removed.
    """
    invalid_chars_pattern = r"[\x00-\x1F\x7F]" # Regex pattern to match control characters
    return re.sub(invalid_chars_pattern, "", json_string) # Replace invalid characters with empty string (effectively removing them)