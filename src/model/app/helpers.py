import os
import datetime
import platform
from typing import Dict, Any, Optional, Callable, List, Union
import base64
import re
from html import unescape
import importlib
import asyncio
import json

# Determine the log file path based on the operating system.
# On Windows, it will be in the ProgramData directory under Sentient/logs.
# On other systems (like Linux, macOS), it will be in /var/log/sentient/.
if platform.system() == "Windows":
    log_file_path = os.path.join(
        os.getenv("PROGRAMDATA"), "Sentient", "logs", "fastapi-backend.log"
    )
else:
    log_file_path = os.path.join("/var", "log", "sentient", "fastapi-backend.log")


def write_to_log(message):
    """
    Writes a message to the log file with a timestamp.

    This function takes a message string, adds a timestamp to it, and then
    writes the timestamped message to the log file specified by `log_file_path`.
    It also handles the creation of the log directory and the log file if they
    do not already exist.

    Args:
        message (str): The message to be written to the log file.
    """
    # Get the current timestamp in ISO format.
    timestamp = datetime.datetime.now().isoformat()
    # Format the log message to include the timestamp and the provided message.
    log_message = f"{timestamp}: {message}\n"

    try:
        # Ensure that the directory for the log file exists.
        # `os.makedirs` creates the directory and any necessary parent directories.
        # `exist_ok=True` prevents an error if the directory already exists.
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Check if the log file exists. If not, create it.
        # This ensures that if the file doesn't exist, it will be created before writing.
        if not os.path.exists(log_file_path):
            with open(log_file_path, "w") as f:
                pass  # 'pass' does nothing, just creates an empty file

        # Open the log file in append mode ('a').
        # This mode ensures that new messages are added to the end of the file,
        # preserving previous log entries.
        with open(log_file_path, "a") as log_file:
            # Write the formatted log message to the log file.
            log_file.write(log_message)
    except Exception as error:
        # If any exception occurs during the process (e.g., file permission issues),
        # print an error message to the console.
        print(f"Error writing to log file: {error}")


def extract_email_body(payload: Dict[str, Any]) -> str:
    """
    Extract the readable email body from a Gmail API message payload.

    This function recursively searches through the parts of a Gmail message payload to find
    and extract the email body, prioritizing 'text/plain' and then 'text/html' MIME types.
    It decodes the Base64 encoded content and returns the decoded text.

    Args:
        payload (Dict[str, Any]): The 'payload' part of a Gmail API message object.

    Returns:
        str: The extracted and decoded email body as a string.
             Returns "No body available." if no body is found or if there is an error during extraction.
    """
    try:
        if "parts" in payload:  # Check if payload has 'parts' (MIME multipart)
            for part in payload["parts"]:  # Iterate through each part
                if (
                    part["mimeType"] == "text/plain"
                ):  # Check if MIME type is 'text/plain'
                    return decode_base64(
                        part["body"].get("data", "")
                    )  # Decode and return plain text body
                elif (
                    part["mimeType"] == "text/html"
                ):  # Check if MIME type is 'text/html'
                    return decode_base64(
                        part["body"].get("data", "")
                    )  # Decode and return HTML body
        elif "body" in payload:  # Check if payload has a direct 'body' (not multipart)
            return decode_base64(
                payload["body"].get("data", "")
            )  # Decode and return body data
    except Exception as e:  # Catch any exceptions during body extraction
        print(f"Error extracting email body: {e}")

    return "No body available."  # Return default message if no body is available


def decode_base64(encoded_data: str) -> str:
    """
    Decode a Base64 encoded string, specifically for URL-safe Base64 in email bodies.

    Args:
        encoded_data (str): The Base64 encoded string to decode.

    Returns:
        str: The decoded string in UTF-8 format.
             Returns "Failed to decode body." if decoding fails.
    """
    try:
        if encoded_data:  # Check if encoded_data is not empty
            decoded_bytes: bytes = base64.urlsafe_b64decode(
                encoded_data
            )  # Decode from URL-safe Base64 to bytes
            return decoded_bytes.decode("utf-8")  # Decode bytes to UTF-8 string
    except Exception as e:  # Catch any exceptions during decoding
        print(f"Error decoding base64: {e}")
    return "Failed to decode body."  # Return default message if decoding fails

def clean_description(description: Optional[str]) -> str:
    """
    Clean a description by removing HTML tags and unescaping HTML entities.

    This function removes any HTML tags from the input description and unescapes
    HTML entities (e.g., '&' becomes '&').

    Args:
        description (Optional[str]): The description string, possibly containing HTML.

    Returns:
        str: The cleaned description string, with HTML tags removed and entities unescaped.
             Returns an empty string if the input description is None or empty.
    """
    if not description:
        return ""  # Return empty string if description is None or empty

    clean_text: str = re.sub(r"<.*?>", "", description)  # Remove HTML tags
    clean_text: str = unescape(clean_text)  # Unescape HTML entities
    return clean_text  # Return cleaned text


def get_function_from_agents(function_name: str) -> Optional[Callable]:
    """
    Dynamically import and retrieve a function from the 'functions' module.

    This function attempts to import a module named 'functions' and then retrieve
    a specific function from it by name. This is used to dynamically access tool functions.

    Args:
        function_name (str): The name of the function to retrieve.

    Returns:
        Optional[Callable]: The function if found and successfully imported, otherwise None.
    """
    try:
        functions_module = importlib.import_module(
            "functions"
        )  # Attempt to import the 'functions' module
        return getattr(
            functions_module, function_name, None
        )  # Get the function by name, return None if not found
    except ModuleNotFoundError:
        return None  # Return None if the 'functions' module is not found


def clean_key(key: str) -> str:
    """
    Clean a string by removing parentheses and extra whitespace.

    This function removes any text within parentheses (including the parentheses themselves)
    and strips leading/trailing whitespace from the given string.

    Args:
        key (str): The input string to clean.

    Returns:
        str: The cleaned string.
    """
    return re.sub(
        r"\s*\(.*?\)\s*", "", key
    ).strip()  # Remove parentheses and enclosed text, then strip whitespace


def clean_description(description: Optional[str]) -> str:
    """
    Clean a description by removing HTML tags and unescaping HTML entities.

    This function removes any HTML tags from the input description and unescapes
    HTML entities (e.g., '&' becomes '&').

    Args:
        description (Optional[str]): The description string, possibly containing HTML.

    Returns:
        str: The cleaned description string, with HTML tags removed and entities unescaped.
             Returns an empty string if the input description is None or empty.
    """
    if not description:
        return ""  # Return empty string if description is None or empty

    clean_text = re.sub(r"<.*?>", "", description)  # Remove HTML tags
    clean_text = unescape(clean_text)  # Unescape HTML entities
    return clean_text


class TimeoutException(Exception):
    """Custom exception class for timeout events."""

    pass


def watchdog(timeout_sec: int):
    """
    Raise a TimeoutError exception with a custom message indicating a timeout.

    Args:
        timeout_sec (int): The timeout duration in seconds.

    Raises:
        TimeoutError: Always raises a TimeoutError with a message indicating the timeout duration.
    """
    raise TimeoutError(
        "Query timed out after {} seconds.".format(timeout_sec)
    )  # Raise TimeoutError with a custom message


async def parse_and_execute_tool_calls(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a tool call dictionary and dynamically execute the corresponding function.

    This function takes a dictionary representing a tool call, extracts the function name and parameters,
    and then dynamically calls the function using the parameters. It handles both synchronous and
    asynchronous functions and catches various exceptions that might occur during execution.

    Args:
        tool_call (Dict[str, Any]): A dictionary containing the tool call details, expected to have:
                                     - "tool_name" (str): The name of the function to call.
                                     - "parameters" (Dict[str, Any]): A dictionary of parameters to pass to the function.

    Returns:
        Dict[str, Any]: A dictionary containing the result of the function call.
                         On success, it returns the result from the called function.
                         On failure, it returns a dictionary with:
                         {"status": "failure", "error": str(error)}
                         where error is a string describing the error.
    """
    try:
        function_name: Optional[str] = tool_call.get(
            "tool_name"
        )  # Get the tool name from the tool call dictionary
        params_dict: Dict[str, Any] = tool_call.get(
            "parameters", {}
        )  # Get the parameters dictionary, default to empty dict if not present

        if not function_name or not isinstance(
            params_dict, dict
        ):  # Validate function_name and params_dict
            raise ValueError(
                "Invalid tool call format."
            )  # Raise ValueError if tool call format is invalid

        function_to_call: Optional[Callable] = get_function_from_agents(
            function_name
        )  # Retrieve the function to call from agents module
        if function_to_call:  # Check if the function is found
            if asyncio.iscoroutinefunction(
                function_to_call
            ):  # Check if the function is asynchronous
                result = await function_to_call(
                    **params_dict
                )  # Await the asynchronous function call
            else:
                result = function_to_call(
                    **params_dict
                )  # Call the synchronous function

            return result  # Return the result of the function call
        else:
            raise NameError(
                f"Function '{function_name}' is not defined."
            )  # Raise NameError if function is not defined

    except json.JSONDecodeError as je:  # Catch JSONDecodeError
        print(f"JSONDecodeError: {je}")
        return {
            "status": "failure",
            "error": "Invalid JSON format.",
        }  # Return failure status with error message
    except ValueError as ve:  # Catch ValueError
        print(f"ValueError: {ve}")
        return {
            "status": "failure",
            "error": str(ve),
        }  # Return failure status with error message
    except NameError as ne:  # Catch NameError
        print(f"NameError: {ne}")
        return {
            "status": "failure",
            "error": str(ne),
        }  # Return failure status with error message
    except TypeError as te:  # Catch TypeError
        print(f"TypeError: {te}")
        return {
            "status": "failure",
            "error": str(te),
        }  # Return failure status with error message
    except Exception as e:  # Catch any other exceptions
        print(f"An unexpected error occurred: {e}")
        return {
            "status": "failure",
            "error": str(e),
        }  # Return failure status with error message


def extract_and_fix_json(json_string: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Extract and fix JSON from a potentially malformed string.

    This function attempts to extract valid JSON objects or arrays from a string that may contain
    malformed JSON. It sanitizes invalid characters, attempts to find and parse JSON matches,
    and if parsing fails, it tries to fix common syntax errors before attempting to parse again.

    Args:
        json_string (str): The input string that may contain JSON.

    Returns:
        Union[Dict[str, Any], List[Any]]: Parsed JSON object (dictionary) or array (list) if successful.
                                          Raises ValueError if JSON parsing fails even after fixing.

    Raises:
        ValueError: If no JSON object is found or if JSON parsing fails even after attempting to fix syntax.
    """
    try:
        sanitized_string: str = sanitize_invalid_characters(
            json_string
        )  # Sanitize invalid characters from JSON string

        json_matches: List[str] = re.findall(
            r"(\{.*\}|\[.*\])", sanitized_string, re.DOTALL
        )  # Find all JSON objects or arrays in the string
        if not json_matches:  # Check if any JSON matches were found
            raise ValueError(
                "No JSON object found in the input string."
            )  # Raise ValueError if no JSON object is found

        for match in json_matches:  # Iterate through each JSON match
            try:
                return json.loads(match)  # Attempt to parse the JSON match
            except json.JSONDecodeError as e:  # Catch JSONDecodeError if parsing fails
                print(f"Error in fixing JSON: {str(e)}")
                continue  # Continue to the next match if parsing fails

        fixed_json_string: str = fix_json_syntax(
            sanitized_string
        )  # Fix JSON syntax errors
        return json.loads(fixed_json_string)  # Attempt to parse the fixed JSON string

    except Exception as e:  # Catch any exceptions during JSON parsing
        print(f"Error parsing JSON: {str(e)}")
        raise ValueError(
            f"Failed to parse JSON: {e}"
        )  # Raise ValueError if JSON parsing fails


def fix_json_syntax(json_string: str) -> str:
    """
    Fix common JSON syntax issues such as missing brackets or braces.

    This function corrects common JSON syntax errors like unbalanced brackets or braces
    by counting the occurrences of opening and closing brackets and braces and appending
    or prepending the necessary characters to balance them.

    Args:
        json_string (str): The input JSON string that may have syntax errors.

    Returns:
        str: The JSON string with corrected syntax.
    """
    json_string = re.sub(
        r"^[^{\[]*", "", json_string
    )  # Remove any characters before the start of JSON object or array
    json_string = re.sub(
        r"[^}\]]*$", "", json_string
    )  # Remove any characters after the end of JSON object or array

    open_braces: int = json_string.count("{")  # Count number of opening braces
    close_braces: int = json_string.count("}")  # Count number of closing braces
    open_brackets: int = json_string.count("[")  # Count number of opening brackets
    close_brackets: int = json_string.count("]")  # Count number of closing brackets

    if (
        open_braces > close_braces
    ):  # Check if there are more opening braces than closing braces
        json_string += "}" * (
            open_braces - close_braces
        )  # Append missing closing braces
    elif (
        close_braces > open_braces
    ):  # Check if there are more closing braces than opening braces
        json_string = (
            "{" * (close_braces - open_braces) + json_string
        )  # Prepend missing opening braces

    if (
        open_brackets > close_brackets
    ):  # Check if there are more opening brackets than closing brackets
        json_string += "]" * (
            open_brackets - close_brackets
        )  # Append missing closing brackets
    elif (
        close_brackets > open_brackets
    ):  # Check if there are more closing brackets than opening brackets
        json_string = (
            "[" * (close_brackets - open_brackets) + json_string
        )  # Prepend missing opening brackets

    return json_string  # Return the JSON string with corrected syntax


def sanitize_invalid_characters(json_string: str) -> str:
    """
    Remove invalid control characters from a JSON string.

    This function removes characters that are not allowed in JSON strings,
    specifically control characters in the ranges U+0000 to U+001F and U+007F.

    Args:
        json_string (str): The input JSON string that may contain invalid characters.

    Returns:
        str: The sanitized JSON string with invalid characters removed.
    """
    invalid_chars_pattern = (
        r"[\x00-\x1F\x7F]"  # Define regex pattern for invalid control characters
    )
    return re.sub(
        invalid_chars_pattern, "", json_string
    )  # Replace invalid characters with empty string

def write_to_log(message: str):
    """
    Write a timestamped message to a log file.

    This function writes a log message to a predefined log file, prepending the current timestamp
    in ISO format. It also handles directory creation if the log directory does not exist
    and creates the log file if it does not exist.

    Args:
        message (str): The message string to write to the log file.
    """
    timestamp: str = (
        datetime.datetime.now().isoformat()
    )  # Generate ISO format timestamp
    log_message: str = f"{timestamp}: {message}\n"  # Format log message with timestamp

    try:
        os.makedirs(
            os.path.dirname(log_file_path), exist_ok=True
        )  # Ensure log directory exists, create if not

        if not os.path.exists(log_file_path):  # Check if log file exists
            with open(log_file_path, "w") as f:  # Create log file if it doesn't exist
                pass  # Do nothing, just create the file

        with open(log_file_path, "a") as log_file:  # Open log file in append mode
            log_file.write(log_message)  # Write the log message to the file
    except Exception as error:  # Catch any exceptions during log writing
        print(
            f"Error writing to log file: {error}"
        )  # Print error message if writing to log file fails