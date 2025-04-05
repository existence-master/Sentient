from typing import Dict, Any
import base64
import json
import importlib
import asyncio
from typing import Callable, Optional

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
            "model.agents.functions"
        )  # Attempt to import the 'functions' module
        return getattr(
            functions_module, function_name, None
        )  # Get the function by name, return None if not found
    except ModuleNotFoundError:
        return None  # Return None if the 'functions' module is not found

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