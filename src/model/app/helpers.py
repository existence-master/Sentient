import os
import datetime
import platform

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