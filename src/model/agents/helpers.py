from typing import Dict, Any
import base64

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
