from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import base64
import os
import requests
from fastapi import HTTPException
import datetime
import platform

from dotenv import load_dotenv

load_dotenv("server/.env")  # Load environment variables from .env file

# Load configuration from environment variables.
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
MANAGEMENT_CLIENT_ID = os.getenv("AUTH0_MANAGEMENT_CLIENT_ID")
MANAGEMENT_CLIENT_SECRET = os.getenv("AUTH0_MANAGEMENT_CLIENT_SECRET")
# Convert hex strings from environment variables to bytes for cryptography.
SECRET_KEY = bytes.fromhex(os.environ.get("AES_SECRET_KEY"))
IV = bytes.fromhex(os.environ.get("AES_IV"))

# Define log file path based on the operating system.
if platform.system() == "Windows":
    log_file_path = os.path.join(
        os.getenv("PROGRAMDATA"), "Sentient", "logs", "fastapi-backend.log"
    )
else:
    log_file_path = os.path.join("/var", "log", "sentient", "fastapi-backend.log")


def aes_encrypt(data: str) -> str:
    """
    Encrypts the given string data using AES-CBC with PKCS7 padding.

    Args:
        data (str): The string data to be encrypted.

    Returns:
        str: Base64 encoded string of the encrypted data.
             Returns an error message string if encryption fails.
    """
    try:
        # Initialize backend, cipher, and encryptor for AES encryption.
        backend = default_backend()
        cipher = Cipher(algorithms.AES(SECRET_KEY), modes.CBC(IV), backend=backend)
        encryptor = cipher.encryptor()

        # Apply PKCS7 padding to the data before encryption.
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(data.encode()) + padder.finalize()

        # Encrypt the padded data and finalize the encryption process.
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        # Encode the encrypted data to Base64 for string representation and return.
        return base64.b64encode(encrypted).decode()
    except Exception as e:
        print(f"Error in aes-encryption: {str(e)}")
        return str(e)


def aes_decrypt(encrypted_data: str) -> str:
    """
    Decrypts AES-CBC encrypted data that is Base64 encoded and uses PKCS7 padding.

    Args:
        encrypted_data (str): Base64 encoded string of the encrypted data.

    Returns:
        str: The decrypted string data.
             Returns an error message string if decryption fails.
    """
    try:
        # Initialize backend, cipher, and decryptor for AES decryption.
        backend = default_backend()
        cipher = Cipher(algorithms.AES(SECRET_KEY), modes.CBC(IV), backend=backend)
        decryptor = cipher.decryptor()

        # Decode the Base64 encoded encrypted data to bytes.
        encrypted_bytes = base64.b64decode(encrypted_data)
        # Decrypt the data and finalize the decryption process.
        decrypted = decryptor.update(encrypted_bytes) + decryptor.finalize()

        # Remove PKCS7 padding from the decrypted data.
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        unpadded_data = unpadder.update(decrypted) + unpadder.finalize()
        # Decode the unpadded bytes to a string and return.
        return unpadded_data.decode()
    except Exception as e:
        print(f"Error in aes decryption: {str(e)}")
        return str(e)


def get_management_token() -> str:
    """
    Retrieves an access token for Auth0 Management API using client credentials grant.

    This function communicates with the Auth0 token endpoint to obtain a management API token.
    It uses the client ID and client secret configured in environment variables.

    Returns:
        str: Access token for Auth0 Management API.
             Raises HTTPException if token retrieval fails.
    Raises:
        HTTPException: If the request to Auth0 token endpoint fails.
    """
    url = f"https://{AUTH0_DOMAIN}/oauth/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": MANAGEMENT_CLIENT_ID,
        "client_secret": MANAGEMENT_CLIENT_SECRET,
        "audience": f"https://{AUTH0_DOMAIN}/api/v2/",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(url, data=payload, headers=headers)
    if response.status_code != 200:
        # Raise an HTTPException if the request to Auth0 fails.
        raise HTTPException(
            status_code=500, detail="Failed to retrieve management token."
        )
    # Return the access token from the response JSON.
    return response.json().get("access_token")


def write_to_log(message: str) -> None:
    """
    Writes a message to the log file with a timestamp.

    The log file path is determined based on the operating system.
    It creates the log directory if it doesn't exist and appends the message to the log file.

    Args:
        message (str): The message to be written to the log file.

    Returns:
        None: This function does not return any value.
    """
    # Get current timestamp in ISO format.
    timestamp = datetime.datetime.now().isoformat()
    log_message = f"{timestamp}: {message}\n"

    try:
        # Ensure the directory for log file exists, create if not.
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # Create the log file if it does not exist.
        if not os.path.exists(log_file_path):
            with open(log_file_path, "w") as f:
                pass  # Create an empty file

        # Open the log file in append mode and write the log message.
        with open(log_file_path, "a") as log_file:
            log_file.write(log_message)
    except Exception as error:
        print(f"Error writing to log file: {error}")
