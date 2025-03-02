from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from helpers import *
import uvicorn
import requests
import nest_asyncio
import keyring

from dotenv import load_dotenv

load_dotenv("../.env")  # Load environment variables from .env file

# Load Auth0 configuration from environment variables.
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
MANAGEMENT_CLIENT_ID = os.getenv("AUTH0_MANAGEMENT_CLIENT_ID")
MANAGEMENT_CLIENT_SECRET = os.getenv("AUTH0_MANAGEMENT_CLIENT_SECRET")

# Initialize FastAPI application.
app = FastAPI(docs_url=None, redoc_url=None)  # Disable default docs and redoc endpoints

# Add CORS middleware to allow cross-origin requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Define Pydantic models for request bodies.
class EncryptionRequest(BaseModel):
    """
    Request model for encrypting data.
    """

    data: str


class DecryptionRequest(BaseModel):
    """
    Request model for decrypting data.
    """

    encrypted_data: str


class UserInfoRequest(BaseModel):
    """
    Request model for user information requests.
    """

    user_id: str


class ReferrerStatusRequest(BaseModel):
    """
    Request model for setting referrer status.
    """

    user_id: str
    referrer_status: bool


class BetaUserStatusRequest(BaseModel):
    """
    Request model for setting beta user status.
    """

    user_id: str
    beta_user_status: bool


class SetReferrerRequest(BaseModel):
    """
    Request model for setting referrer based on referral code.
    """

    referral_code: str

class SetApiKeyRequest(BaseModel):
    provider: str
    api_key: str

class HasApiKeyRequest(BaseModel):
    provider: str
    
class DeleteApiKeyRequest(BaseModel):
    provider: str

nest_asyncio.apply()  # Apply nest_asyncio for running async operations in sync context


@app.get("/")
async def main() -> JSONResponse:
    """
    Root endpoint of the API.

    Returns:
        JSONResponse: A simple welcome message.
    """
    return JSONResponse(
        status_code=200,
        content={
            "message": "Hello, I am Sentient, your private, decentralized and interactive AI companion who feels human"
        },
    )


@app.post("/initiate", status_code=200)
async def initiate() -> JSONResponse:
    """
    Endpoint to initiate the model (currently a placeholder).

    Returns:
        JSONResponse: Success or error message indicating initiation status.
    """
    try:
        return JSONResponse(
            status_code=200, content={"message": "Model initiated successfully"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.post("/get-role")
async def get_role(request: UserInfoRequest) -> JSONResponse:
    """
    Retrieves the role of a user from Auth0.

    Args:
        request (UserInfoRequest): Request containing the user_id.

    Returns:
        JSONResponse: User's role or error message.
    """
    try:
        management_api_access_token = (
            get_management_token()
        )  # Obtain management API token

        user_id = request.user_id
        roles_response = requests.get(
            f"https://{AUTH0_DOMAIN}/api/v2/users/{user_id}/roles",
            headers={
                "Authorization": f"Bearer {management_api_access_token}"
            },  # Include token in header
        )

        if roles_response.status_code != 200:
            raise HTTPException(
                status_code=roles_response.status_code,
                detail=f"Error fetching user roles: {roles_response.text}",  # Raise HTTP exception if request fails
            )

        roles = roles_response.json()
        if not roles or len(roles) == 0:
            return JSONResponse(
                status_code=404, content={"message": "No roles found for user."}
            )  # Return 404 if no roles are found

        return JSONResponse(
            status_code=200, content={"role": roles[0]["name"].lower()}
        )  # Return the first role name in lowercase

    except Exception as e:
        print(f"Error in get-role: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions


@app.post("/get-beta-user-status")
def get_beta_user_status(request: UserInfoRequest) -> JSONResponse:
    """
    Retrieves the beta user status from Auth0 app_metadata.

    Args:
        request (UserInfoRequest): Request containing the user_id.

    Returns:
        JSONResponse: Beta user status or error message.
    """
    try:
        token = get_management_token()  # Obtain management API token
        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {
            "Authorization": f"Bearer {token}",  # Include token in header
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error fetching user info: {response.text}",  # Raise HTTP exception if request fails
            )

        user_data = response.json()
        beta_user_status = user_data.get("app_metadata", {}).get(
            "betaUser"
        )  # Extract betaUser status from app_metadata

        if beta_user_status is None:
            return JSONResponse(
                status_code=404, content={"message": "Beta user status not found."}
            )  # Return 404 if beta user status not found

        return JSONResponse(
            status_code=200, content={"betaUserStatus": beta_user_status}
        )  # Return beta user status
    except Exception as e:
        print(f"Error in beta-user-status: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions


@app.post("/get-referral-code")
async def get_referral_code(request: UserInfoRequest) -> JSONResponse:
    """
    Retrieves the referral code from Auth0 app_metadata.

    Args:
        request (UserInfoRequest): Request containing the user_id.

    Returns:
        JSONResponse: Referral code or error message.
    """
    try:
        token = get_management_token()  # Obtain management API token
        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {
            "Authorization": f"Bearer {token}",  # Include token in header
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error fetching user info: {response.text}",  # Raise HTTP exception if request fails
            )

        user_data = response.json()
        referral_code = user_data.get("app_metadata", {}).get(
            "referralCode"
        )  # Extract referralCode from app_metadata
        if not referral_code:
            return JSONResponse(
                status_code=404, content={"message": "Referral code not found."}
            )  # Return 404 if referral code not found

        return JSONResponse(
            status_code=200, content={"referralCode": referral_code}
        )  # Return referral code
    except Exception as e:
        print(f"Error in get-referral-code: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions


@app.post("/get-referrer-status")
async def get_referrer_status(request: UserInfoRequest) -> JSONResponse:
    """
    Retrieves the referrer status from Auth0 app_metadata.

    Args:
        request (UserInfoRequest): Request containing the user_id.

    Returns:
        JSONResponse: Referrer status or error message.
    """
    try:
        token = get_management_token()  # Obtain management API token
        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {
            "Authorization": f"Bearer {token}",  # Include token in header
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error fetching user info: {response.text}",  # Raise HTTP exception if request fails
            )

        user_data = response.json()
        referrer_status = user_data.get("app_metadata", {}).get(
            "referrer"
        )  # Extract referrer status from app_metadata
        if referrer_status is None:
            return JSONResponse(
                status_code=404, content={"message": "Referrer status not found."}
            )  # Return 404 if referrer status not found

        return JSONResponse(
            status_code=200, content={"referrerStatus": referrer_status}
        )  # Return referrer status
    except Exception as e:
        print(f"Error in referrer-status: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions


@app.post("/set-referrer-status")
async def set_referrer_status(request: ReferrerStatusRequest) -> JSONResponse:
    """
    Sets the referrer status in Auth0 app_metadata.

    Args:
        request (ReferrerStatusRequest): Request containing user_id and referrer_status.

    Returns:
        JSONResponse: Success or error message.
    """
    try:
        token = get_management_token()  # Obtain management API token
        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {
            "Authorization": f"Bearer {token}",  # Include token in header
            "Content-Type": "application/json",
        }

        payload = {
            "app_metadata": {
                "referrer": request.referrer_status  # Set referrer status in app_metadata
            }
        }

        response = requests.patch(
            url, headers=headers, json=payload
        )  # Use PATCH to update user metadata
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error updating referrer status: {response.text}",  # Raise HTTP exception if request fails
            )

        return JSONResponse(
            status_code=200,
            content={"message": "Referrer status updated successfully."},
        )  # Return success message
    except Exception as e:
        print(f"Error in set-referrer-status: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions


@app.post("/get-user-and-set-referrer-status")
async def get_user_and_set_referrer_status(request: SetReferrerRequest) -> JSONResponse:
    """
    Searches for a user by referral code and sets their referrer status to true.

    Args:
        request (SetReferrerRequest): Request containing referral_code.

    Returns:
        JSONResponse: Success or error message.
    """
    try:
        token = get_management_token()  # Obtain management API token
        headers = {
            "Authorization": f"Bearer {token}",  # Include token in header
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        search_url = f"https://{AUTH0_DOMAIN}/api/v2/users?q=app_metadata.referralCode%3A%22{request.referral_code}%22"  # Search URL to find user by referral code
        search_response = requests.get(search_url, headers=headers)

        if search_response.status_code != 200:
            raise HTTPException(
                status_code=search_response.status_code,
                detail=f"Error searching for user: {search_response.text}",  # Raise HTTP exception if search fails
            )

        users = search_response.json()

        if not users or len(users) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No user found with referral code: {request.referral_code}",  # Raise 404 if no user found
            )

        user_id = users[0]["user_id"]  # Get user_id from search results

        referrer_status_payload = {"user_id": user_id, "referrer_status": True}

        set_status_url = f"http://localhost:5005/set-referrer-status"  # URL to set referrer status (assuming local service)
        set_status_response = requests.post(
            set_status_url, json=referrer_status_payload
        )  # Call local service to set referrer status

        if set_status_response.status_code != 200:
            raise HTTPException(
                status_code=set_status_response.status_code,
                detail=f"Error setting referrer status: {set_status_response.text}",  # Raise HTTP exception if setting status fails
            )

        return JSONResponse(
            status_code=200,
            content={"message": "Referrer status updated successfully."},
        )  # Return success message

    except Exception as e:
        print(f"Error in get-user-and-set-referrer-status: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions


@app.post("/set-beta-user-status")
def set_beta_user_status(request: BetaUserStatusRequest) -> JSONResponse:
    """
    Sets the beta user status in Auth0 app_metadata.

    Args:
        request (BetaUserStatusRequest): Request containing user_id and beta_user_status.

    Returns:
        JSONResponse: Success or error message.
    """
    try:
        token = get_management_token()  # Obtain management API token
        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {
            "Authorization": f"Bearer {token}",  # Include token in header
            "Content-Type": "application/json",
        }

        payload = {
            "app_metadata": {
                "betaUser": request.beta_user_status  # Set betaUser status in app_metadata
            }
        }

        response = requests.patch(
            url, headers=headers, json=payload
        )  # Use PATCH to update user metadata
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error updating beta user status: {response.text}",  # Raise HTTP exception if request fails
            )

        return JSONResponse(
            status_code=200,
            content={"message": "Beta user status updated successfully."},
        )  # Return success message
    except Exception as e:
        print(f"Error in set-beta-user-status: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions


@app.post("/get-user-and-invert-beta-user-status")
def get_user_and_invert_beta_user_status(request: UserInfoRequest) -> JSONResponse:
    """
    Searches for a user by user id and inverts the beta user status in Auth0 app_metadata.

    Args:
        request (UserInfoRequest): Request containing user_id.

    Returns:
        JSONResponse: Success or error message.
    """
    try:
        token = get_management_token()  # Obtain management API token
        url = f"https://{AUTH0_DOMAIN}/api/v2/users/{request.user_id}"
        headers = {
            "Authorization": f"Bearer {token}",  # Include token in header
            "Accept": "application/json",
        }

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error fetching user info: {response.text}",  # Raise HTTP exception if request fails
            )

        user_data = response.json()
        beta_user_status = user_data.get("app_metadata", {}).get(
            "betaUser"
        )  # Get current betaUser status

        # Invert the beta user status (string boolean to boolean and then invert)
        beta_user_status_payload = {
            "user_id": request.user_id,
            "beta_user_status": False
            if str(beta_user_status).lower() == "true"
            else True,
        }

        set_status_url = f"http://localhost:5005/set-beta-user-status"  # URL to set beta user status (assuming local service)
        set_status_response = requests.post(
            set_status_url, json=beta_user_status_payload
        )  # Call local service to set inverted beta user status

        if set_status_response.status_code != 200:
            raise HTTPException(
                status_code=set_status_response.status_code,
                detail=f"Error inverting beta user status: {set_status_response.text}",  # Raise HTTP exception if setting status fails
            )

        return JSONResponse(
            status_code=200,
            content={"message": "Beta user status inverted successfully."},
        )  # Return success message
    except Exception as e:
        print(f"Error in get-user-and-invert-beta-user-status: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions


@app.post("/encrypt")
async def encrypt_data(request: EncryptionRequest) -> JSONResponse:
    """
    Encrypts the provided data using AES encryption.

    Args:
        request (EncryptionRequest): Request containing the data to encrypt.

    Returns:
        JSONResponse: Encrypted data or error message.
    """
    try:
        data = request.data
        encrypted_data = aes_encrypt(
            data
        )  # Encrypt data using aes_encrypt helper function
        return JSONResponse(
            status_code=200, content={"encrypted_data": encrypted_data}
        )  # Return encrypted data
    except Exception as e:
        print(f"Error in encrypt: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions


@app.post("/decrypt")
async def decrypt_data(request: DecryptionRequest) -> JSONResponse:
    """
    Decrypts the provided encrypted data using AES decryption.

    Args:
        request (DecryptionRequest): Request containing the encrypted data to decrypt.

    Returns:
        JSONResponse: Decrypted data or error message.
    """
    try:
        data = request.encrypted_data
        decrypted_data = aes_decrypt(
            data
        )  # Decrypt data using aes_decrypt helper function
        return JSONResponse(
            status_code=200, content={"decrypted_data": decrypted_data}
        )  # Return decrypted data
    except Exception as e:
        print(f"Error in decrypt: {str(e)}")
        return JSONResponse(
            status_code=500, content={"message": str(e)}
        )  # Return 500 for any exceptions

# Endpoint to store an encrypted API key in keyring
@app.post("/set-api-key")
async def set_api_key(request: SetApiKeyRequest):
    """
    Encrypts the provided API key and stores it in keyring for the given provider.

    Args:
        request (SetApiKeyRequest): Request containing the provider and API key.

    Returns:
        dict: Success status or raises an HTTPException on failure.
    """
    try:
        # Encrypt the API key using the existing aes_encrypt function
        encrypted_key = aes_encrypt(request.api_key)
        # Store the encrypted key in keyring with "Sentient" as the service name
        keyring.set_password("electron-openid-oauth", request.provider, encrypted_key)
        return JSONResponse(
            status_code=200, content={"success": True}
        )  
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing API key: {str(e)}")

# Endpoint to check if an API key exists for a provider
@app.post("/has-api-key")
async def has_api_key(request: HasApiKeyRequest):
    """
    Checks if an encrypted API key exists in keyring for the given provider.

    Args:
        request (HasApiKeyRequest): Request containing the provider.

    Returns:
        dict: Whether the key exists or raises an HTTPException on failure.
    """
    try:
        # Retrieve the encrypted key from keyring
        encrypted_key = keyring.get_password("electron-openid-oauth", request.provider)
        # Return true if a key exists, false otherwise
        return JSONResponse(
            status_code=200, content={"exists": encrypted_key is not None})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking API key: {str(e)}")

# Endpoint to get providers with stored API keys
@app.get("/get-stored-providers")
async def get_stored_providers():
    """
    Returns a dictionary of known providers and whether they have API keys stored in the keyring.
    
    Returns:
        JSONResponse: Dictionary with provider names as keys and boolean values indicating key presence.
    """
    try:
        providers = ["openai", "gemini", "claude"]
        stored_status = {
            provider: keyring.get_password("electron-openid-oauth", provider) is not None
            for provider in providers
        }
        return JSONResponse(status_code=200, content=stored_status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stored providers: {str(e)}")

# Endpoint to delete an API key for a specific provider
@app.post("/delete-api-key")
async def delete_api_key(request: DeleteApiKeyRequest):
    """
    Deletes the API key for the specified provider from the keyring.
    
    Args:
        request (DeleteApiKeyRequest): Request containing the provider name.
    
    Returns:
        JSONResponse: Success status or error message.
    """
    try:
        keyring.delete_password("electron-openid-oauth", request.provider)
        return JSONResponse(status_code=200, content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting API key: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app, port=5005
    )  # Run the FastAPI application using uvicorn on port 5005
