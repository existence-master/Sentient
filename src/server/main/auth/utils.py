import os
import json
import logging
import traceback
from typing import Optional, Dict, Any, List, Tuple

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import base64
import requests

from jose import jwt, JWTError
from jose.exceptions import JOSEError
from fastapi import HTTPException, status, Depends, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer
from json_extractor import JsonExtractor

from main.config import (
    ENVIRONMENT, SELF_HOST_AUTH_SECRET,
    AES_SECRET_KEY, AES_IV, AUTH0_SCOPE, AUTH0_NAMESPACE,
    AUTH0_DOMAIN, AUTH0_AUDIENCE, ALGORITHMS,
    AUTH0_MANAGEMENT_CLIENT_ID, AUTH0_MANAGEMENT_CLIENT_SECRET
)

logger = logging.getLogger(__name__)

# --- JWKS Fetching ---
jwks = None
if AUTH0_DOMAIN:
    jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
    try:
        logger.info(f"Fetching JWKS from {jwks_url}...")
        jwks_response = requests.get(jwks_url, timeout=10)
        jwks_response.raise_for_status()
        jwks = jwks_response.json()
        logger.info("JWKS fetched successfully.")
    except requests.exceptions.RequestException as e:
        logger.critical(f"Could not fetch JWKS: {e}", exc_info=True)
        jwks = None
    except Exception as e:
        logger.critical(f"Error processing JWKS: {e}", exc_info=True)
        jwks = None
else:
    logger.critical("AUTH0_DOMAIN not set. Cannot fetch JWKS.")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class AuthHelper:
    async def _validate_token_and_get_payload(self, token: str) -> dict:
        if ENVIRONMENT == "selfhost":
            if not SELF_HOST_AUTH_SECRET:
                logger.critical("selfhost mode is active but SELF_HOST_AUTH_SECRET is not set.")
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Self-host auth secret not configured.")
            if token == SELF_HOST_AUTH_SECRET:
                # Return a mock payload with a static user_id and all permissions
                permissions = AUTH0_SCOPE.split() if AUTH0_SCOPE else []
                return {
                    "sub": "self-hosted-user",
                    "permissions": permissions,
                    "email": "selfhost@example.com" # Add a dummy email
                }
            else:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid self-host token")
        if not jwks:
            logger.error("JWKS not available for token validation.")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth service config error (JWKS).")

        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            unverified_header = jwt.get_unverified_header(token)
            token_kid = unverified_header.get("kid")
            if not token_kid:
                raise credentials_exception

            rsa_key_data = {}
            for key_entry in jwks.get("keys", []):
                if isinstance(key_entry, dict) and key_entry.get("kid") == token_kid:
                    rsa_key_data = {comp: key_entry[comp] for comp in ["kty", "kid", "use", "n", "e"] if comp in key_entry}
                    if all(k in rsa_key_data for k in ["kty", "n", "e"]): break
            
            if not rsa_key_data or not all(k in rsa_key_data for k in ["kty", "n", "e"]):
                raise credentials_exception

            payload = jwt.decode(
                token, rsa_key_data, algorithms=ALGORITHMS,
                audience=AUTH0_AUDIENCE, issuer=f"https://{AUTH0_DOMAIN}/"
            )
            return payload
        except JWTError as e:
            logger.warning(f"JWT validation error: {e}")
            raise credentials_exception
        except JOSEError as e:
            logger.warning(f"JOSE validation error: {e}")
            raise credentials_exception
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected token validation error: {e}", exc_info=True)
            raise credentials_exception

    async def get_current_user_id_plan_and_permissions(self, token: str = Depends(oauth2_scheme)) -> Tuple[str, str, List[str]]:
        payload = await self._validate_token_and_get_payload(token)
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User ID (sub) not in token")

        permissions = payload.get("permissions", [])

        # Determine plan
        plan = "free"
        if ENVIRONMENT == "selfhost":
            plan = "selfhost"
        else:
            roles = payload.get(f"{AUTH0_NAMESPACE}/roles", [])
            if "Pro" in roles:
                plan = "pro"

        return user_id, plan, permissions

    async def get_current_user_id(self, token: str = Depends(oauth2_scheme)) -> str:
        user_id, _, _ = await self.get_current_user_id_plan_and_permissions(token=token)
        return user_id
    
    async def get_decoded_payload_with_claims(self, token: str = Depends(oauth2_scheme)) -> dict:
        payload = await self._validate_token_and_get_payload(token)
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User ID (sub) not in token")
        payload["user_id"] = user_id 

        # Add plan to payload for easy access in endpoints
        plan = "free"
        if ENVIRONMENT == "selfhost": plan = "selfhost"
        elif "Pro" in payload.get(f"{AUTH0_NAMESPACE}/roles", []): plan = "pro"
        payload["plan"] = plan

        # Add the namespaced email to the top-level 'email' key for convenience
        if AUTH0_NAMESPACE:
            payload["email"] = payload.get(f"{AUTH0_NAMESPACE}/email")

        return payload

    async def ws_authenticate_with_data(self, websocket: WebSocket) -> Optional[Dict]:
        try:
            auth_message_str = await websocket.receive_text()
            auth_message = JsonExtractor.extract_valid_json(auth_message_str)
            
            if not auth_message or not isinstance(auth_message, dict) or auth_message.get("type") != "auth" or not auth_message.get("token"):
                await websocket.send_json({"type": "auth_failure", "message": "Invalid auth message format."})
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return None

            token = auth_message["token"]
            payload = await self._validate_token_and_get_payload(token)
            user_id = payload.get("sub")

            if not user_id:
                await websocket.send_json({"type": "auth_failure", "message": "User ID not found in token."})
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return None
            
            # Add other data from the auth message to the return dict
            auth_data = {
                "user_id": user_id,
                **{k: v for k, v in auth_message.items() if k not in ["type", "token"]}
            }

            await websocket.send_json({"type": "auth_success", "user_id": user_id})
            return auth_data
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected during authentication.")
            return None
        except HTTPException as e:
            await websocket.send_json({"type": "auth_failure", "message": f"Token validation failed: {e.detail}"})
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None
        except Exception as e:
            logger.error(f"Unexpected error during WebSocket authentication: {e}", exc_info=True)
            try:
                await websocket.send_json({"type": "auth_failure", "message": "Internal server error during auth."})
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            except: pass
            return None

    async def ws_authenticate(self, websocket: WebSocket) -> Optional[str]:
        auth_data = await self.ws_authenticate_with_data(websocket)
        return auth_data.get("user_id") if auth_data else None

    async def get_current_user_id_and_plan(self, token: str = Depends(oauth2_scheme)) -> Tuple[str, str]:
        user_id, plan, _ = await self.get_current_user_id_plan_and_permissions(token=token)
        return user_id, plan

class PermissionChecker:
    def __init__(self, required_permissions: List[str]):
        self.required_permissions = set(required_permissions)

    async def __call__(self, token: str = Depends(oauth2_scheme)):
        from main.dependencies import auth_helper
        user_id, _, token_permissions_list = await auth_helper.get_current_user_id_plan_and_permissions(token=token)
        token_permissions_set = set(token_permissions_list)

        if not self.required_permissions.issubset(token_permissions_set):
            missing = self.required_permissions - token_permissions_set
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Missing permissions: {', '.join(missing)}")
        return user_id

# --- AES Encryption/Decryption ---
def aes_encrypt(data: str) -> str:
    if not AES_SECRET_KEY or not AES_IV:
        raise ValueError("AES encryption keys are not configured.")
    backend = default_backend()
    cipher = Cipher(algorithms.AES(AES_SECRET_KEY), modes.CBC(AES_IV), backend=backend)
    encryptor = cipher.encryptor()
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data.encode()) + padder.finalize()
    encrypted = encryptor.update(padded_data) + encryptor.finalize()
    return base64.b64encode(encrypted).decode()

def aes_decrypt(encrypted_data: str) -> str:
    if not AES_SECRET_KEY or not AES_IV:
        raise ValueError("AES encryption keys are not configured.")
    backend = default_backend()
    cipher = Cipher(algorithms.AES(AES_SECRET_KEY), modes.CBC(AES_IV), backend=backend)
    decryptor = cipher.decryptor()
    encrypted_bytes = base64.b64decode(encrypted_data)
    decrypted = decryptor.update(encrypted_bytes) + decryptor.finalize()
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    unpadded_data = unpadder.update(decrypted) + unpadder.finalize()
    return unpadded_data.decode()

# --- Auth0 Management API Token ---
def get_management_token() -> str:
    if not all([AUTH0_DOMAIN, AUTH0_MANAGEMENT_CLIENT_ID, AUTH0_MANAGEMENT_CLIENT_SECRET]):
        logger.error("Auth0 Management API credentials not fully configured.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Auth0 Management API config error.")
    
    url = f"https://{AUTH0_DOMAIN}/oauth/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": AUTH0_MANAGEMENT_CLIENT_ID,
        "client_secret": AUTH0_MANAGEMENT_CLIENT_SECRET,
        "audience": f"https://{AUTH0_DOMAIN}/api/v2/",
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        response = requests.post(url, data=payload, headers=headers, timeout=10)
        response.raise_for_status()
        token_data = response.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid token response from Auth0 Mgmt API.")
        return access_token
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get Auth0 management token: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Failed to get management token: {e}")
    except Exception as e:
        logger.error(f"Error processing Auth0 management token response: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing management token response.")