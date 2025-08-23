import os
import re
import logging
from typing import Tuple
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from main.dependencies import auth_helper, mongo_manager
from main.plans import PLAN_LIMITS
from main.plans import PLAN_LIMITS
from main.config import FILE_MANAGEMENT_TEMP_DIR # Import the base directory constant
from .utils import get_user_temp_dir

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/files",
    tags=["File Management"]
)

def sanitize_filename(filename: str) -> str:
    """Prevents path traversal and removes unsafe characters."""
    # Remove directory separators
    filename = filename.replace("/", "").replace("\\", "")
    # Remove characters that could be used for path traversal
    filename = re.sub(r'\.\.', '', filename)
    # Remove other potentially problematic characters (optional, but good practice)
    filename = re.sub(r'[<>:"|?*]', '', filename)
    return filename

@router.post("/upload", summary="Upload a file for AI context")
async def upload_file(
    file: UploadFile = File(...),
    user_id_and_plan: Tuple[str, str] = Depends(auth_helper.get_current_user_id_and_plan)
):
    user_id, plan = user_id_and_plan

    # --- Check Usage Limit ---
    usage = await mongo_manager.get_or_create_daily_usage(user_id)
    limit = PLAN_LIMITS[plan].get("file_uploads_daily", 0)
    current_count = usage.get("file_uploads", 0)

    if current_count >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"You have reached your daily file upload limit of {limit}. Please upgrade or try again tomorrow."
        )

    # Create a user-specific subdirectory
    safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ('-', '_'))
    user_specific_dir = os.path.join(FILE_MANAGEMENT_TEMP_DIR, safe_user_id)
    os.makedirs(user_specific_dir, exist_ok=True)

    # Sanitize the original filename to prevent security issues
    sanitized_filename = sanitize_filename(file.filename)

    # Overwrite file if it exists, no need to make it unique.
    file_path = os.path.join(user_specific_dir, sanitized_filename)

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Increment usage after successful save
        await mongo_manager.increment_daily_usage(user_id, "file_uploads")
        logger.info(f"User {user_id} successfully uploaded file '{sanitized_filename}' to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save file for user {user_id}: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to save file: {e}"})
    
    # Return only the filename, not the full path with user_id
    return JSONResponse(content={"filename": sanitized_filename})

@router.get("/download/{filepath:path}", summary="Download a file")
async def download_file(
    filepath: str,
    user_id: str = Depends(auth_helper.get_current_user_id)
):
    """
    Provides a secure way to download a file from the user's temporary directory.
    """
    try:
        # This is the directory for the specific user.
        user_specific_dir = get_user_temp_dir(user_id)

        # Sanitize the filename received from the URL to prevent path traversal.
        sanitized_filename = sanitize_filename(filepath)

        # Construct the full, absolute path to the requested file.
        full_path = os.path.join(user_specific_dir, sanitized_filename)

        # Security Check: Ensure the final, resolved path is inside the user's designated directory.
        # This is a redundant but important check against any os.path.join trickery.
        if not os.path.abspath(full_path).startswith(os.path.abspath(user_specific_dir)):
            raise HTTPException(status_code=403, detail="Forbidden: Access denied.")

        if not os.path.isfile(full_path):
            raise HTTPException(status_code=404, detail="File not found.")

        logger.info(f"User {user_id} initiated download for file: {sanitized_filename}")
        # Use FileResponse to stream the file back to the client.
        return FileResponse(path=full_path, filename=sanitized_filename, media_type='application/octet-stream')

    except HTTPException as he:
        logger.warning(f"Download failed for user {user_id}, file {filepath}. Status: {he.status_code}, Detail: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error during file download for user {user_id}, file {filepath}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))