import os
import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

# --- Configuration from Environment Variables ---
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
GOOGLE_CLIENT_EMAIL = os.getenv("GOOGLE_CLIENT_EMAIL")
# Private key needs special handling for newline characters when loaded from .env
GOOGLE_PRIVATE_KEY = os.getenv("GOOGLE_PRIVATE_KEY", "").replace('\\n', '\n')

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SHEET_NAME = "Sheet1" # Assuming the sheet name is static

def _get_sheets_service():
    """Authenticates and returns a Google Sheets service object."""
    if not all([GOOGLE_SHEET_ID, GOOGLE_CLIENT_EMAIL, GOOGLE_PRIVATE_KEY]):
        logger.warning("Google Sheets credentials are not fully configured. Skipping sheet update.")
        return None

    try:
        creds = service_account.Credentials.from_service_account_info(
            {
                "client_email": GOOGLE_CLIENT_EMAIL,
                "private_key": GOOGLE_PRIVATE_KEY,
                "token_uri": "https://oauth2.googleapis.com/token",
                "project_id": os.getenv("GOOGLE_PROJECT_ID") # Optional but good practice
            },
            scopes=SCOPES
        )
        service = build('sheets', 'v4', credentials=creds)
        return service
    except Exception as e:
        logger.error(f"Failed to create Google Sheets service: {e}", exc_info=True)
        return None

async def update_onboarding_data_in_sheet(user_email: str, onboarding_data: dict, plan: str):
    """Finds a user by email and updates their onboarding information in the sheet."""
    service = _get_sheets_service()
    if not service:
        return

    try:
        # 1. Find user row by email in Column C
        range_to_read = f"{SHEET_NAME}!C:C"
        result = service.spreadsheets().values().get(spreadsheetId=GOOGLE_SHEET_ID, range=range_to_read).execute()
        rows = result.get('values', [])

        row_index = -1
        for i, row in enumerate(rows):
            if row and row[0] == user_email:
                row_index = i
                break

        if row_index == -1:
            logger.warning(f"User with email {user_email} not found in Google Sheet. Cannot update onboarding data.")
            return

        # 2. Prepare data for batch update
        row_number = row_index + 1

        # Handle location which can be a string or a dict
        location = onboarding_data.get('location', '')
        if isinstance(location, dict):
            lat = location.get('latitude')
            lon = location.get('longitude')
            if lat is not None and lon is not None:
                location = f"Lat: {lat}, Lon: {lon}"
            else:
                location = str(location)

        # Prepare a list of values to update. None will skip the cell.
        # Columns: B (Contact), D (Location), E (Profession), F (Hobbies), H (Plan)
        values_to_update = [
            [
                onboarding_data.get('whatsapp_notifications_number', ''), # B
                None, # C - Email (skip)
                location, # D
                onboarding_data.get('professional-context', ''), # E
                onboarding_data.get('personal-context', ''), # F
                None, # G - Insider (skip)
                plan.capitalize() # H
            ]
        ]

        # 3. Update the sheet row from column B to H
        range_to_update = f"{SHEET_NAME}!B{row_number}:H{row_number}"
        service.spreadsheets().values().update(
            spreadsheetId=GOOGLE_SHEET_ID,
            range=range_to_update,
            valueInputOption='USER_ENTERED',
            body={'values': values_to_update}
        ).execute()
        logger.info(f"Successfully updated onboarding data for {user_email} in Google Sheet.")

    except Exception as e:
        logger.error(f"An error occurred while updating Google Sheet for {user_email}: {e}", exc_info=True)
async def update_plan_in_sheet(user_email: str, new_plan: str):
    """Finds a user by email and updates only their plan in the sheet."""
    service = _get_sheets_service()
    if not service:
        return

    try:
        range_to_read = f"{SHEET_NAME}!C:C"
        result = service.spreadsheets().values().get(spreadsheetId=GOOGLE_SHEET_ID, range=range_to_read).execute()
        rows = result.get('values', [])

        row_index = -1
        for i, row in enumerate(rows):
            if row and row[0] == user_email:
                row_index = i
                break

        if row_index != -1:
            range_to_update = f"{SHEET_NAME}!H{row_index + 1}"
            service.spreadsheets().values().update(
                spreadsheetId=GOOGLE_SHEET_ID,
                range=range_to_update,
                valueInputOption='USER_ENTERED',
                body={'values': [[new_plan.capitalize()]]}
            ).execute()
            logger.info(f"Successfully updated plan to '{new_plan}' for {user_email} in Google Sheet.")
        else:
            logger.warning(f"User with email {user_email} not found in Google Sheet. Could not update plan.")
    except Exception as e:
        logger.error(f"An error occurred while updating plan in Google Sheet for {user_email}: {e}", exc_info=True)
