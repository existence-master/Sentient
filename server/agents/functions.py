import os
from wrapt_timeout_decorator import *
import json
import asyncio
import uuid
import pickle
from datetime import datetime, timedelta
from googleapiclient.errors import HttpError
from email.mime.text import MIMEText
from base64 import urlsafe_b64encode
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload, MediaIoBaseUpload
import httpx
from sentence_transformers import SentenceTransformer, util
from urllib.parse import quote
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
import time
import matplotlib.pyplot as plt
import io
import requests
import aiohttp

from server.app.helpers import *

load_dotenv("server/.env")  # Load environment variables from .env file


def authenticate_service(api_name: str, api_version: str):
    """
    Authenticate and return the specified Google API service.

    This function attempts to load credentials from a token pickle file.
    If the file exists, it loads the credentials. Otherwise, authentication
    would typically need to be handled through an OAuth flow (not shown here but implied).

    Args:
        api_name (str): The name of the Google API (e.g., 'gmail', 'docs').
        api_version (str): The version of the Google API (e.g., 'v1', 'v3').

    Returns:
        googleapiclient.discovery.Resource: The authenticated Google API service resource.
    """
    creds = None

    if os.path.exists("server/token.pickle"):
        with open("server/token.pickle", "rb") as token:
            creds = pickle.load(token)

    return build(api_name, api_version, credentials=creds)


def authenticate_gmail():
    """
    Authenticate and return the Gmail API service.

    Returns:
        googleapiclient.discovery.Resource: Authenticated Gmail API service resource.
    """
    return authenticate_service("gmail", "v1")


def authenticate_docs():
    """
    Authenticate and return the Google Docs API service.

    Returns:
        googleapiclient.discovery.Resource: Authenticated Google Docs API service resource.
    """
    return authenticate_service("docs", "v1")


def authenticate_calendar():
    """
    Authenticate and return the Google Calendar API service.

    Returns:
        googleapiclient.discovery.Resource: Authenticated Google Calendar API service resource.
    """
    return authenticate_service("calendar", "v3")


def authenticate_sheets():
    """
    Authenticate and return the Google Sheets API service.

    Returns:
        googleapiclient.discovery.Resource: Authenticated Google Sheets API service resource.
    """
    return authenticate_service("sheets", "v4")


def authenticate_slides():
    """
    Authenticate and return the Google Slides API service.

    Returns:
        googleapiclient.discovery.Resource: Authenticated Google Slides API service resource.
    """
    return authenticate_service("slides", "v1")


def authenticate_drive():
    """
    Authenticate and return the Google Drive API service.

    Returns:
        googleapiclient.discovery.Resource: Authenticated Google Drive API service resource.
    """
    return authenticate_service("drive", "v3")

async def create_message(to: str, subject: str, message: str) -> str:
    """
    Create a MIME message for an email and elaborate the message body using an external service.

    This function takes recipient, subject, and a message body, then calls an external
    'elaborator' service to potentially enhance or reformat the message. Finally, it constructs
    a MIMEText message ready to be sent via email.

    Args:
        to (str): Recipient email address.
        subject (str): Email subject.
        message (str): Email body text.

    Returns:
        str: Raw, URL-safe base64 encoded MIME message, ready to be sent via Gmail API.
             Raises an exception if there is an error during message creation or elaboration.
    """
    try:
        async with HTTPXAsyncClient() as client: # Use the imported alias
            response = await client.post(
                "http://localhost:5000/elaborator",
                json={
                    "input": f"ELABORATE THIS EMAIL IN THE FIRST PERSON. USE THE NECESSARY SALUTATIONS.\n\nEMAIL SUBJECT: {subject}.\n\nEMAIL BODY: {message}",
                    "purpose": "Email",
                },
                timeout=30.0 # Add a timeout
            )

            response_json = response.json()
            email_body_to_send = message # Default to original message

            if response.status_code == 200:
                elaborated_content = response_json.get("message")
                if isinstance(elaborated_content, dict) and "email" in elaborated_content and "body" in elaborated_content["email"]:
                    email_body_to_send = elaborated_content["email"]["body"]
                elif isinstance(elaborated_content, str): # If elaborator returns a simple string
                    email_body_to_send = elaborated_content
                # else, email_body_to_send remains the original message
        # else: # If elaborator call failed, email_body_to_send remains the original message

        final_body = email_body_to_send + "\n\nThis email was sent via Sentient, a personal AI companion with agentic integrations and graph memory.\n\nLearn more at https://existence.technology/sentient\n\n Emails are generated by AI models and may not always be accurate or contextually relevant."
        msg = MIMEText(final_body)
        msg.add_header("To", to)
        msg.add_header("Subject", subject)

        raw = urlsafe_b64encode(msg.as_bytes()).decode()
        return raw
    except Exception as error:
        print(f"Error creating message: {str(error)}")
        raise Exception(f"Error creating message: {error}")
    
async def search_unsplash_image(query: str) -> str:
    """Search for an image on Unsplash and return its URL.

    Args:
        query (str): The search term to find an image (e.g., 'nature').

    Returns:
        str: The URL of the first image result, or None if no image is found or an error occurs.
    """
    try:
        access_key = os.getenv("UNSPLASH_ACCESS_KEY")
        if not access_key:
            print("Unsplash access key is missing. Please set UNSPLASH_ACCESS_KEY in your .env file.")
            return None

        url = f"https://api.unsplash.com/search/photos?query={query}"
        headers = {'Authorization': f'Client-ID {access_key}'}
        # Use httpx for consistency if this were async, but requests is fine for sync
        import requests as sync_requests # Alias to avoid conflict if httpx.requests exists
        response = sync_requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            return None
            
        data = response.json()
        if data.get("results"):
            return data["results"][0]["urls"]["regular"]
            
        print(f"No images found for query: {query}")
        return None
    except sync_requests.RequestException as e: # Use aliased requests
        print(f"Error searching Unsplash: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

async def upload_image_to_slide(service, drive_service, presentation_id, slide_id, image_url):
    """Upload an image from a URL to a specific slide.

    Args:
        service: Google Slides API service instance.
        drive_service: Google Drive API service instance.
        presentation_id (str): ID of the presentation.
        slide_id (str): ID of the slide to add the image to.
        image_url (str): URL of the image to upload.

    Returns:
        dict: Status and result or error message.
    """
    try:
        import requests as sync_requests # Alias
        response = sync_requests.get(image_url)
        if response.status_code != 200:
            return {"status": "failure", "error": f"Failed to download image from {image_url}"}
        image_bytes = response.content
        file_name = f'slide_image_{int(time.time())}.jpg'
        mimetype = 'image/jpeg'
        return add_image_to_slide_from_bytes(service, drive_service, presentation_id, slide_id, image_bytes, file_name, mimetype)
    except Exception as e:
        return {"status": "failure", "error": str(e)}

def generate_chart_bytes(chart_type, categories, data):
    """Generate a chart image in memory and return its bytes.

    Args:
        chart_type (str): Type of chart ('bar', 'pie', 'line').
        categories (list): Categories for the chart.
        data (list): Data points for the chart.

    Returns:
        bytes: The chart image in bytes.
    """
    try:
        plt.figure(figsize=(6, 4))
        if chart_type == "bar":
            plt.bar(categories, data)
        elif chart_type == "pie":
            plt.pie(data, labels=categories, autopct='%1.1f%%')
        elif chart_type == "line":
            plt.plot(categories, data)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        plt.title("Chart")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        raise RuntimeError(f"Error generating chart: {e}")

def add_image_to_slide_from_bytes(service, drive_service, presentation_id, slide_id, image_bytes, file_name, mimetype):
    """Upload image bytes to Google Drive and add the image to a specific slide.

    Args:
        service: Google Slides API service instance.
        drive_service: Google Drive API service instance.
        presentation_id (str): ID of the presentation.
        slide_id (str): ID of the slide to add the image to.
        image_bytes (bytes): The image data in bytes.
        file_name (str): The name to give the file in Google Drive.
        mimetype (str): The MIME type of the image (e.g., 'image/png').

    Returns:
        dict: Status and result or error message.
    """
    try:
        # Upload to Google Drive
        file_metadata = {'name': file_name}
        media = MediaIoBaseUpload(io.BytesIO(image_bytes), mimetype=mimetype)
        uploaded_file = drive_service.files().create(
            body=file_metadata, media_body=media, fields='id'
        ).execute()
        file_id = uploaded_file.get('id')

        # Make the file publicly accessible
        permission = {'type': 'anyone', 'role': 'reader'}
        drive_service.permissions().create(fileId=file_id, body=permission).execute()
        image_url = f"https://drive.google.com/uc?id={file_id}"

        # Add the image to the slide
        create_image_request = {
            "createImage": {
                "url": image_url,
                "elementProperties": {
                    "pageObjectId": slide_id,
                    "size": {"width": {"magnitude": 450, "unit": "PT"}, "height": {"magnitude": 300, "unit": "PT"}},
                    "transform": {"scaleX": 1, "scaleY": 1, "translateX": 50, "translateY": 150, "unit": "PT"}
                }
            }
        }
        service.presentations().batchUpdate(
            presentationId=presentation_id, body={"requests": [create_image_request]}
        ).execute()
        return {"status": "success", "result": "Image added to slide"}
    except Exception as e:
        return {"status": "failure", "error": str(e)}

async def add_chart_to_slide(service, drive_service, presentation_id, slide_id, chart_type, categories, data):
    """Generate a chart and add it to a specific slide.

    Args:
        service: Google Slides API service instance.
        drive_service: Google Drive API service instance.
        presentation_id (str): ID of the presentation.
        slide_id (str): ID of the slide to add the chart to.
        chart_type (str): Type of chart ('bar', 'pie', 'line').
        categories (list): Categories for the chart.
        data (list): Data points for the chart.

    Returns:
        dict: Status and result or error message.
    """
    try:
        # Generate chart bytes
        image_bytes = generate_chart_bytes(chart_type, categories, data)
        file_name = f'chart_{chart_type}_{int(time.time())}.png'
        mimetype = 'image/png'

        # Add chart to slide
        return add_image_to_slide_from_bytes(
            service, drive_service, presentation_id, slide_id, image_bytes, file_name, mimetype
        )
    except Exception as e:
        return {"status": "failure", "error": str(e)}
    

async def create_presentation(service, title: str) -> Dict[str, Any]:
    """
    Create a new Google Slides presentation.

    Args:
        service: Authenticated Google Slides API service.
        title (str): Title of the new presentation.

    Returns:
        Dict[str, Any]: The created presentation resource.
                         Raises an exception if there is an error during presentation creation.
    """
    try:
        presentation = service.presentations().create(body={"title": title}).execute()
        return presentation
    except HttpError as error:
        print(str(error))
        raise Exception(error)
    
async def create_google_presentation(outline: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a Google Slides presentation with enhanced features from gslides.py and title slide logic."""
    print("Starting create_google_presentation function")
    print(f"Topic: {outline.get('topic')}, Username: {outline.get('username')}")
    try:
        print("OUTLINE: ", outline)
        service = authenticate_slides()
        drive_service = authenticate_drive()
        print("Successfully authenticated Google Slides and Drive services.")
        title = outline["topic"]
        user_name = outline["username"]
        presentation = await create_presentation(service, title)
        presentation_id = presentation["presentationId"]
        presentation_url = f"https://docs.google.com/presentation/d/{presentation_id}"
        print(f"Presentation created successfully. ID: {presentation_id}, URL: {presentation_url}")

        # Update title slide with title and subtitle
        print("Updating title slide...")
        presentation_data = service.presentations().get(presentationId=presentation_id).execute()
        default_slide = presentation_data["slides"][0]

        title_placeholder_id = None
        subtitle_placeholder_id = None

        for element in default_slide.get("pageElements", []):
            if "shape" in element:
                if element["shape"]["shapeType"] == "TEXT_BOX":
                    if title_placeholder_id is None:
                        title_placeholder_id = element["objectId"]
                    else:
                        subtitle_placeholder_id = element["objectId"]

        print(f"Title placeholder ID: {title_placeholder_id}, Subtitle placeholder ID: {subtitle_placeholder_id}")

        if title_placeholder_id:
            print("Updating title placeholder...")
            service.presentations().batchUpdate(
                presentationId=presentation_id,
                body={
                    "requests": [
                        {
                            "insertText": {
                                "objectId": title_placeholder_id,
                                "text": title,
                                "insertionIndex": 0,
                            }
                        }
                    ]
                },
            ).execute()
            print("Title placeholder updated.")

        if subtitle_placeholder_id:
            print("Updating subtitle placeholder...")
            service.presentations().batchUpdate(
                presentationId=presentation_id,
                body={
                    "requests": [
                        {
                            "insertText": {
                                "objectId": subtitle_placeholder_id,
                                "text": f"Created by {user_name}",
                                "insertionIndex": 0,
                            }
                        }
                    ]
                },
            ).execute()
            print("Subtitle placeholder updated.")
        else:
            print("Subtitle placeholder not found. Creating a new subtitle textbox.")
            requests = [
                {
                    "createShape": {
                        "objectId": "SubtitleTextBox",
                        "shapeType": "TEXT_BOX",
                        "elementProperties": {
                            "pageObjectId": default_slide["objectId"],
                            "size": {
                                "width": {"magnitude": 5000000, "unit": "EMU"},
                                "height": {"magnitude": 1000000, "unit": "EMU"},
                            },
                            "transform": {
                                "translateX": 1000000,
                                "translateY": 2000000,
                                "unit": "EMU",
                            },
                        },
                    }
                },
                {
                    "insertText": {
                        "objectId": "SubtitleTextBox",
                        "text": f"Created by {user_name}",
                        "insertionIndex": 0,
                    }
                },
            ]
            service.presentations().batchUpdate(
                presentationId=presentation_id, body={"requests": requests}
            ).execute()
            print("New subtitle textbox created and updated.")
        print("Title slide updated.")

        # Add slides with content, images, and charts
        print("Adding slides from outline...")
        for slide in outline["slides"]:
            slide_title = slide.get("title", "Untitled Slide")
            slide_content = slide.get("content", ["No content provided."])
            if not isinstance(slide_content, list):
                slide_content = [slide_content]

            print(f"Creating slide with title: '{slide_title}' and content: {slide_content}")
            create_slide_request = {
                "createSlide": {"slideLayoutReference": {"predefinedLayout": "TITLE_AND_BODY"}}
            }
            response = service.presentations().batchUpdate(presentationId=presentation_id, body={"requests": [create_slide_request]}).execute()
            slide_id = response["replies"][0]["createSlide"]["objectId"]
            print(f"Slide created with ID: {slide_id}")

            # Add title and content
            slide_data = service.presentations().get(presentationId=presentation_id).execute()["slides"][-1]
            title_id = next((e["objectId"] for e in slide_data["pageElements"] if "shape" in e), None)
            body_id = next((e["objectId"] for e in slide_data["pageElements"] if e["objectId"] != title_id), None)
            requests = [
                {"insertText": {"objectId": title_id, "text": slide_title, "insertionIndex": 0}},
                {"insertText": {"objectId": body_id, "text": "\n".join(slide_content), "insertionIndex": 0}}
            ]
            service.presentations().batchUpdate(presentationId=presentation_id, body={"requests": requests}).execute()
            print("Slide title and content updated.")

            # Add image if present
            if "image_description" in slide:
                image_description = slide["image_description"]
                print(f"Adding image to slide with description: '{image_description}'")
                image_url = await search_unsplash_image(image_description)
                if image_url:
                    print(f"Image URL found: {image_url}. Uploading image...")
                    await upload_image_to_slide(service, drive_service, presentation_id, slide_id, image_url)
                    print("Image uploaded to slide.")
                else:
                    print("No image URL found for the description.")

            # Add chart if present
            if "chart" in slide:
                chart = slide["chart"]
                print(f"Adding chart to slide: {chart}")
                chart_result = await add_chart_to_slide(
                    service, 
                    drive_service, 
                    presentation_id, 
                    slide_id, 
                    chart["type"], 
                    chart["categories"], 
                    chart["data"]
                )
                if chart_result["status"] == "success":
                    print("Chart added to slide successfully.")
                else:
                    print(f"Failed to add chart to slide. Error: {chart_result.get('error')}")

        print("All slides added successfully.")
        print("Presentation creation successful.")
        return {
            "status": "success",
            "result": {
                "response": "Presentation created successfully",
                "presentationUrl": presentation_url,
                "presentation_id": presentation_id
            }
        }
    except Exception as e:
        error_message = str(e)
        print(f"Error in create_google_presentation: {error_message}")
        return {"status": "failure", "error": error_message}

async def create_slide(service, presentation_id: str, title: str, content: List[str]):
    """
    Create a slide in an existing Google Slides presentation and insert text into title and body.

    This function creates a new slide with 'TITLE_AND_BODY' layout, then finds or creates text boxes
    for the title and body, and inserts the provided title and content.

    Args:
        service: Authenticated Google Slides API service.
        presentation_id (str): ID of the presentation to add the slide to.
        title (str): Title text for the slide.
        content (List[str]): List of strings representing the content for the slide body.
                             Each string will be a new line in the body text box.

    Raises:
        Exception: If there's an error in creating or updating the slide.
    """
    try:
        requests = [
            {
                "createSlide": {
                    "slideLayoutReference": {"predefinedLayout": "TITLE_AND_BODY"}
                }
            }
        ]
        response = (
            service.presentations()
            .batchUpdate(presentationId=presentation_id, body={"requests": requests})
            .execute()
        )

        slide_id = response["replies"][0]["createSlide"]["objectId"]

        slide_objects = (
            service.presentations()
            .get(presentationId=presentation_id)
            .execute()["slides"]
        )
        title_box_id = None
        body_box_id = None

        for slide in slide_objects:
            if slide["objectId"] == slide_id:
                elements = slide.get("pageElements", [])
                for element in elements:
                    shape = element.get("shape")
                    if shape:
                        shape_type = shape.get("shapeType")
                        if shape_type == "TEXT_BOX":
                            if title_box_id == None:
                                title_box_id = element["objectId"]
                            else:
                                body_box_id = element["objectId"]
                break

        requests = []
        if title_box_id:
            requests.append(
                {
                    "insertText": {
                        "objectId": title_box_id,
                        "text": title,
                        "insertionIndex": 0,
                    }
                }
            )
        else:
            unique_title_id = f"TitleBox_{uuid.uuid4().hex}"
            requests.append(
                {
                    "createShape": {
                        "objectId": unique_title_id,
                        "shapeType": "TEXT_BOX",
                        "elementProperties": {
                            "pageObjectId": slide_id,
                            "size": {
                                "width": {"magnitude": 3000000, "unit": "EMU"},
                                "height": {"magnitude": 500000, "unit": "EMU"},
                            },
                            "transform": {
                                "scaleX": 1,
                                "scaleY": 1,
                                "translateX": 200000,
                                "translateY": 200000,
                                "unit": "EMU",
                            },
                        },
                    }
                }
            )
            requests.append(
                {
                    "insertText": {
                        "objectId": unique_title_id,
                        "text": title,
                        "insertionIndex": 0,
                    }
                }
            )

        if body_box_id:
            requests.append(
                {
                    "insertText": {
                        "objectId": body_box_id,
                        "text": "\n".join(content),
                        "insertionIndex": 0,
                    }
                }
            )
        else:
            unique_body_id = f"BodyBox_{uuid.uuid4().hex}"
            requests.append(
                {
                    "createShape": {
                        "objectId": unique_body_id,
                        "shapeType": "TEXT_BOX",
                        "elementProperties": {
                            "pageObjectId": slide_id,
                            "size": {
                                "width": {"magnitude": 4000000, "unit": "EMU"},
                                "height": {"magnitude": 2000000, "unit": "EMU"},
                            },
                            "transform": {
                                "scaleX": 1,
                                "scaleY": 1,
                                "translateX": 200000,
                                "translateY": 1000000,
                                "unit": "EMU",
                            },
                        },
                    }
                }
            )
            requests.append(
                {
                    "insertText": {
                        "objectId": unique_body_id,
                        "text": "\n".join(content),
                        "insertionIndex": 0,
                    }
                }
            )

        if requests:
            service.presentations().batchUpdate(
                presentationId=presentation_id, body={"requests": requests}
            ).execute()
        else:
            raise Exception(
                "No valid text placeholders found on the slide, and no text boxes created."
            )

    except HttpError as error:
        print(f"Error creating or updating slide: {str(error)}")
        raise Exception(f"Error creating or updating slide: {error}")

async def send_email(to: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Send an email using the Gmail API.

    Args:
        to (str): Recipient email address.
        subject (str): Email subject.
        body (str): Email body text.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure.
                         Returns {"status": "success", "result": "Email sent successfully."} on success.
                         Returns {"status": "failure", "error": str(error)} on failure.
    """
    try:
        service = authenticate_gmail()
        message_body = {"raw": await create_message(to, subject, body)}
        service.users().messages().send(userId="me", body=message_body).execute()
        return {"status": "success", "result": "Email sent successfully."}
    except Exception as error:
        print(f"Error sending email: {str(error)}")
        return {"status": "failure", "error": str(error)}


async def create_draft(to: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Create a draft email using the Gmail API.

    Args:
        to (str): Recipient email address for the draft.
        subject (str): Email subject for the draft.
        body (str): Email body text for the draft.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure.
                         Returns {"status": "success", "result": f"Draft created successfully with ID: {draft['id']}"} on success.
                         Returns {"status": "failure", "error": str(error)} on failure.
    """
    try:
        service = authenticate_gmail()
        message_body = {"raw": await create_message(to, subject, body)}
        draft = (
            service.users()
            .drafts()
            .create(userId="me", body={"message": message_body})
            .execute()
        )
        return {
            "status": "success",
            "result": f"Draft created successfully with ID: {draft['id']}",
        }
    except Exception as error:
        print(f"Error creating email draft: {str(error)}")
        return {"status": "failure", "error": str(error)}


def search_inbox(query: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Search the Gmail inbox for emails matching the query.

    This function searches the user's Gmail inbox for emails that match the given query.
    It retrieves up to 10 matching emails, extracts relevant information (id, subject, sender, snippet, body),
    and constructs a Gmail search URL for direct access in the browser.

    Args:
        query (str): The search query string (e.g., "from:sender@example.com subject:meeting").
        **kwargs (Any):  Arbitrary keyword arguments. Not used in this function but included for compatibility.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure and search results.
                         On success, returns:
                         {
                             "status": "success",
                             "result": {
                                 "response": "Emails found successfully",
                                 "email_data": List[Dict], # List of email details
                                 "gmail_search_url": str # URL to view search results in Gmail
                             }
                         }
                         On failure, returns:
                         {"status": "failure", "error": str(error)}
    """
    try:
        service = authenticate_gmail()
        results = service.users().messages().list(userId="me", q=query).execute()
        messages = results.get("messages", [])
        email_data: List[Dict[str, Any]] = []
        for message in messages[:10]:
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=message["id"], format="full")
                .execute()
            )

            headers = {
                h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])
            }
            snippet = msg.get("snippet", "No preview available")
            email_body = extract_email_body(msg.get("payload", {}))

            email_data.append(
                {
                    "id": message["id"],
                    "subject": headers.get("Subject", "No Subject"),
                    "from": headers.get("From", "Unknown Sender"),
                    "snippet": snippet,
                    "body": email_body,
                }
            )

        gmail_search_url = f"https://mail.google.com/mail/u/0/#search/{quote(query)}"

        return {
            "status": "success",
            "result": {
                "response": "Emails found successfully",
                "email_data": email_data,
                "gmail_search_url": gmail_search_url,
            },
        }
    except Exception as error:
        print(f"Error searching inbox: {error}")
        return {"status": "failure", "error": str(error)}


async def find_best_matching_email(query: str) -> Dict[str, Any]:
    """
    Search Gmail inbox and find the best matching email ID based on semantic similarity.

    This function searches the user's inbox for recent emails, encodes email subjects and bodies
    using a sentence transformer model, and finds the email with the highest cosine similarity
    to the input query.

    Args:
        query (str): The query string to compare against email content.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure and the best matching email details.
                         On success, returns:
                         {
                             "status": "success",
                             "email_id": str, # ID of the best matching email
                             "email_details": Dict # Details of the best matching email
                         }
                         On failure (no emails found or error during processing), returns:
                         {"status": "failure", "error": str(error)}
    """
    try:
        service = authenticate_gmail()
        results = service.users().messages().list(userId="me", q="in:inbox").execute()
        messages = results.get("messages", [])

        if not messages:
            return {"status": "failure", "error": "No matching emails found."}

        email_data: List[Dict[str, Any]] = []
        for message in messages[:10]:
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=message["id"], format="full")
                .execute()
            )

            headers = {
                h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])
            }
            snippet = msg.get("snippet", "No preview available")
            email_body = extract_email_body(msg.get("payload", {}))

            email_data.append(
                {
                    "id": message["id"],
                    "subject": headers.get("Subject", "No Subject"),
                    "from": headers.get("From", "Unknown Sender"),
                    "snippet": snippet,
                    "body": email_body,
                }
            )

        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode(query, convert_to_tensor=True)
        email_embeddings = model.encode(
            [e["subject"] + " " + e["body"] for e in email_data], convert_to_tensor=True
        )

        scores = util.pytorch_cos_sim(query_embedding, email_embeddings)[0]
        best_match_index = scores.argmax().item()

        best_email = email_data[best_match_index]
        return {
            "status": "success",
            "email_id": best_email["id"],
            "email_details": best_email,
        }

    except HttpError as error:
        return {"status": "failure", "error": f"Google API Error: {str(error)}"}
    except Exception as error:
        print(f"Error in finding best email: {str(error)}")
        return {"status": "failure", "error": str(error)}


async def reply_email(query: str, body: str) -> Dict[str, Any]:
    """
    Reply to an existing email based on a query.

    This function finds the best matching email using `find_best_matching_email` and then sends
    a reply to the sender of that email with the provided body.

    Args:
        query (str): Query to find the email to reply to.
        body (str): Body of the reply email.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure of the reply action.
                         Returns {"status": "success", "result": {"response": "Reply sent successfully."}} on success.
                         Returns {"status": "failure", "error": str(error)} on failure.
    """
    try:
        match = await find_best_matching_email(query)
        if match["status"] != "success":
            return {"status": "failure", "error": "No matching email found."}

        service = authenticate_gmail()

        to = match["email_details"]["from"]
        subject = match["email_details"]["subject"]

        reply_body = body

        message_body = {"raw": await create_message(to, subject, reply_body)}

        service.users().messages().send(userId="me", body=message_body).execute()
        return {"status": "success", "result": {"response": "Reply sent successfully."}}
    except Exception as error:
        print(f"Error in replying email: {str(error)}")
        return {"status": "failure", "error": str(error)}


async def forward_email(query: str, to: str) -> Dict[str, Any]:
    """
    Forward an existing email based on a query to a specified recipient.

    This function finds the best matching email using `find_best_matching_email` and then forwards
    that email to the provided recipient email address.

    Args:
        query (str): Query to find the email to forward.
        to (str): Recipient email address for forwarding.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure of the forward action.
                         Returns {"status": "success", "result": {"response": f"Email forwarded to {to} successfully."}} on success.
                         Returns {"status": "failure", "error": str(error)} on failure.
    """
    try:
        match = await find_best_matching_email(query)
        if match["status"] != "success":
            return {"status": "failure", "error": "No matching email found."}

        service = authenticate_gmail()
        subject = match["email_details"]["subject"]
        body = match["email_details"]["body"]
        message_body = {"raw": await create_message(to, subject, body)}
        service.users().messages().send(userId="me", body=message_body).execute()

        return {
            "status": "success",
            "result": {"response": f"Email forwarded to {to} successfully."},
        }
    except Exception as error:
        print(f"Error in forwarding email: {str(error)}")
        return {"status": "failure", "error": str(error)}


async def delete_email(query: str) -> Dict[str, Any]:
    """
    Delete an email based on a query.

    This function finds the best matching email using `find_best_matching_email` and then deletes
    that email from the user's inbox.

    Args:
        query (str): Query to find the email to delete.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure of the delete action.
                         Returns {"status": "success", "result": {"response": "Email deleted successfully."}} on success.
                         Returns {"status": "failure", "error": str(error)} on failure.
    """
    try:
        match = await find_best_matching_email(query)
        if match["status"] != "success":
            return {"status": "failure", "error": "No matching email found."}

        email_id = match["email_id"]
        service = authenticate_gmail()
        service.users().messages().delete(userId="me", id=email_id).execute()
        return {
            "status": "success",
            "result": {"response": "Email deleted successfully."},
        }
    except Exception as error:
        print(f"Error in deleting email: {str(error)}")
        return {"status": "failure", "error": str(error)}


async def mark_email_as_read(query: str) -> Dict[str, Any]:
    """
    Mark an email as read based on a query.

    This function finds the best matching email using `find_best_matching_email` and then marks
    that email as read in the user's inbox.

    Args:
        query (str): Query to find the email to mark as read.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure of marking as read.
                         Returns {"status": "success", "result": {"response": "Email marked as read."}} on success.
                         Returns {"status": "failure", "error": str(error)} on failure.
    """
    try:
        match = await find_best_matching_email(query)
        if match["status"] != "success":
            return {"status": "failure", "error": "No matching email found."}

        email_id = match["email_id"]
        service = authenticate_gmail()
        service.users().messages().modify(
            userId="me", id=email_id, body={"removeLabelIds": ["UNREAD"]}
        ).execute()
        return {"status": "success", "result": {"response": "Email marked as read."}}
    except Exception as error:
        print(f"Error in marking email as read: {str(error)}")
        return {"status": "failure", "error": str(error)}


async def mark_email_as_unread(query: str) -> Dict[str, Any]:
    """
    Mark an email as unread based on a query.

    This function finds the best matching email using `find_best_matching_email` and then marks
    that email as unread in the user's inbox.

    Args:
        query (str): Query to find the email to mark as unread.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure of marking as unread.
                         Returns {"status": "success", "result": {"response": "Email marked as unread."}} on success.
                         Returns {"status": "failure", "error": str(error)} on failure.
    """
    try:
        match = await find_best_matching_email(query)
        if match["status"] != "success":
            return {"status": "failure", "error": "No matching email found."}

        email_id = match["email_id"]
        service = authenticate_gmail()
        service.users().messages().modify(
            userId="me", id=email_id, body={"addLabelIds": ["UNREAD"]}
        ).execute()
        return {"status": "success", "result": {"response": "Email marked as unread."}}
    except Exception as error:
        print(f"Error in marking email as unread: {str(error)}")
        return {"status": "failure", "error": str(error)}


async def delete_spam_emails(**kwargs: Any) -> Dict[str, Any]:
    """
    Delete all emails from the spam folder.

    This function retrieves all emails in the user's spam folder and deletes them.

    Args:
        **kwargs (Any): Arbitrary keyword arguments. Not used in this function but included for compatibility.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure of deleting spam emails.
                         Returns {"status": "success", "result": {"response": f"Deleted {len(messages)} spam messages."}} on success.
                         Returns {"status": "failure", "error": str(error)} on failure.
    """
    try:
        service = authenticate_gmail()
        results = service.users().messages().list(userId="me", q="in:spam").execute()
        messages = results.get("messages", [])

        if not messages:
            return {"status": "success", "result": "No spam messages found."}

        for message in messages:
            service.users().messages().delete(userId="me", id=message["id"]).execute()

        return {
            "status": "success",
            "result": {"response": f"Deleted {len(messages)} spam messages."},
        }

    except Exception as error:
        print(f"Error in deleting spam emails: {str(error)}")
        return {"status": "failure", "error": str(error)}


async def elaborate_text(input_text: str) -> str:
    """
    Call the /elaborator endpoint to elaborate the input text.

    This function sends a POST request to an external 'elaborator' service to enhance or reformat
    the given input text, intended for document content.

    Args:
        input_text (str): The text to be elaborated.

    Returns:
        str: The elaborated text returned by the service, or the original text if elaboration fails.
    """
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                "http://localhost:5000/elaborator",
                json={"input": input_text, "purpose": "Document"},
            )
            if response.status_code == 200:
                return response.json().get("message", input_text)
            else:
                return input_text
    except Exception as e:
        print(f"Error during elaboration: {str(e)}")
        return input_text


async def search_unsplash_image(query: str) -> Tuple[str | None, bytes | None]:
    """Search for an image on Unsplash and return its URL and raw bytes.

    Args:
        query (str): The search term to find an image (e.g., 'nature').

    Returns:
        Tuple[str | None, bytes | None]: The URL of the first image result and its raw bytes,
        or (None, None) if no image is found or an error occurs.
    """
    access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not access_key:
        print("Unsplash access key is missing. Please set UNSPLASH_ACCESS_KEY in your .env file.")
        return None, None

    url = f"https://api.unsplash.com/search/photos?query={query}&per_page=1"
    headers = {'Authorization': f'Client-ID {access_key}'}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    print(f"Unsplash API Error: {response.status}")
                    return None, None
                data = await response.json()
                if not data.get("results"):
                    print(f"No images found for query: {query}")
                    return None, None
                image_url = data["results"][0]["urls"]["regular"]

            async with session.get(image_url) as image_response:
                if image_response.status != 200:
                    print(f"Failed to download image from {image_url}: {image_response.status}")
                    return None, None
                image_bytes = await image_response.read()
                return image_url, image_bytes
    except Exception as e:
        print(f"Error searching Unsplash: {e}")
        return None, None

async def upload_image_to_drive(drive_service, image_bytes: bytes, file_name: str, mimetype: str) -> str | None:
    """Upload image bytes to Google Drive and return the public URL.

    Args:
        drive_service: Authenticated Google Drive API service instance.
        image_bytes (bytes): The image data in bytes.
        file_name (str): The name to give the file in Google Drive.
        mimetype (str): The MIME type of the image (e.g., 'image/jpeg').

    Returns:
        str | None: The public URL of the uploaded image, or None if upload fails.
    """
    try:
        file_metadata = {'name': file_name}
        media = MediaIoBaseUpload(io.BytesIO(image_bytes), mimetype=mimetype)
        uploaded_file = drive_service.files().create(
            body=file_metadata, media_body=media, fields='id'
        ).execute()
        file_id = uploaded_file.get('id')
        permission = {'type': 'anyone', 'role': 'reader'}
        drive_service.permissions().create(fileId=file_id, body=permission).execute()
        return f"https://drive.google.com/uc?id={file_id}"
    except Exception as e:
        print(f"Failed to upload image to Google Drive: {e}")
        return None

async def create_document(service, title: str) -> str:
    """Create a new Google Doc with a specified title and return its document ID.

    Args:
        service: Authenticated Google Docs API service.
        title (str): Title of the new document.

    Returns:
        str: The ID of the newly created Google Document.
    """
    try:
        document = service.documents().create(body={"title": title}).execute()
        print(f"Created document with title: {document['title']}")
        return document["documentId"]
    except HttpError as error:
        print(f"An error occurred while creating the document: {error}")
        raise

async def create_google_doc(content: Dict[str, Any]) -> Dict[str, Any]:
    """Create a Google Document with structured content and return its URL.

    Args:
        content (Dict[str, Any]): Structured content dictionary with title and sections.
            Expected format:
            {
                "title": "Document Title",
                "sections": [
                    {
                        "heading": "Section Title",
                        "heading_level": "H1" or "H2",
                        "paragraphs": ["Paragraph 1 text", "Paragraph 2 text"],
                        "bullet_points": ["Bullet 1 with **bold** text", "Bullet 2"],
                        "image_description": "Descriptive image search query"  # Optional
                    }
                ]
            }

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure.
            On success: {"status": "success", "result": {"response": str, "url": str}}
            On failure: {"status": "failure", "error": str}
    """
    try:
        print("Starting create_google_doc function...")
        docs_service = authenticate_docs()
        drive_service = authenticate_drive()
        document_id = await create_document(docs_service, content["title"])
        document_url = f"https://docs.google.com/document/d/{document_id}/edit"
        
        # Initialize request list and index
        requests = []
        current_index = 1

        # Add title
        title_text = content["title"] + "\n\n"
        requests.append({"insertText": {"location": {"index": current_index}, "text": title_text}})
        requests.append({
            "updateParagraphStyle": {
                "range": {"startIndex": current_index, "endIndex": current_index + len(content["title"])},
                "paragraphStyle": {"namedStyleType": "HEADING_1"},
                "fields": "namedStyleType"
            }
        })
        current_index += len(title_text)

        # Process each section
        for section in content["sections"]:
            # Add heading
            heading_text = section["heading"] + "\n"
            requests.append({"insertText": {"location": {"index": current_index}, "text": heading_text}})
            requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": current_index, "endIndex": current_index + len(section["heading"])},
                    "paragraphStyle": {"namedStyleType": f"HEADING_{'1' if section['heading_level'] == 'H1' else '2'}"},
                    "fields": "namedStyleType"
                }
            })
            current_index += len(heading_text)

            # Add paragraphs
            for para in section["paragraphs"]:
                para_text = para + "\n\n"
                requests.append({"insertText": {"location": {"index": current_index}, "text": para_text}})
                current_index += len(para_text)

            # Add bullet points with bold formatting
            for bullet in section["bullet_points"]:
                bullet_text = bullet + "\n"
                requests.append({"insertText": {"location": {"index": current_index}, "text": bullet_text}})
                requests.append({
                    "createParagraphBullets": {
                        "range": {"startIndex": current_index, "endIndex": current_index + len(bullet_text) - 1},
                        "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE"
                    }
                })
                # Apply bold formatting for **text**
                bold_starts = [i for i in range(len(bullet)) if bullet.startswith("**", i)]
                for i in range(0, len(bold_starts), 2):
                    start = bold_starts[i] + 2
                    end = bold_starts[i + 1] if i + 1 < len(bold_starts) else len(bullet)
                    if end > start:
                        requests.append({
                            "updateTextStyle": {
                                "range": {"startIndex": current_index + start, "endIndex": current_index + end},
                                "textStyle": {"bold": True},
                                "fields": "bold"
                            }
                        })
                current_index += len(bullet_text)

            # Add image if present
            if "image_description" in section and section["image_description"]:
                image_url, image_bytes = await search_unsplash_image(section["image_description"])
                if image_bytes:
                    drive_url = await upload_image_to_drive(
                        drive_service=drive_service,
                        image_bytes=image_bytes,
                        file_name=f"{section['image_description'].replace(' ', '_')}.jpg",
                        mimetype="image/jpeg"
                    )
                    if drive_url:
                        # Add newline before image
                        requests.append({"insertText": {"location": {"index": current_index}, "text": "\n"}})
                        current_index += 1
                        # Add image
                        requests.append({
                            "insertInlineImage": {
                                "location": {"index": current_index},
                                "uri": drive_url,
                                "objectSize": {
                                    "height": {"magnitude": 200, "unit": "PT"},
                                    "width": {"magnitude": 300, "unit": "PT"}
                                }
                            }
                        })
                        current_index += 1  # Image counts as 1 character
                        # Add newline after image
                        requests.append({"insertText": {"location": {"index": current_index}, "text": "\n"}})
                        current_index += 1
                    else:
                        print(f"Failed to upload image: {section['image_description']}")
                        requests.append({
                            "insertText": {
                                "location": {"index": current_index},
                                "text": f"[Image failed to load: {section['image_description']}]\n\n"
                            }
                        })
                        current_index += len(f"[Image failed to load: {section['image_description']}]\n\n")
                else:
                    print(f"No image found for: {section['image_description']}")
                    requests.append({
                        "insertText": {
                            "location": {"index": current_index},
                            "text": f"[No image found for: {section['image_description']}]\n\n"
                        }
                    })
                    current_index += len(f"[No image found for: {section['image_description']}]\n\n")

            # Add spacing between sections
            requests.append({"insertText": {"location": {"index": current_index}, "text": "\n"}})
            current_index += 1

        # Execute all requests in one batch
        if requests:
            print(f"Executing batchUpdate with {len(requests)} requests...")
            docs_service.documents().batchUpdate(documentId=document_id, body={"requests": requests}).execute()
            print("Document updated successfully.")

        return {
            "status": "success",
            "result": {
                "response": "Document created successfully",
                "url": document_url
            }
        }
    except Exception as e:
        print(f"Error in create_google_doc: {e}")
        return {"status": "failure", "error": str(e)}


async def add_event(
    start: str,
    end: str,
    timezone: str,
    attendees: Optional[List[str]] = None,
    description: Optional[str] = None,
    summary: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Add an event to Google Calendar.

    Args:
        start (str): Start datetime of the event in ISO format (e.g., '2023-10-27T10:00:00-07:00').
        end (str): End datetime of the event in ISO format (e.g., '2023-10-27T11:00:00-07:00').
        timezone (str): Timezone for the event (e.g., 'America/Los_Angeles').
        attendees (Optional[List[str]]): List of attendee email addresses (optional).
        description (Optional[str]): Event description (optional).
        summary (Optional[str]): Event summary or title (optional, defaults to 'No Title').

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure and event details.
                         On success, returns:
                         {
                             "status": "success",
                             "result": {
                                 "response": "Event created successfully",
                                 "event_id": str # URL to the created event in Google Calendar
                             }
                         }
                         On failure, returns:
                         {"status": "failure", "error": str(error)}
    """
    try:
        service = authenticate_calendar()
        event = {
            "summary": summary or "No Title",
            "description": description or "",
            "start": {
                "dateTime": start,
                "timeZone": timezone,
            },
            "end": {
                "dateTime": end,
                "timeZone": timezone,
            },
            "attendees": [{"email": email} for email in attendees] if attendees else [],
        }
        created_event = (
            service.events().insert(calendarId="primary", body=event).execute()
        )
        return {
            "status": "success",
            "result": {
                "response": "Event created successfully",
                "event_id": created_event.get("htmlLink"),
            },
        }
    except HttpError as error:
        print(f"Error adding event: {error}")
        return {"status": "failure", "error": str(error)}


async def search_events(query: str) -> Dict[str, Any]:
    """
    Search for events in Google Calendar based on a query.

    Args:
        query (str): Query string to search for in event titles and descriptions.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure and event search results.
                         On success, returns:
                         {
                             "status": "success",
                             "result": {
                                 "response": "Events found successfully",
                                 "events": List[Dict] # List of event resources
                             }
                         }
                         On failure, returns:
                         {"status": "failure", "error": str(error)}
    """
    try:
        service = authenticate_calendar()
        now = datetime.datetime.utcnow().isoformat() + "Z"
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                q=query,
                timeMin=now,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])
        return {
            "status": "success",
            "result": {"response": "Events found successfully", "events": events},
        }
    except HttpError as error:
        print(f"Error searching events: {error}")
        return {"status": "failure", "error": str(error)}


async def list_upcoming_events(days: int = 7) -> Dict[str, Any]:
    """
    List upcoming events for the next n days.

    Args:
        days (int): Number of days to look ahead for upcoming events (default is 7).

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure and upcoming events.
                         On success, returns:
                         {
                             "status": "success",
                             "result": {
                                 "response": "Upcoming events found successfully",
                                 "events": List[Dict] # List of event resources
                             }
                         }
                         On failure, returns:
                         {"status": "failure", "error": str(error)}
    """
    try:
        service = authenticate_calendar()
        now = datetime.datetime.utcnow().isoformat() + "Z"
        max_time = (datetime.datetime.utcnow() + timedelta(days=days)).isoformat() + "Z"
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now,
                timeMax=max_time,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])
        return {
            "status": "success",
            "result": {
                "response": "Upcoming events found successfully",
                "events": events,
            },
        }
    except HttpError as error:
        print(f"Error listing upcoming events: {error}")
        return {"status": "failure", "error": str(error)}


async def search_file_in_gdrive(query: str) -> Dict[str, Any]:
    """
    Search for files in Google Drive matching the query and return their URLs.

    Args:
        query (str): Query string to search for files by name in Google Drive.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure and file search results.
                         On success, returns:
                         {
                             "status": "success",
                             "result": {
                                 "response": "File search successful",
                                 "files": List[Dict] # List of file details with name and URL
                             }
                         }
                         On failure, returns:
                         {"status": "failure", "error": str(error)}
    """
    try:
        service = authenticate_drive()
        results = (
            service.files()
            .list(q=f"name contains '{query}'", fields="files(id, name)")
            .execute()
        )
        files = results.get("files", [])

        file_urls: List[Dict[str, str]] = [
            {
                "name": file["name"],
                "url": f"https://drive.google.com/file/d/{file['id']}/view",
            }
            for file in files
        ]

        return {
            "status": "success",
            "result": {"response": "File search successful", "files": file_urls},
        }
    except HttpError as error:
        print(f"Error searching files: {error}")
        return {"status": "failure", "error": str(error)}


async def search_folder(service, folder_name: str) -> str:
    """
    Search for a folder in Google Drive by name and return its folder ID.

    Args:
        service: Authenticated Google Drive API service.
        folder_name (str): Name of the folder to search for.

    Returns:
        str: The ID of the folder if found.
             Raises an exception if no folder is found or if there's an error during search.
    """
    try:
        query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder'"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        folders = results.get("files", [])
        if not folders:
            raise Exception(f"No folder found with the name '{folder_name}'.")
        return folders[0]["id"]
    except Exception as e:
        raise Exception(f"Error searching for folder: {e}")


async def upload_file_to_folder(
    service, file_path: str, folder_id: Optional[str]
) -> Dict[str, Any]:
    """
    Upload a file to a specified folder in Google Drive and return the file URL.

    Args:
        service: Authenticated Google Drive API service.
        file_path (str): Local path to the file to upload.
        folder_id (Optional[str]): ID of the Google Drive folder to upload to.
                                  If None, the file is uploaded to the root of Google Drive.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure and upload details.
                         On success, returns:
                         {
                             "status": "success",
                             "result": {
                                 "response": "File upload successful",
                                 "fileId": str, # ID of the uploaded file
                                 "url": str, # URL to view the uploaded file in Google Drive
                                 "folder_id": str # ID of the folder the file was uploaded to
                             }
                         }
                         On failure, returns:
                         {"status": "failure", "error": str(error)}
    """
    try:
        file_metadata = {
            "name": file_path.split("/")[-1],
            "parents": [folder_id] if folder_id else [],
        }
        media = MediaFileUpload(file_path, resumable=True)
        file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        file_id = file.get("id")
        file_url = f"https://drive.google.com/file/d/{file_id}/view"
        return {
            "status": "success",
            "result": {
                "response": "File upload successful",
                "fileId": file_id,
                "url": file_url,
                "folder_id": folder_id,
            },
        }
    except HttpError as error:
        print(f"Error uploading file: {error}")
        return {"status": "failure", "error": str(error)}


async def upload_file_to_gdrive(
    file_path: str, folder_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for a folder by name and upload a file to it. If no folder name is provided, upload to root.

    Args:
        file_path (str): Local path to the file to upload.
        folder_name (Optional[str]): Name of the Google Drive folder to upload to (optional).
                                     If provided, the function will search for the folder by name.
                                     If None, the file will be uploaded to the root of Google Drive.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure and upload details.
                         Delegates to `upload_file_to_folder` after finding the folder ID (if folder_name is given).
    """
    try:
        service = authenticate_drive()
        folder_id = None

        if folder_name:
            folder_id = await search_folder(service, folder_name)

        result = await upload_file_to_folder(service, file_path, folder_id)
        return result
    except Exception as e:
        print(f"Error in upload_file_to_gdrive: {e}")
        return {"status": "failure", "error": str(e)}


async def search_file(service, query: str) -> str:
    """
    Search for a file in Google Drive by query and return its file ID.

    Args:
        service: Authenticated Google Drive API service.
        query (str): Query string to search for files by name in Google Drive.

    Returns:
        str: The ID of the file if found.
             Raises an exception if no files are found matching the query or if there's an error during search.
    """
    try:
        results = (
            service.files()
            .list(q=f"name contains '{query}'", fields="files(id, name)")
            .execute()
        )
        files = results.get("files", [])
        if not files:
            raise Exception("No files found matching the query.")
        return files[0]["id"]
    except Exception as e:
        raise Exception(f"Error searching for file: {e}")


async def download_file(
    service,
    file_id: str,
    destination: Optional[str] = None,
    export_format: str = "application/pdf",
) -> Dict[str, Any]:
    """
    Download a file from Google Drive using its file ID. Supports exporting Docs Editors files to specified formats.

    Args:
        service: Authenticated Google Drive API service.
        file_id (str): ID of the file to download from Google Drive.
        destination (Optional[str]): Local directory path to save the downloaded file.
                                     If None, defaults to the user's Downloads directory.
        export_format (str): MIME type to export Google Docs, Sheets, or Slides files as (default is "application/pdf").
                             For binary files, this parameter is ignored.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure and download details.
                         On success, returns:
                         {
                             "status": "success",
                             "result": {
                                 "response": "File download successful",
                                 "file_name": str, # Name of the downloaded file
                                 "file_id": str, # ID of the downloaded file
                                 "download_path": str # Full path to the downloaded file on the local system
                             }
                         }
                         On failure, returns:
                         {"status": "failure", "error": str(error)}
    """
    try:
        if destination is None:
            home_dir = os.path.expanduser("~")
            destination = os.path.join(home_dir, "Downloads")

        if not os.path.exists(destination):
            os.makedirs(destination)

        file_metadata = service.files().get(fileId=file_id).execute()
        file_name = file_metadata["name"]

        file_path = os.path.join(
            destination,
            f"{file_name}.pdf" if export_format == "application/pdf" else file_name,
        )

        if file_metadata["mimeType"].startswith("application/vnd.google-apps"):
            print(f"Exporting '{file_name}' as '{export_format}'")
            request = service.files().export_media(
                fileId=file_id, mimeType=export_format
            )
        else:
            print(f"Downloading binary file '{file_name}'")
            request = service.files().get_media(fileId=file_id)

        with open(file_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download progress: {int(status.progress() * 100)}%")

        return {
            "status": "success",
            "result": {
                "response": "File download successful",
                "file_name": file_name,
                "file_id": file_id,
                "download_path": file_path,
            },
        }

    except HttpError as http_error:
        return {"status": "failure", "error": f"HTTP Error: {http_error}"}
    except Exception as e:
        return {"status": "failure", "error": str(e)}


async def search_and_download_file_from_gdrive(
    file_name: str, destination: str
) -> Dict[str, Any]:
    """
    Search for a file in Google Drive by name and then download it to the specified destination.

    Args:
        file_name (str): Name of the file to search for and download.
        destination (str): Local directory path to save the downloaded file.

    Returns:
        Dict[str, Any]: Status dictionary indicating success or failure and download details.
                         Delegates to `search_file` and `download_file` functions.
    """
    try:
        service = authenticate_drive()
        file_id = await search_file(service, file_name)
        result = await download_file(service, file_id, destination)
        return result
    except Exception as e:
        print(f"Error in search and download: {e}")
        return {"status": "failure", "error": str(e)}


import asyncio
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assuming these are defined elsewhere
async def create_spreadsheet(service, title: str) -> Dict[str, Any]:
    def sync_create():
        spreadsheet = {"properties": {"title": title}}
        return service.spreadsheets().create(body=spreadsheet).execute()
    return await asyncio.to_thread(sync_create)

async def get_spreadsheet(service, spreadsheet_id: str) -> Dict[str, Any]:
    def sync_get():
        return service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    return await asyncio.to_thread(sync_get)

async def rename_sheet(service, spreadsheet_id: str, sheet_id: int, sheet_title: str) -> None:
    def sync_rename():
        requests = [{
            "updateSheetProperties": {
                "properties": {"sheetId": sheet_id, "title": sheet_title},
                "fields": "title"
            }
        }]
        return service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body={"requests": requests}).execute()
    await asyncio.to_thread(sync_rename)

async def add_sheet(service, spreadsheet_id: str, sheet_title: str) -> None:
    def sync_add():
        requests = [{"addSheet": {"properties": {"title": sheet_title}}}]
        return service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body={"requests": requests}).execute()
    await asyncio.to_thread(sync_add)

async def store_data_in_spreadsheet(service, spreadsheet_id: str, values: List[List[Any]], range_name: str) -> Dict[str, Any]:
    def sync_store():
        body = {"values": values}
        result = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption="RAW",
            body=body
        ).execute()
        return {"status": "success", "result": {"response": "Data stored", "updated_cells": result.get("updatedCells")}}
    return await asyncio.to_thread(sync_store)

async def format_headers(service, spreadsheet_id: str, sheet_title: str) -> None:
    async def get_sheet_id():
        spreadsheet = await get_spreadsheet(service, spreadsheet_id)
        for sheet in spreadsheet['sheets']:
            if sheet['properties']['title'] == sheet_title:
                return sheet['properties']['sheetId']
        return None

    sheet_id = await get_sheet_id()
    def sync_format():
        requests = [{
            "repeatCell": {
                "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1},
                "cell": {"userEnteredFormat": {"textFormat": {"bold": True}}},
                "fields": "userEnteredFormat.textFormat.bold"
            }
        }]
        return service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body={"requests": requests}).execute()
    await asyncio.to_thread(sync_format)

async def apply_table_borders(service, spreadsheet_id: str, sheet_title: str, num_rows: int, num_columns: int) -> None:
    async def get_sheet_id():
        spreadsheet = await get_spreadsheet(service, spreadsheet_id)
        for sheet in spreadsheet['sheets']:
            if sheet['properties']['title'] == sheet_title:
                return sheet['properties']['sheetId']
        return None

    sheet_id = await get_sheet_id()
    def sync_borders():
        border_style = {"style": "SOLID", "color": {"red": 0.0, "green": 0.0, "blue": 0.0}}
        requests = [{
            "updateBorders": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": num_rows,
                    "startColumnIndex": 0,
                    "endColumnIndex": num_columns
                },
                "top": border_style,
                "bottom": border_style,
                "left": border_style,
                "right": border_style,
                "innerHorizontal": border_style,
                "innerVertical": border_style
            }
        }]
        return service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body={"requests": requests}).execute()
    await asyncio.to_thread(sync_borders)

async def create_google_sheet(title: str, sheets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Creates a Google Spreadsheet with multiple sheets, populates them with data, and applies formatting.

    Args:
        title (str): The title of the spreadsheet.
        sheets (List[Dict[str, Any]]): List of sheets, each containing "title" and "table" with "headers" and "rows".

    Returns:
        Dict[str, Any]: {"status": "success", "result": {"response": str, "spreadsheetUrl": str, "spreadsheet_id": str}}
                        or {"status": "failure", "error": str}
    """
    try:
        service = authenticate_sheets()
        data = {"title": title, "sheets": sheets}

        print("Creating spreadsheet...")
        create_response = await create_spreadsheet(service, data["title"])
        if "spreadsheetId" not in create_response:
            return {"status": "failure", "error": "Failed to create spreadsheet"}
        
        spreadsheet_id = create_response["spreadsheetId"]
        spreadsheet_url = create_response.get("spreadsheetUrl", "N/A")

        logger.info("Getting spreadsheet metadata...")
        spreadsheet = await get_spreadsheet(service, spreadsheet_id)
        default_sheet_id = spreadsheet['sheets'][0]['properties']['sheetId']

        for i, sheet in enumerate(data["sheets"]):
            sheet_title = sheet["title"]
            table = sheet["table"]
            headers = table["headers"]
            rows = table["rows"]
            values = [headers] + rows
            num_rows = len(rows) + 1
            num_columns = len(headers)

            if i == 0:
                logger.info(f"Renaming default sheet to {sheet_title}...")
                await rename_sheet(service, spreadsheet_id, default_sheet_id, sheet_title)
            else:
                logger.info(f"Adding sheet {sheet_title}...")
                await add_sheet(service, spreadsheet_id, sheet_title)

            range_name = f"'{sheet_title}'!A1"
            logger.info(f"Writing data to {sheet_title}...")
            store_response = await store_data_in_spreadsheet(service, spreadsheet_id, values, range_name)
            if store_response["status"] != "success":
                return store_response

            logger.info(f"Formatting headers in {sheet_title}...")
            await format_headers(service, spreadsheet_id, sheet_title)
            logger.info(f"Applying borders in {sheet_title}...")
            await apply_table_borders(service, spreadsheet_id, sheet_title, num_rows, num_columns)

        return {
            "status": "success",
            "result": {
                "response": "Spreadsheet created successfully with data",
                "spreadsheetUrl": spreadsheet_url,
                "spreadsheet_id": spreadsheet_id
            }
        }
    except Exception as e:
        logger.error(f"Error in create_google_sheet: {str(e)}")
        return {"status": "failure", "error": str(e)}

# Removed get_chat_history as it's now handled by server.chat.functions.get_chat_history

