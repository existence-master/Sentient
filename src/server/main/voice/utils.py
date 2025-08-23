import httpx
import logging
from main.config import GOOGLE_API_KEY

logger = logging.getLogger(__name__)

async def translate_text(text: str, target_language: str, source_language: str = None) -> str:
    """
    Translates text using the Google Translate API.
    """
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not set. Cannot perform translation.")
        raise ValueError("Translation service is not configured.")

    if not text:
        return ""

    url = "https://translation.googleapis.com/language/translate/v2"
    
    params = {
        "q": text,
        "target": target_language,
        "key": GOOGLE_API_KEY,
        "format": "text"
    }
    if source_language:
        params["source"] = source_language

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, params=params)
            response.raise_for_status()
            result = response.json()
            
            translated_text = result['data']['translations'][0]['translatedText']
            logger.info(f"Translated '{text[:30]}...' to '{target_language}': '{translated_text[:30]}...'")
            return translated_text
        except httpx.HTTPStatusError as e:
            logger.error(f"Google Translate API error: {e.response.status_code} - {e.response.text}")
            # Fallback to original text on error
            return text
        except Exception as e:
            logger.error(f"Error during translation: {e}", exc_info=True)
            # Fallback to original text on error
            return text