import requests
import json
import os
import re
import textract

def extract_text_from_pdf(pdf_path):
    text = textract.process(pdf_path, encoding='utf-8')
    return text.decode('utf-8')

def extract_json_from_response(response_text):
    """
    Extract and sanitize JSON object from model response.
    """
    # Match the first { and the last } to extract the largest JSON block
    start = response_text.find('{')
    end = response_text.rfind('}')
    
    if start == -1 or end == -1 or start >= end:
        print("Error: No JSON object boundaries found.")
        return None

    json_str = response_text[start:end+1]

    # Sanitize common mistakes
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)  # remove trailing commas

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Extracted string preview:\n", json_str[:500])
        return None

def generate_json_from_text(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    # text = preprocess_text_for_model(text)
    print(text)
    
    format = {
    "type": "object",
    "properties": {
        "contact": {
            "type": "object",
            "properties": {
                "email": {"type": "string"},
                "phone": {"type": "string"},
                "linkedin": {"type": "string"},
                "other_links": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["email", "phone", "linkedin", "other_links"],
            "additionalProperties": False
        },
        "summary": {"type": "string"},
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "title": {"type": "string"},
                    "dates": {"type": "string"},
                    "location": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["company", "title", "dates", "location", "description"],
                "additionalProperties": False
            }
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "institution": {"type": "string"},
                    "degree": {"type": "string"},
                    "dates": {"type": "string"}
                },
                "required": ["institution", "degree", "dates"],
                "additionalProperties": False
            }
        },
        "skills": {
            "type": "array",
            "items": {"type": "string"}
        },
        "languages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "proficiency": {"type": "string"}
                },
                "required": ["language", "proficiency"],
                "additionalProperties": False
            }
        },
        "certifications": {
            "type": "array",
            "items": {"type": "string"}
        },
        "honors_awards": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["contact", "summary", "experience", "education", "skills", "languages", "certifications", "honors_awards"],
    "additionalProperties": False
}

    
    prompt = f"""
You are an AI assistant tasked with extracting structured information from a LinkedIn profile text.

Your job is to return a valid JSON object that strictly follows the provided schema and accurately represents the content in the input text.

Instructions:

1. Extract only what is explicitly present in the text.
2. If a section is present, include it in the output and populate it fully. If not present, omit it from the output.
3. Do not hallucinate, infer, or guess missing information.
4. The `summary` may appear in fragments across multiple places. Merge them into a coherent and complete `summary` string without truncation.
5. In sections like `skills`, `certifications`, `honors_awards`, or `languages`, even if bullet points or formatting is inconsistent, extract all valid values into appropriate lists.
6. Ensure nested structures (`experience`, `education`, `languages`) match the JSON schema, including arrays of objects with required fields.
7. Do not include empty arrays or null values.
8. Avoid duplicate entries in any list fields.
9. Only return the JSON object. No additional commentary or text.

Edge Cases to Handle:

- Scattered or repeated section headers (e.g. multiple 'Summary' fragments).
- Line breaks or symbols breaking up phrases.
- Unexpected bullet styles (e.g. '-', '*', '•', or numbered lists).
- Duplicated or partial entries (only keep complete and unique records).
- Typo variations in section headers (e.g. "Honors-Awards", "Honours and Awards", etc.).

Here is the profile text to analyze:

{text}

"""

    print("Prompt length:", len(prompt + text))

    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "format": format,
            "options": {
                "num_ctx": 12000
            },
            "stream": False
        })
        response.raise_for_status()
        result = response.json()
        raw_output = result.get("response", "")
        return extract_json_from_response(raw_output)
    
    except requests.RequestException as e:
        print(f"Error communicating with Ollama API: {e}")
        return None


if __name__ == "__main__":
    pdf_path = "Profile.pdf"
    try:
        structured_data = generate_json_from_text(pdf_path)
        if structured_data:
            print(json.dumps(structured_data, indent=2))
        else:
            print("Failed to generate structured data.")
    except Exception as e:
        print(f"Error: {e}")
