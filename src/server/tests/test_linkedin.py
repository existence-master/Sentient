import pdfplumber
import requests
import json
import os

def extract_text_from_pdf(pdf_path):
    """
    Extract text from all pages of a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Extracted text from the PDF.
    Raises:
        FileNotFoundError: If the PDF file doesn’t exist.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def generate_json_from_text(pdf_path):
    """
    Extract text from a PDF and use Llama 3.2:3b from Ollama to generate a JSON response.
    
    Args:
        pdf_path (str): Path to the LinkedIn profile PDF.
    Returns:
        dict: Parsed JSON data from the AI model, or None if parsing fails.
    """
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Construct the prompt
    prompt = f"""
    You are an AI assistant tasked with extracting structured information from a LinkedIn profile text. The text provided is extracted from a LinkedIn profile PDF and may contain sections such as Contact, Summary, Experience, Education, Skills, Languages, Certifications, and Honors-Awards.

    Your job is to analyze the text and extract the relevant information into a JSON object with the following structure:

    {{
    "contact": {{
        "email": "string",
        "phone": "string",
        "linkedin": "string",
        "other_links": ["string"]
    }},
    "summary": "string",
    "experience": [
        {{
        "company": "string",
        "title": "string",
        "dates": "string",
        "location": "string",
        "description": "string"
        }}
    ],
    "education": [
        {{
        "institution": "string",
        "degree": "string",
        "dates": "string"
        }}
    ],
    "skills": ["string"],
    "languages": [
        {{
        "language": "string",
        "proficiency": "string"
        }}
    ],
    "certifications": ["string"],
    "honors_awards": ["string"]
    }}

    Please ensure that:
    - All fields are populated if the information is present in the text.
    - If a section is missing, omit that key from the JSON object (do not include it with null or empty values).
    - For "experience" and "education", include all relevant entries in the arrays.
    - For "languages", parse both the language and proficiency if available.
    - Be accurate and do not invent information that isn’t explicitly stated in the text.
    - Output only the JSON object and no additional text.

    Here is the text to analyze:

    {text}
    """
    
    print(text)
    
    # Send the prompt to the Ollama API
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        })
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get the generated text
        result = response.json()
        generated_text = result.get("response", "")
        
        # Parse the generated text as JSON
        try:
            data = json.loads(generated_text)
            return data
        except json.JSONDecodeError:
            print("Error: The model did not return valid JSON.")
            return None
    except requests.RequestException as e:
        print(f"Error communicating with Ollama API: {e}")
        return None

if __name__ == "__main__":
    # Replace with the actual path to your LinkedIn profile PDF
    pdf_path = "Profile.pdf"
    
    try:
        structured_data = generate_json_from_text(pdf_path)
        if structured_data:
            print(json.dumps(structured_data, indent=2))
        else:
            print("Failed to generate structured data.")
    except Exception as e:
        print(f"Error: {e}")