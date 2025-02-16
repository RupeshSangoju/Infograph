import requests
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

def fetch_info_perplexity_type1(topic):
    """Fetch information using Perplexity AI API (Type 1 - Essay format)."""
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": f"Provide a 300-word essay about {topic}."}
        ],
        "max_tokens": 300,
        "temperature": 0.2,
        "top_p": 0.9,
        "return_images": False,
        "return_related_questions": False
    }
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    response = requests.post(PERPLEXITY_URL, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if data.get("summary_points"):  # Check if output is empty
            return data["summary_points"]
    
    return None  # Trigger fallback if empty or failed


def fetch_info_perplexity_type2(topic, num_points):
    """Fetch structured key facts using Perplexity AI API (Type 2)."""
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Provide an informative and structured response."},
            {"role": "user", "content": f"List {num_points} key facts about {topic}. Each fact should be in a separate line and no more than 20 words long."}
        ],
        "max_tokens": 400,
        "temperature": 0.3,
        "top_p": 0.9,
        "return_images": False,
        "return_related_questions": False
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    response = requests.post(PERPLEXITY_URL, json=payload, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if data.get("summary_points"):  # Check if output is empty
            return data["summary_points"]
    
    return None  # If both methods fail
