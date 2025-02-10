from fastapi import FastAPI, HTTPException
import requests
import random
import re
import os
from gensim.summarization.summarizer import summarize
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# Initialize FastAPI app
app = FastAPI()

def fetch_info_from_perplexity(topic, num_points):
    """Fetch information using Perplexity AI API with structured output and error handling."""
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Provide an informative and structured response."},
            {"role": "user", "content": f"List {num_points} key facts about {topic}. Each fact should be in a separate line and no more than 20 words long."}
        ],
        "max_tokens": 400,  # Reduce to prevent truncation
        "temperature": 0.3,
        "top_p": 0.9,
        "return_images": False,
        "return_related_questions": False
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(PERPLEXITY_URL, json=payload, headers=headers)

  #  print(f"API Response Code: {response.status_code}")  # Debugging
   # print("Full API Response JSON:", response.json())  # Print full response

    if response.status_code == 200:
        response_json = response.json()
        
        # Try extracting content
        choices = response_json.get("choices", [{}])
        if choices:
            content = choices[0].get("message", {}).get("content", "").strip()
            delta_content = choices[0].get("delta", {}).get("content", "").strip()
            final_content = content if content else delta_content  # Prioritize content

            if not final_content:
                return "Error: Perplexity returned an empty response. Try a different query or check API permissions."
            
            return final_content

        return "Error: Unexpected response format."
    
    return "Error fetching data from Perplexity API."
def preprocess_text(text):
    """Cleans text before processing (removes extra spaces, but keeps formatting)."""
    return re.sub(r'\s+', ' ', text).strip()

def generate_summary(text, num_points):
    """Generate a summary with unique points."""
    try:
        text = preprocess_text(text)  # Clean text before summarization
        
        if text.count('.') < 2:
            print("Text lacks proper sentence structure. Using fallback method.")
            summary = [' '.join(text.split()[:num_points * 15])]  # Take first X words
            return remove_duplicates(summary)  # Ensure unique points

        print(f"Summarizing the text: {text[:200]}...")  # Log first 200 characters
        
        summary = summarize(text, word_count=num_points * 15)  # Gensim summarization

        if not summary.strip():
            print("Summarization failed. Using alternative method.")
            summary = ' '.join(text.split()[:num_points * 15])  # Fallback

        points = summary.split("\n")

        unique_points = remove_duplicates(points)  # Ensure unique summary points

        if len(unique_points) < num_points:
            additional_points = text.split(". ")  # Split by sentence
            for point in additional_points:
                if point not in unique_points and len(unique_points) < num_points:
                    unique_points.append(point.strip())

        return unique_points[:num_points]  # Return exactly `num_points` unique points
    
    except Exception as e:
        print(f"Error while generating summary: {e}")
        return ["Error generating summary."]

def remove_duplicates(summary_list):
    """Removes duplicate summary points while preserving order and keeping hyphens in years."""
    seen = set()
    unique_summary = []
    
    for point in summary_list:
        # Normalize text but KEEP hyphens between digits (e.g., 2019-20)
        normalized_point = re.sub(r'(\d{4})-(\d{2})', r'\1-\2', point.strip().lower())  # Preserve year format
        normalized_point = re.sub(r'\s+', ' ', normalized_point)  # Normalize spaces

        if normalized_point not in seen:
            seen.add(normalized_point)
            unique_summary.append(point)  # Append original text
    
    return unique_summary

@app.get("/generate_summary/")
async def generate_summary_endpoint(topic: str = None, text: str = None):
    """FastAPI endpoint to generate a summary based on input."""
    if not topic and not text:
        raise HTTPException(status_code=400, detail="Either 'topic' or 'text' must be provided.")
    
    num_points = random.randint(5, 8)  # Generate 5-8 points randomly

    if topic:
        text = fetch_info_from_perplexity(topic, num_points)
    
    summary_points = generate_summary(text, num_points)
    
    return {"summary_points": summary_points}
