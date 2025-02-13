from fastapi import FastAPI, HTTPException
import requests
import os
from transformers import BartTokenizer, BartForConditionalGeneration
from dotenv import load_dotenv
import random  

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# Load BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Initialize FastAPI app
app = FastAPI()
num_points = random.randint(4, 8)

def fetch_info_perplexity(topic, num_points):
    """Fetch structured information with headings using Perplexity AI API."""
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
        response_json = response.json()
        choices = response_json.get("choices", [{}])
        if choices:
            content = choices[0].get("message", {}).get("content", "").strip()
            delta_content = choices[0].get("delta", {}).get("content", "").strip()
            raw_text = content if content else delta_content
            
            # Ensure each fact is in a new line
            points = raw_text.split("\n")
            if len(points) < num_points:
                points = raw_text.split(". ")  # Fallback if new lines are missing
            
            # Generate a heading for each point
            headings_payload = {
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": "Generate a short heading (2-3 words) for each fact listed below."},
                    {"role": "user", "content": "\n".join(points)}
                ],
                "max_tokens": 100,
                "temperature": 0.3,
                "top_p": 0.9
            }
            
            headings_response = requests.post(PERPLEXITY_URL, json=headings_payload, headers=headers)
            if headings_response.status_code == 200:
                headings_json = headings_response.json()
                headings_content = headings_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                headings = headings_content.split("\n")
            else:
                headings = [f"Point {i+1}" for i in range(len(points))]  # Fallback generic headings
            
            formatted_output = "\n".join([f"**{headings[i].strip()}**\n• {points[i].strip()}" for i in range(min(len(points), len(headings)))])
            return formatted_output

        return "Error: Unexpected response format."
    
    return "Error fetching data from Perplexity API."



def summarize_into_points(text, num_points=random.randint(4, 8)):
    """Summarize text into bullet points using BART."""
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Split by new lines or periods to ensure proper point separation
    summary_points = summary.split("\n")
    if len(summary_points) < num_points:
        summary_points = summary.split(". ")

    # Limit to the required number of points and ensure new lines
    summary_points = summary_points[:num_points]
    formatted_summary = "\n".join([f"• {point.strip()}" for point in summary_points if point.strip()])
    
    return formatted_summary


@app.get("/generate_summary/")
async def generate_summary_endpoint(topic: str = None, text: str = None):
    """FastAPI endpoint to summarize a topic or given text."""
    if not topic and not text:
        raise HTTPException(status_code=400, detail="Either 'topic' or 'text' must be provided.")
    
    if topic:
        text = fetch_info_perplexity(topic, num_points)
    
    summary = summarize_into_points(text, num_points)
    return {"summary": summary}
