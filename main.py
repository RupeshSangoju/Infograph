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
            points = [point.strip() for point in content.split("\n") if point.strip()]
            if len(points) < num_points:
                points.extend(content.split(". "))  # Fallback if new lines are missing
            return points[:num_points]
    return []

def generate_headings(points):
    """Generate short headings (2-3 words) for each point using Perplexity API."""
    if not points:
        return []
    
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Generate a short heading (2-3 words) for each fact listed below."},
            {"role": "user", "content": "\n".join(points)}
        ],
        "max_tokens": 100,
        "temperature": 0.3,
        "top_p": 0.9
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    response = requests.post(PERPLEXITY_URL, json=payload, headers=headers)

    if response.status_code == 200:
        response_json = response.json()
        headings_content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        headings = [heading.strip() for heading in headings_content.split("\n") if heading.strip()]
        return headings[:len(points)]
    return [f"Point {i+1}" for i in range(len(points))]

def summarize_into_points(text, num_points=random.randint(4, 8)):
    """Summarize text into bullet points using BART."""
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    summary_points = [point.strip() for point in summary.split("\n") if point.strip()]
    if len(summary_points) < num_points:
        summary_points.extend(summary.split(". "))
    
    return summary_points[:num_points]

@app.get("/generate_summary/")
async def generate_summary_endpoint(topic: str = None, text: str = None):
    """FastAPI endpoint to summarize a topic or given text with headings."""
    if not topic and not text:
        raise HTTPException(status_code=400, detail="Either 'topic' or 'text' must be provided.")
    
    if topic:
        points = fetch_info_perplexity(topic, num_points)
    else:
        points = summarize_into_points(text, num_points)
    
    headings = generate_headings(points)
    formatted_output = "\n\n".join([f"**{headings[i]}**\n• {points[i]}" for i in range(len(points)) if i < len(headings)])
    
    return {"summary": formatted_output}
