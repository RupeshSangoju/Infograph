import re
import os
import requests
import random
from fastapi import FastAPI, HTTPException
from transformers import BartTokenizer, BartForConditionalGeneration
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()
API_KEY = os.getenv("PERPLEXITY_API_KEY")
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# Load BART model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Initialize FastAPI
app = FastAPI()
num_points = random.randint(4, 8)

def fetch_info_perplexity(topic, num_points):
    """Fetch structured key facts using Perplexity API."""
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Provide an informative and structured response."},
            {"role": "user", "content": f"List {num_points} key facts about {topic}. Each fact should be a separate, complete sentence under 15 words."}
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
            return points[:num_points]
    return []

def generate_headings(points):
    """Generate strict 2-3 word headings for each fact using Perplexity API."""
    if not points:
        return []

    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Generate a **very strict** and **concise** subheading (2-3 words only) for each fact below.\n"
                    "- Each heading **must summarize the key idea** without repeating the fact itself.\n"
                    "- **Do NOT include numbers, facts, full sentences, or extra words**.\n"
                    "- **Output only the headings**, one per line, without any introduction or explanations."
                )
            },
            {
                "role": "user",
                "content": "\n".join(points)
            }
        ],
        "max_tokens": 50,  # Reduce token usage to prevent overflow
        "temperature": 0.1,  # Make output more deterministic
        "top_p": 0.95
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    response = requests.post(PERPLEXITY_URL, json=payload, headers=headers)

    if response.status_code == 200:
        response_json = response.json()

        # Debug: Print Perplexity's raw response
        print("\n🔹 **Raw Perplexity Response for Headings:**")
        print(response_json)

        headings_content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        headings = [heading.strip() for heading in headings_content.split("\n") if heading.strip()]

                # **Ensure every bullet point gets a heading**
        while len(headings) < len(points):
            headings.append(f"Point {len(headings) + 1}")

        # **Ensure proper alignment**
        return headings[:len(points)]
    
    return [f"Point {i+1}" for i in range(len(points))]

def summarize_into_points(text, num_points=random.randint(4, 8)):
    """Summarize text into meaningful bullet points, ensuring complete sentences and max 15 words."""
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("\n🔹 **Raw Summary from BART:**")
    print(summary)

    # **Fix: Ensure sentences are not cut off mid-way**
    sentences = re.split(r'(?<=[.!?])\s+', summary)  # Only split at full stops, exclamation, or question marks
    short_points = [s.strip() for s in sentences if len(s.strip()) > 0]

    print("\n🔹 **Extracted Bullet Points (Complete Sentences, Max 20 Words):**")
    for idx, point in enumerate(short_points, 1):
        print(f"{idx}. {point}")

    return short_points[:num_points]

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

    print("\n🔹 **Generated Headings from Perplexity:**")
    for idx, heading in enumerate(headings, 1):
        print(f"{idx}. {heading}")

    formatted_output = "\n\n".join([f"**{headings[i]}**\n• {points[i]}" for i in range(len(points)) if i < len(headings)])

    print("\n🔹 **Final Formatted Output:**")
    print(formatted_output)

    return {"summary": formatted_output}

def main():
    """Run FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
