import requests
import random
import os
from gensim.summarization.summarizer import summarize

# Perplexity API configuration
API_KEY = os.getenv("PERPLEXITY_API_KEY")  # Fetch the API key from an environment variable
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

def fetch_info_from_perplexity(topic, num_points):
    """Fetch information using Perplexity AI API"""
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": f"Provide {num_points} concise points about {topic}."}
        ],
        "max_tokens": 300,
        "temperature": 0.2,
        "top_p": 0.9,
        "return_images": False,
        "return_related_questions": False
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(PERPLEXITY_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        return "Error fetching data from Perplexity API."

def generate_summary_from_text(text, num_points):
    """Generate a summary from the given text."""
    summary = summarize(text, word_count=num_points * 15)  # Approx. 20 words per point
    points = summary.split("\n")  # Split into bullet points
    
    # Ensure we have at least 5 points, if fewer we repeat/modify the points
    while len(points) < num_points:
        points.append(f"Additional point: {points[random.randint(0, len(points)-1)]}")
    
    return points[:num_points]

def main():
    user_input = input("Enter a topic name or paste text: ").strip()
    
    # Randomly choose between 5 to 7 points
    num_points = random.randint(5, 7)
    
    # Check if input seems to be a topic (short length)
    if len(user_input.split()) < 5:  # Assume it's a topic if it's short
        print("Fetching details from Perplexity API...")
        text = fetch_info_from_perplexity(user_input, num_points)
    else:
        text = user_input  # Directly use the input text
    
    # Generate summary points based on the content
    summary_points = generate_summary_from_text(text, num_points)
    
    print("\nGenerated Infographic Points:")
    for i, point in enumerate(summary_points, start=1):
        print(f"{i}. {point}")

if __name__ == "__main__":
    main()
