import requests
from bs4 import BeautifulSoup
from googlesearch import search

def scrape_wikipedia(topic, num_points=5):
    """Scrape Wikipedia for key information about the topic."""
    url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None  # Wikipedia page not found

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.select("p")  

    key_points = []
    for para in paragraphs:
        text = para.get_text().strip()
        if text and len(text.split()) > 10:  
            key_points.append(text)
        if len(key_points) >= num_points:
            break

    return key_points if key_points else None

def scrape_google_search(topic, num_points=5):
    """Use Google search to find relevant Wikipedia content."""
    query = f"{topic} key facts site:wikipedia.org"
    try:
        search_results = list(search(query, num_results=3))
        for url in search_results:
            if "wikipedia.org" in url:
                return scrape_wikipedia(url, num_points)
    except Exception as e:
        print(f"Google Search failed: {e}")
    
    return None
