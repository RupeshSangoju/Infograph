import re
import random
from summa import summarizer as text_rank_summarizer





def preprocess_text(text):
    """Clean text before processing."""
    return re.sub(r'\s+', ' ', text).strip()

def generate_summary(text):
    """Generate a summary using TextRank."""
    text = preprocess_text(text)
    num_points = random.randint(5, 8)  # Random number of points
    return text_rank_summarizer.summarize(text, words=num_points * 15)

text = """
In a case study evaluating student performance analysts use simple linear regression to examine the relationship between study hours and exam scores. By collecting data on the number of hours students studied and their corresponding exam results the analysts developed a model that reveal correlation, for each additional hour spent studying, students exam scores increased by an average of 5 points. This case highlights the utility of simple linear regression in understanding and improving academic performance.
Another case study focus on marketing and sales where businesses uses simple linear regression to forecast sales based on historical data particularly examining how factors like advertising expenditure influence revenue. By collecting data on past advertising spending and corresponding sales figures analysts develop a regression model that tells the relationship between these variables. For instance if the analysis reveals that for every additional dollar spent on advertising sales increase by $10. This predictive capability enables companies to optimize their advertising strategies and allocate resources effectively.
"""

# Summarize into 3 points
num_points = 5
summary = generate_summary(text)

print("Summary:")
print(summary)
