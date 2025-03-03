from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import re

app = FastAPI()
@app.post("/")
def read_root():
    return {"message": "Welcome to the chatbot API"}

# Step 1: Extract text from PDF and create a knowledge base
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_knowledge_base(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    # Split text into chunks (e.g., paragraphs or sections)
    knowledge_base = text.split("\n\n")  # Adjust based on your PDF structure
    return knowledge_base

# Step 2: Vectorize the knowledge base for similarity search
def vectorize_knowledge_base(knowledge_base):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(knowledge_base)
    return vectorizer, tfidf_matrix

# Step 3: Search the knowledge base for relevant answers
def search_knowledge_base(query, vectorizer, tfidf_matrix, knowledge_base, threshold=0.2):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_match_index = np.argmax(similarities)
    if similarities[best_match_index] > threshold:
        return knowledge_base[best_match_index]
    return None

# Step 4: Query DeepSeek API
def query_deepseek_api(prompt):
    api_key = "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"  # Replace with your actual API key
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
    }
    payload = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        return response.json()
    else:
        # Handle errors
        error_response = response.json()
        print(f"DeepSeek API Error: {error_response}")
        return {"error": "Failed to retrieve API response. Please try again later."}

# Step 5: Check if the query is related to mining
def is_mining_related(query):
    mining_keywords = [
        "mining", "coal", "mine", "safety", "regulation", "law", "act", "mineral", 
        "excavation", "drilling", "ventilation", "explosive", "hazard", "worker", 
        "environment", "rescue", "geology", "seam", "shaft", "tunnel", "ore", 
        "extraction", "quarry", "underground", "surface", "mining equipment",
        "metals", "ores", "processing", "refining", "conveyor", "open-pit", 
        "deep mining", "strip mining", "gold", "diamond", "copper", "zinc", "lead", 
        "nickel", "bauxite", "iron ore", "uranium", "rare earth elements", 
        "smelting", "beneficiation", "tailings", "waste disposal", "reclamation", 
        "mine closure", "mine planning", "rock mechanics", "hydrogeology", 
        "drainage", "mine surveying", "land subsidence", "mine ventilation", 
        "gas detection", "methane", "dust control", "fumes", "cyanide", "leaching", 
        "heap leaching", "mineral rights", "royalty", "permit", "lease", "licensing", 
        "blast", "explosives", "tunneling", "mining operations", "dredging", "placer mining",
        "acid mine drainage", "mine rehabilitation", "mining machinery", "mine safety act", 
        "occupational health", "mine workers", "mining corporation", "government regulations",
        "small-scale mining", "artisanal mining", "illegal mining", "mine accidents", 
        "stripping ratio", "mine slope stability", "groundwater control", "ore grade", 
        "sampling", "core drilling", "exploration", "surveying", "rock blasting"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in mining_keywords)

# Step 6: Remove Markdown
def remove_markdown(text):
    """Remove Markdown formatting from text."""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold (**text**)
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italics (*text*)
    text = re.sub(r'#+ ', '', text)              # Remove headings (#, ##, ###)
    text = re.sub(r'- ', '', text)               # Remove list dashes (- item)
    text = re.sub(r'\n+', '\n', text)            # Normalize multiple newlines
    return text.strip()

# Step 7: Chatbot function
def chatbot(query, vectorizer, tfidf_matrix, knowledge_base):
    if not is_mining_related(query):
        return "Please ask questions related to mining."

    # Search the knowledge base
    answer = search_knowledge_base(query, vectorizer, tfidf_matrix, knowledge_base)
    if answer:
        return remove_markdown(answer)  # Convert Markdown to plain text

    # If not found, generate a general answer using DeepSeek API
    prompt = f"Generate a general answer related to mining laws for the query: {query}"
    response = query_deepseek_api(prompt)

    # Check if response is valid before accessing elements
    if response and 'choices' in response and response['choices']:
        raw_answer = response['choices'][0]['message']['content']
        return remove_markdown(raw_answer)  # Convert Markdown to plain text
    else:
        return "I'm sorry, I couldn't find an answer to your question. Please try rephrasing it or contacting support."

# Pydantic model for request body
class QueryModel(BaseModel):
    query: str

# Initialize knowledge base and vectorizer
pdf_path = "C:\\Users\\dellg\\OneDrive\\Desktop\\chatbot\\merged.pdf"  # Replace with your PDF file path
knowledge_base = create_knowledge_base(pdf_path)
vectorizer, tfidf_matrix = vectorize_knowledge_base(knowledge_base)

@app.post("/chat")
async def chat(query_model: QueryModel):
    query = query_model.query
    response = chatbot(query, vectorizer, tfidf_matrix, knowledge_base)
    return {"response": response}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1", port=8000)
