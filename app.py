
'''
Author: Harshita Jangde
Project: AI Customer Support Bot
Description: Flask backend for an AI-powered customer support chatbot.
             - Loads cleaned FAQ and precomputed answer embeddings.
             - Uses Sentence-BERT for semantic search.
             - Falls back to OpenAI GPT-3.5-turbo for unmatched queries.
             - Handles simple greetings and empty inputs.
'''
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import openai
import os
from dotenv import load_dotenv

#Load env variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") #OPENAI generated code

app = Flask(__name__)

# Load cleaned FAQ dataset 
faq_df = pd.read_csv("faq_auto.csv")
answers = faq_df["answer"].tolist()
questions = faq_df["question"].tolist()

# Load precomputed answer embeddings
answer_embeddings = np.load("answer_embeddings.npy") #load embeddings generated in notebook
model = SentenceTransformer('all-MiniLM-L6-v2')

# sentence BERT for semantic search
def get_semantic_answer(user_input, top_k=1):
    user_emb = model.encode([user_input], convert_to_numpy=True)
    # cosine similarity
    sims = np.dot(answer_embeddings, user_emb.T).squeeze()
    idx = sims.argmax()
    if sims[idx] > 0.6:  # threshold for confident match
        return answers[idx]
    return None

# Fallback to OpenAI for unmatched queries
def get_llm_response(user_input):
    #Queries OpenAI for a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":"You are a helpful customer support assistant."},
                  {"role":"user","content":user_input}]
    )
    return response["choices"][0]["message"]["content"]

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form["message"].strip()
        # Handle empty inputs
        if not user_input:
            return jsonify({"response": "Please type a message."})
        # Handle simple greetings
        if user_input.lower() in ["hi", "hello", "hey"]:
            return jsonify({"response": "Hi there! How can I help you?"})
        
        
        # Try FAQ semantic search
        answer = get_semantic_answer(user_input)
        # Fallback to LLM if FAQ match is not confident 
        if not answer:
            answer = get_llm_response(user_input)
        # Final Fallback
        if not answer or answer.strip() == "":
            answer = "Sorry, I don't understand."
        return jsonify({"response": answer})
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
