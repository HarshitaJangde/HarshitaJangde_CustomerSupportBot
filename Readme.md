# AI Customer Support Bot

An AI-powered chatbot that answers FAQs using embeddings-based semantic search.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Setup Instructions](#setup-instructions)  
5. [Usage](#usage)  
6. [Notes](#notes)  
7. [Acknowledgements](#acknowledgements)  

---

## Overview

This project implements a local AI customer support bot:

- Uses `twcs.csv` to generate processed FAQs (`faq_auto.csv`).  
- Generates embeddings (`answer_embeddings.npy`) for semantic search.  
- Returns relevant answers for user queries using embeddings.  

---

## Features

- FAQ search using AI embeddings  
- Local and simple to run  
- Lightweight web interface using Flask  

---

## Project Structure

ai_bot/
├── app.py
├── requirements.txt
├── faq.csv
├── faq_auto.csv
├── twcs.csv
├── answer_embeddings.npy
├── templates/
│ └── index.html
├── static/
  └── style.css

- **app.py** – main Flask app to run the bot  
- **faq.csv** – original FAQ file  
- **faq_auto.csv** – processed FAQ generated from `twcs.csv`  
- **twcs.csv** – dataset from Kaggle (user must download)  
- **answer_embeddings.npy** – generated embeddings for FAQs  
- **templates/** – HTML templates for Flask  
- **static/** – CSS and other static files  

---

## Setup Instructions

1. **Download the dataset**  
   Download `twcs.csv` from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter/data) and place it in the `ai_bot/` folder.  

2. **Generate processed FAQs and embeddings**  
   - Open your notebook or Python script for preprocessing (if provided).  
   - Run it to generate:  
     - `faq_auto.csv`  
     - `answer_embeddings.npy`  

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt

---

## Usage

1. Run the Flask app:

```bash
python app.py
```
2. **Open the web interface**  

```text
Open your browser and go to: http://127.0.0.1:5000
```

---

3. **Ask a question**

- Type your query in the input box.
- The bot will return the most relevant answer from the FAQ dataset.

---

## Notes

- Make sure faq_auto.csv and answer_embeddings.npy exist in the folder before running app.py.
- OpenAI API key is required if embeddings need to be regenerated.
- Deployment is not included; this project runs locally.

## Acknowledgements

- Kaggle dataset contributors
- OpenAI for embeddings API