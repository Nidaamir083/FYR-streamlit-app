import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wikipedia
import arxiv
from Bio import Entrez
from transformers import pipeline

# Set page config FIRST (only once)
st.set_page_config(
    page_title="Find Your Research", 
    layout="wide",
    page_icon="🔬"
)

# Set background image (right after page config)
background_url = "https://astrixinc.com/wp-content/uploads/2025/04/AI-Image-1.jpg"  # or your preferred image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_url}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }}
    h1, h2, h3 {{
        color: #2c3e50;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# App content
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🔬 Find Your Research</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Welcome to FYR!</h3>", unsafe_allow_html=True)

# Rest of your functions (fetch_pubmed_articles, get_wikipedia_background, etc.)
# [Keep all your existing functions here...]

# Sidebar
with st.sidebar:
    st.header("Settings")
    option = st.selectbox("Choose an option", ["Option 1", "Option 2", "Option 3"])

# Main content
topic = st.text_input("Enter a research topic:", "drug repurposing for anaplastic thyroid cancer")

if topic:
    with st.spinner("Fetching data..."):
        data = build_merged_report(topic)
    st.success("Data fetched successfully!")

    st.subheader("📊 Source Distribution")
    visualize_results(data)

    st.subheader("📚 Sources")
    for doc in data:
        st.markdown(f"- **{doc.get('source')}**: {doc.get('title', doc.get('abstract', doc.get('summary', 'N/A')))[:80]}...")

    st.subheader("🧠 Ask a Scientific Question")
    question = st.text_input("What would you like to ask?", "What AI tools are used in the diagnosis of Thyroid cancer?")
    if st.button("Get Answer"):
        context_texts = " ".join(doc.get('summary', '') or doc.get('abstract', '') for doc in data)[:4000]
        answer = ask_scientific_question(question, context_texts)
        st.markdown("### 🤖 Answer")
        st.write(answer)
   
  
    
        
    




