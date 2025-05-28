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

# Set page config FIRST (before any other Streamlit commands)
st.set_page_config(
    page_title="Find Your Research", 
    layout="wide",
    page_icon="🔬"
)

# Add custom CSS for background and styling
def add_bg_and_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://framerusercontent.com/images/Kif2pDqp7QTQvgJEDrQFEV2FBAY.jpg?scale-down-to=1024");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        .main .block-container {{
            background-color: rgba(255, 255, 255, 0.93);
            border-radius: 15px;
            padding: 2rem;
            margin-top: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .stButton>button {{
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
        }}
        .stTextInput>div>div>input {{
            border-radius: 5px;
            padding: 0.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_and_css()

# Initialize Entrez
Entrez.email = "nida.amir0083@gmail.com"

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

qa_pipeline = load_model()

def fetch_pubmed_articles(query, start_year=2015, end_year=2024, max_results=20):
    handle = Entrez.esearch(db="pubmed", term=query, mindate=f"{start_year}/01/01",
                            maxdate=f"{end_year}/12/31", retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
    abstracts = [a.strip() for a in handle.read().split("\n\n") if len(a.strip()) > 100]
    return pd.DataFrame({"abstract": abstracts, "source": ["PubMed"] * len(abstracts)})

def get_wikipedia_background(topic):
    try:
        summary = wikipedia.summary(topic, sentences=5)
        return [{"source": "Wikipedia", "title": topic, "date": topic, "summary": summary}]
    except Exception:
        return []

def fetch_arxiv_articles(query, max_results=5):
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    articles = []
    for result in search.results():
        if 2015 <= result.published.year <= 2024:
            articles.append({
                "source": "arXiv",
                "title": result.title,
                "date": result.published,
                "summary": result.summary
            })
    return articles

def build_merged_report(topic, pubmed_limit=5, arxiv_limit=5):
    pubmed = fetch_pubmed_articles(topic, max_results=pubmed_limit)
    arxiv_articles = fetch_arxiv_articles(topic, max_results=arxiv_limit)
    wiki = get_wikipedia_background(topic)
    return pubmed.to_dict('records') + arxiv_articles + wiki

def visualize_results(data):
    for doc in data:
        doc['source'] = doc.get('source', 'Unknown')
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x='source', order=df['source'].value_counts().index, palette='pastel', ax=ax)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.xticks(rotation=45)
    plt.title('Sources Distribution', fontsize=14)
    plt.xlabel('Source', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    st.pyplot(fig)

def ask_scientific_question(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}"
    return qa_pipeline(prompt, max_new_tokens=300)[0]["generated_text"].strip()

# App content
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🔬 Find Your Research</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: large;'>Discover scientific insights with AI-powered research assistance</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Research Settings")
    topic = st.text_input("Enter a research topic:", "drug repurposing in thyroid cancer")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This tool helps you find and analyze scientific research using AI.")

# Main content
if topic:
    with st.spinner("🔍 Gathering research data..."):
        data = build_merged_report(topic)
    st.success("✅ Data fetched successfully!")

    # Visualize the sources
    st.subheader("📊 Research Sources Overview")
    visualize_results(data)

    # Ask questions section
    st.subheader("🤖 Ask a Scientific Question")
    question = st.text_input("What would you like to ask about this topic?", 
                           "What AI tools are used in the diagnosis of Thyroid cancer?")
    
    if st.button("Get Answer", key="ask_question"):
        with st.spinner("🧠 Analyzing research and formulating answer..."):
            context_texts = " ".join(doc.get('summary', '') or doc.get('abstract', '') for doc in data)[:4000]
            answer = ask_scientific_question(question, context_texts)
            
            st.markdown("### 📝 Answer")
            st.markdown(f"""
            <div style="
                background-color: #f8f9fa;
                border-left: 4px solid #4CAF50;
                padding: 1rem;
                border-radius: 0 5px 5px 0;
                margin: 1rem 0;
            ">
                {answer}
            </div>
            """, unsafe_allow_html=True) 


