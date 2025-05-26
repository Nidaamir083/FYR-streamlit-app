import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wikipedia
import arxiv
from Bio import Entrez
from transformers import pipeline

# Initialize Entrez
Entrez.email = "nida.amir0083@gmail.com"

# ========== FUNCTION DEFINITIONS ==========
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

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
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='source', order=df['source'].value_counts().index, palette='pastel', ax=ax)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                   textcoords='offset points')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def ask_scientific_question(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}"
    return qa_pipeline(prompt, max_new_tokens=300)[0]["generated_text"].strip()

# ========== APP CONFIGURATION ==========
st.set_page_config(
    page_title="Find Your Research", 
    layout="wide",
    page_icon="🔬"
)

# Load model
qa_pipeline = load_model()

# ========== CUSTOM STYLING ==========
background_url = "https://astrixinc.com/wp-content/uploads/2025/04/AI-Image-1.jpg"
st.markdown(
    f"""
    <style>
    /* Background styling */
    .stApp {{
        background-image: url("{background_url}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    
    /* Main container styling */
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 90%;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    
    /* Title styling */
    .title {{
        color: #2E7D32 !important;
        text-align: center;
        font-size: 2.8rem !important;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }}
    
    /* Header styling */
    h2, h3 {{
        color: #1B5E20 !important;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 0.5rem;
    }}
    
    /* Table styling */
    .stDataFrame {{
        max-height: 400px;
        overflow: auto;
        margin: 1rem 0;
    }}
    
    /* Table header */
    .stDataFrame thead th {{
        background-color: #2E7D32 !important;
        color: white !important;
        position: sticky;
        top: 0;
        font-weight: 600;
    }}
    
    /* Table cells */
    .stDataFrame tbody td {{
        color: #333 !important;
        font-size: 0.95rem !important;
    }}
    
    /* Hover effect */
    .stDataFrame tbody tr:hover {{
        background-color: #E8F5E9 !important;
    }}
    
    /* Input fields */
    .stTextInput input {{
        background-color: rgba(255,255,255,0.9) !important;
    }}
    
    /* Button styling */
    .stButton>button {{
        background-color: #2E7D32 !important;
        color: white !important;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 500;
    }}
    .stButton>button:hover {{
        background-color: #1B5E20 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ========== APP LAYOUT ==========
st.markdown('<h1 class="title">🔬 Find Your Research</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #1B5E20;">Welcome to FYR - Your Research Companion</h3>', unsafe_allow_html=True)

# ========== MAIN APP LOGIC ==========
with st.container():
    topic = st.text_input("Enter a research topic:", "drug repurposing for anaplastic thyroid cancer")

if topic:
    with st.spinner("🔍 Searching across PubMed, arXiv, and Wikipedia..."):
        data = build_merged_report(topic)
    
    st.success("✅ Data fetched successfully!")
    
    # Display results in a compact table
    st.subheader("📊 Source Distribution")
    visualize_results(data)
    
    st.subheader("📚 Research Results")
    df = pd.DataFrame(data)[['source', 'title', 'date']]
    st.dataframe(
        df,
        height=350,
        use_container_width=True,
        hide_index=True,
        column_config={
            "source": st.column_config.TextColumn("Source", width="small"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "date": st.column_config.DatetimeColumn("Date", width="small")
        }
    )
    
    st.subheader("🧠 Ask a Scientific Question")
    question = st.text_input("What would you like to ask about this research?", 
                           "What AI tools are used in the diagnosis of Thyroid cancer?")
    
    if st.button("Get Answer", type="primary"):
        with st.spinner("🤖 Analyzing research and generating answer..."):
            context_texts = " ".join(doc.get('summary', '') or doc.get('abstract', '') for doc in data)[:4000]
            answer = ask_scientific_question(question, context_texts)
            
            st.markdown("### 💡 Answer")
            st.markdown(f'<div style="background-color: #E8F5E9; padding: 1rem; border-radius: 8px; border-left: 4px solid #2E7D32;">{answer}</div>', 
                       unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown('<div style="text-align: center; color: #666; font-size: 0.9rem;">Find Your Research © 2024 | Powered by Streamlit</div>', 
            unsafe_allow_html=True)
        
    




