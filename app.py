import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"  # Suppress warnings

import streamlit as st
import pandas as pd
import wikipedia
import arxiv
from Bio import Entrez
from transformers import pipeline
import asyncio

# Disable Streamlit file watcher completely
st.runtime.legacy_caching.cache_data.clear()
st.runtime.legacy_caching.cache_resource.clear()

# Initialize Entrez
Entrez.email = "nida.amir0083@gmail.com"

# ========== FUNCTION DEFINITIONS ==========
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        # Explicitly set device to CPU to avoid any GPU-related issues
        return pipeline("text2text-generation", 
                      model="google/flan-t5-base",
                      device="cpu")
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def fetch_pubmed_articles(query, max_results=5):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        ids = record["IdList"]
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
        abstracts = [a.strip() for a in handle.read().split("\n\n") if len(a.strip()) > 100]
        return [{"source": "PubMed", "title": f"PubMed {i+1}", "abstract": abstract} 
               for i, abstract in enumerate(abstracts)]
    except Exception as e:
        st.error(f"PubMed error: {str(e)}")
        return []

def get_wikipedia_background(topic):
    try:
        summary = wikipedia.summary(topic, sentences=3)
        return [{"source": "Wikipedia", "title": topic, "summary": summary}]
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            summary = wikipedia.summary(e.options[0], sentences=3)
            return [{"source": "Wikipedia", "title": e.options[0], "summary": summary}]
        except:
            return []
    except Exception as e:
        return []

def fetch_arxiv_articles(query, max_results=5):
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results)
        results = list(client.results(search))
        return [{
            "source": "arXiv",
            "title": result.title,
            "summary": result.summary,
            "date": result.published
        } for result in results]
    except Exception as e:
        st.error(f"arXiv error: {str(e)}")
        return []

def build_merged_report(topic):
    pubmed = fetch_pubmed_articles(topic)
    arxiv_articles = fetch_arxiv_articles(topic)
    wiki = get_wikipedia_background(topic)
    return pubmed + arxiv_articles + wiki

def display_compact_results(data):
    if not data:
        st.warning("No results found. Try a different search term.")
        return
    
    df = pd.DataFrame(data)
    if 'date' not in df.columns:
        df['date'] = pd.NaT
    
    display_df = df[['source', 'title', 'date']].rename(columns={
        'source': 'Source',
        'title': 'Title',
        'date': 'Date'
    })
    
    st.dataframe(
        display_df,
        height=min(400, 45 * len(display_df)),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Source": st.column_config.TextColumn(width="small"),
            "Title": st.column_config.TextColumn(width="medium"),
            "Date": st.column_config.DateColumn(width="small")
        }
    )

# ========== APP CONFIGURATION ==========
st.set_page_config(
    page_title="Find Your Research", 
    layout="centered",
    page_icon="🔬"
)

# Load model with error handling
qa_pipeline = load_model()
if qa_pipeline is None:
    st.error("Failed to initialize the AI model. The Q&A feature will be disabled.")
    qa_enabled = False
else:
    qa_enabled = True

# ========== APP LAYOUT ==========
st.title("🔬 Find Your Research")

# Search input
topic = st.text_input("Enter a research topic:", 
                     value="drug repurposing for anaplastic thyroid cancer",
                     help="Try medical or scientific topics")

if st.button("Search", type="primary") or topic:
    with st.spinner("Searching PubMed, arXiv, and Wikipedia..."):
        data = build_merged_report(topic)
    
    if data:
        st.success(f"Found {len(data)} results")
        st.subheader("Research Results")
        display_compact_results(data)
        
        st.subheader("Detailed View")
        with st.expander(f"View {data[0]['source']} content", expanded=True):
            if 'abstract' in data[0]:
                st.write(data[0]['abstract'])
            else:
                st.write(data[0]['summary'])
        
        if qa_enabled:
            st.subheader("Ask About This Research")
            question = st.text_input("Your question:", 
                                   value="What are the key findings?",
                                   key="question_input")
            
            if st.button("Get Answer", type="primary"):
                context = data[0].get('abstract', data[0].get('summary', ''))
                if context:
                    try:
                        answer = qa_pipeline(
                            f"question: {question} context: {context}", 
                            max_new_tokens=200
                        )[0]["generated_text"]
                        st.info(f"**Answer:** {answer}")
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
                else:
                    st.warning("No content available to generate an answer.")

st.markdown("---")
st.caption("© 2024 Find Your Research | Data sources: PubMed, arXiv, Wikipedia")


