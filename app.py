#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===== ABSOLUTELY FIRST COMMANDS =====
import streamlit as st

# MUST be the very first Streamlit command
st.set_page_config(
    page_title="Find Your Research",
    layout="centered",
    page_icon="🔬"
)

# ===== NOW SAFE TO IMPORT OTHER LIBS =====
import os
import pandas as pd
import wikipedia
import arxiv
from Bio import Entrez
from transformers import pipeline
import asyncio
import base64

# ===== ENVIRONMENT CONFIG =====
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Entrez
Entrez.email = "nida.amir0083@gmail.com"

# ===== FUNCTION DEFINITIONS =====
def add_bg_from_url(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main {{
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return pipeline(
            "text2text-generation", 
            model="google/flan-t5-base",
            device="cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def fetch_pubmed_articles(query, max_results=5):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        ids = record["IdList"]
        
        handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
        records = Entrez.read(handle)
        
        articles = []
        for paper in records['PubmedArticle']:
            title = str(paper['MedlineCitation']['Article']['ArticleTitle'])
            
            abstract = ""
            if 'Abstract' in paper['MedlineCitation']['Article']:
                abstract_texts = []
                for text in paper['MedlineCitation']['Article']['Abstract']['AbstractText']:
                    if hasattr(text, 'attributes') and text.attributes.get('Label'):
                        abstract_texts.append(f"{text.attributes['Label']}: {text}")
                    else:
                        abstract_texts.append(str(text))
                abstract = ' '.join(abstract_texts)
            
            pub_date = None
            if 'ArticleDate' in paper['MedlineCitation']['Article']:
                date = paper['MedlineCitation']['Article']['ArticleDate'][0]
                pub_date = f"{date['Year']}-{date.get('Month','01')}-{date.get('Day','01')}"
            
            articles.append({
                "source": "PubMed",
                "title": title,
                "abstract": abstract,
                "date": pub_date
            })
        
        return articles[:max_results]
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
            "date": result.published.date()
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
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
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

# ===== MAIN APP =====
def main():
    # Set background
    add_bg_from_url("https://images.unsplash.com/photo-1532094349884-543bc11b234d?ixlib=rb-4.0.3")
    
    # Initialize asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Load model
    qa_pipeline = load_model()
    qa_enabled = qa_pipeline is not None
    
    # App layout
    st.markdown("""
    <style>
    h1, h2, h3, h4, h5, h6 {
        color: #2E7D32 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.title("🔬 Find Your Research")
        
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
                    if 'abstract' in data[0] and data[0]['abstract']:
                        st.write(data[0]['abstract'])
                    elif 'summary' in data[0]:
                        st.write(data[0]['summary'])
                    else:
                        st.warning("No content available for this result")
                
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

if __name__ == "__main__":
    main()
