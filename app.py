#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import os
import pandas as pd
import wikipedia
import arxiv
from Bio import Entrez
from transformers import pipeline
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import base64

# ===== ABSOLUTELY FIRST CONFIG =====
st.set_page_config(
    page_title="Find Your Research",
    layout="centered",
    page_icon="🔬"
)

# ===== ENVIRONMENT CONFIG =====
os.environ["TOKENIZERS_PARALLELISM"] = "false"
Entrez.email = "your-email@example.com"  # Replace with your email

# ===== BACKGROUND IMAGE =====
def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded_string});
            background-size: cover;
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

# Add background (using local image for Streamlit Cloud)
add_bg_from_local("background.jpg")  # Add this file to your repo

# ===== MODEL LOADING =====
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    try:
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device="cpu",
            torch_dtype="auto"
        )
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# ===== API FUNCTIONS WITH RETRY =====
def retry_api_call(func, max_retries=3, *args, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)

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
            abstract = ' '.join(
                str(text) for text in 
                paper['MedlineCitation']['Article']['Abstract']['AbstractText']
            ) if 'Abstract' in paper['MedlineCitation']['Article'] else ""
            
            articles.append({
                "source": "PubMed",
                "title": title,
                "abstract": abstract,
                "date": paper['MedlineCitation']['Article']['ArticleDate'][0]['Year']
            })
        
        return articles
    except Exception as e:
        st.toast(f"PubMed error: {str(e)}", icon="⚠️")
        return []

def get_wikipedia_background(topic):
    try:
        wikipedia.set_lang("en")
        search_results = wikipedia.search(topic)
        if not search_results:
            return []
        
        try:
            summary = wikipedia.summary(search_results[0], sentences=3)
            return [{
                "source": "Wikipedia", 
                "title": search_results[0], 
                "summary": summary
            }]
        except wikipedia.exceptions.DisambiguationError as e:
            return [{
                "source": "Wikipedia",
                "title": e.options[0],
                "summary": f"Disambiguation: {e.options[0]}"
            }]
    except Exception:
        return []

def fetch_arxiv_articles(query, max_results=5):
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        return [{
            "source": "arXiv",
            "title": result.title,
            "abstract": result.summary,
            "date": result.published.date()
        } for result in search.results()]
    except Exception as e:
        st.toast(f"arXiv error: {str(e)}", icon="⚠️")
        return []

# ===== ASYNC DATA FETCHING =====
async def build_merged_report(topic):
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor() as executor:
        pubmed_future = loop.run_in_executor(
            executor, 
            lambda: retry_api_call(fetch_pubmed_articles, topic)
        )
        arxiv_future = loop.run_in_executor(
            executor, 
            lambda: retry_api_call(fetch_arxiv_articles, topic)
        )
        wiki_future = loop.run_in_executor(
            executor, 
            lambda: retry_api_call(get_wikipedia_background, topic)
        )
        
        pubmed, arxiv_articles, wiki = await asyncio.gather(
            pubmed_future, arxiv_future, wiki_future
        )
    
    return pubmed + arxiv_articles + wiki

# ===== MAIN APP =====
def main():
    st.title("🔬 Find Your Research")
    
    # Initialize Q&A model
    qa_model = load_model()
    qa_enabled = qa_model is not None
    
    # Search interface
    topic = st.text_input(
        "Enter a research topic:", 
        value="drug repurposing",
        help="Try medical or scientific topics"
    )
    
    if st.button("Search", type="primary"):
        with st.spinner("Searching across PubMed, arXiv, and Wikipedia..."):
            try:
                data = asyncio.run(build_merged_report(topic))
                
                if not data:
                    st.warning("No results found. Try a different search term.")
                    return
                
                # Display results
                st.success(f"Found {len(data)} results")
                display_results(data)
                
                # Q&A Section
                if qa_enabled:
                    st.divider()
                    st.subheader("Ask About This Research")
                    
                    question = st.text_input(
                        "Your question:", 
                        value="What are the key findings?",
                        key="question_input"
                    )
                    
                    if st.button("Get Answer", type="secondary"):
                        context = data[0].get('abstract') or data[0].get('summary', '')
                        if context:
                            with st.spinner("Generating answer..."):
                                try:
                                    answer = qa_model(
                                        f"question: {question} context: {context}",
                                        max_new_tokens=150
                                    )[0]["generated_text"]
                                    st.info(f"**Answer:** {answer}")
                                except Exception as e:
                                    st.error(f"Error generating answer: {str(e)}")
                        else:
                            st.warning("No content available to answer questions")
                
            except Exception as e:
                st.error(f"Search failed: {str(e)}")

def display_results(data):
    df = pd.DataFrame(data)
    
    # Clean data
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    
    # Display compact view
    st.dataframe(
        df[['source', 'title', 'date']],
        column_config={
            "source": "Source",
            "title": "Title",
            "date": "Date"
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Detailed view
    with st.expander("View detailed content", expanded=True):
        tab1, tab2 = st.tabs(["First Result", "All Content"])
        with tab1:
            display_article_content(data[0])
        with tab2:
            for article in data:
                st.markdown(f"**{article['title']}** ({article['source']})")
                display_article_content(article)
                st.divider()

def display_article_content(article):
    content = article.get('abstract') or article.get('summary', 'No content available')
    st.write(content)
    if 'date' in article:
        st.caption(f"Published: {article['date']}")

if __name__ == "__main__":
    main()


    
        
  
       


