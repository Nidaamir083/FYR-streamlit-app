import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

import streamlit as st
import pandas as pd
import wikipedia
import arxiv
from Bio import Entrez
from transformers import pipeline
import base64
import asyncio

# Fix for event loop error
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Initialize Entrez
Entrez.email = "nida.amir0083@gmail.com"

# ========== FUNCTION DEFINITIONS ==========
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

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
            st.error(f"Wikipedia disambiguation error for: {topic}")
            return []
    except Exception as e:
        st.error(f"Wikipedia error: {str(e)}")
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
         </style>
         """,
         unsafe_allow_html=True
     )

# ========== APP CONFIGURATION ==========
st.set_page_config(
    page_title="Find Your Research", 
    layout="centered",
    page_icon="🔬"
)

# Load model
try:
    qa_pipeline = load_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Add background
add_bg_from_url("https://images.unsplash.com/photo-1532094349884-543bc11b234d?ixlib=rb-4.0.3")

# ========== APP LAYOUT ==========
st.markdown("""
<style>
/* Main content container */
.st-emotion-cache-1y4p8pa {
    background-color: rgba(255, 255, 255, 0.9) !important;
    border-radius: 10px;
    padding: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="color: white; text-align: center; font-size: 2.5rem; text-shadow: 2px 2px 4px #000000;">🔬 Find Your Research</h1>', 
            unsafe_allow_html=True)

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
        
        st.subheader("Ask About This Research")
        question = st.text_input("Your question:", 
                               value="What are the key findings?",
                               key="question_input")
        
        if st.button("Get Answer", type="primary"):
            context = data[0].get('abstract', data[0].get('summary', ''))
            if context:
                try:
                    answer = qa_pipeline(f"question: {question} context: {context}", 
                                       max_new_tokens=200)[0]["generated_text"]
                    st.info(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
            else:
                st.warning("No content available to generate an answer.")

st.markdown("---")
st.caption("© 2024 Find Your Research | Data sources: PubMed, arXiv, Wikipedia")
    
 







   
    



