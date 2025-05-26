import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

import streamlit as st
import pandas as pd
import wikipedia
import arxiv
from Bio import Entrez
from transformers import pipeline
import base64
from PIL import Image
import requests
from io import BytesIO

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
        search = arxiv.Search(query=query, max_results=max_results, 
                            sort_by=arxiv.SortCriterion.Relevance)
        return [{
            "source": "arXiv",
            "title": result.title,
            "summary": result.summary,
            "date": result.published
        } for result in search.results()]
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

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
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
qa_pipeline = load_model()

# ========== CUSTOM STYLING ==========
st.markdown("""
<style>
/* Main container */
.main {
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 10px;
    padding: 2rem;
    margin: 1rem auto;
    max-width: 1000px;
}

/* Title styling */
.title {
    color: #2E7D32 !important;
    text-align: center;
    font-size: 2.2rem !important;
    margin-bottom: 1rem;
}

/* Table styling */
[data-testid="stDataFrame"] {
    font-size: 14px !important;
}

/* Table headers */
[data-testid="stDataFrame"] thead th {
    background-color: #2E7D32 !important;
    color: white !important;
    position: sticky;
    top: 0;
}

/* Table cells */
[data-testid="stDataFrame"] tbody td {
    padding: 8px 12px !important;
    white-space: normal !important;
}

/* Smaller input fields */
.stTextInput>div>div>input {
    padding: 8px 12px !important;
    font-size: 14px !important;
}

/* Error messages */
.stAlert {
    font-size: 14px !important;
}

/* Button styling */
.stButton>button {
    background-color: #2E7D32;
    color: white;
    border-radius: 5px;
    padding: 8px 16px;
    border: none;
}

.stButton>button:hover {
    background-color: #1B5E20;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Add background image (choose one method)
# Method 1: Local image (place a file named 'background.jpg' in same directory)
# add_bg_from_local("background.jpg")

# Method 2: Online image
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

add_bg_from_url("https://images.unsplash.com/photo-1532094349884-543bc11b234d?ixlib=rb-4.0.3")

# ========== APP LAYOUT ==========
st.markdown('<h1 style="color: white; text-align: center; font-size: 2.5rem; text-shadow: 2px 2px 4px #000000;">🔬 Find Your Research</h1>', 
            unsafe_allow_html=True)

# Search input
col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input("Enter a research topic:", 
                         value="drug repurposing for anaplastic thyroid cancer",
                         help="Try medical or scientific topics")
with col2:
    st.write("")
    st.write("")
    search_clicked = st.button("Search", type="primary")

# Main content
if search_clicked or topic:
    with st.spinner("Searching PubMed, arXiv, and Wikipedia..."):
        data = build_merged_report(topic)
    
    if data:
        st.success(f"Found {len(data)} results")
        
        # Display results in compact table
        st.subheader("Research Results")
        display_compact_results(data)
        
        # Show detailed view of first result
        st.subheader("Detailed View")
        with st.expander(f"View {data[0]['source']} content", expanded=True):
            if 'abstract' in data[0]:
                st.write(data[0]['abstract'])
            else:
                st.write(data[0]['summary'])
        
        # QA Section
        st.subheader("Ask About This Research")
        question = st.text_input("Your question:", 
                               value="What are the key findings?",
                               key="question_input")
        
        if st.button("Get Answer", type="primary"):
            context = data[0].get('abstract', data[0].get('summary', ''))
            if context:
                answer = qa_pipeline(f"question: {question} context: {context}", 
                                   max_new_tokens=200)[0]["generated_text"]
                st.info(f"**Answer:** {answer}")
            else:
                st.warning("No content available to generate an answer.")

# Footer
st.markdown("---")
st.caption("© 2024 Find Your Research | Data sources: PubMed, arXiv, Wikipedia")
    



