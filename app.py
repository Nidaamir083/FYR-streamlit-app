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
            "summary": result.summary
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
    
    # Create DataFrame with limited columns
    df = pd.DataFrame(data)
    if 'date' not in df.columns:
        df['date'] = pd.NaT
    
    # Select and rename columns for display
    display_df = df[['source', 'title', 'date']].rename(columns={
        'source': 'Source',
        'title': 'Title',
        'date': 'Date'
    })
    
    # Display compact table
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
</style>
""", unsafe_allow_html=True)

# ========== APP LAYOUT ==========
# Replace the existing title markup with this:
st.markdown('<h1 style="color: black; text-align: center; font-size: 2.5rem;">🔬 Find Your Research</h1>', 
            unsafe_allow_html=True)

# Search input
topic = st.text_input("Enter a research topic:", 
                     value="drug repurposing for anaplastic thyroid cancer",
                     help="Try medical or scientific topics")

# Main content
if st.button("Search") or topic:
    with st.spinner("Searching PubMed, arXiv, and Wikipedia..."):
        data = build_merged_report(topic)
    
    if data:
        st.success(f"Found {len(data)} results")
        
        # Display results in compact table
        st.subheader("Research Results")
        display_compact_results(data)
        
        # Show detailed view of first result
        st.subheader("Detailed View")
        with st.expander(f"View {data[0]['source']} content"):
            if 'abstract' in data[0]:
                st.write(data[0]['abstract'])
            else:
                st.write(data[0]['summary'])
        
        # QA Section
        st.subheader("Ask About This Research")
        question = st.text_input("Your question:", 
                               value="What are the key findings?",
                               key="question_input")
        
        if st.button("Get Answer"):
            context = data[0].get('abstract', data[0].get('summary', ''))
            answer = qa_pipeline(f"question: {question} context: {context}", 
                               max_new_tokens=200)[0]["generated_text"]
            st.info(f"**Answer:** {answer}")
    else:
        st.error("No results found. Please try a different search term.")

# Footer
st.markdown("---")
st.caption("© 2024 Find Your Research | Data sources: PubMed, arXiv, Wikipedia")


