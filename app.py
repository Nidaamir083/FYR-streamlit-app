import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from transformers import pipeline

import streamlit as st
from Bio import Entrez
import arxiv
import wikipediaapi

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Configuration
Entrez.email = "nida.amir@gmail.com"

@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

qa_pipeline = load_model()

@st.cache_data
def fetch_pubmed_articles(query, start_year=2015, end_year=2024, max_results=20):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, mindate=f"{start_year}/01/01",
                              maxdate=f"{end_year}/12/31", retmax=max_results)
        record = Entrez.read(handle)
        ids = record["IdList"]
        if not ids:
            return pd.DataFrame(columns=["abstract", "source"])
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
        abstracts = [a.strip() for a in handle.read().split("\n\n") if len(a.strip()) > 100]
        return pd.DataFrame({"abstract": abstracts, "source": ["PubMed"] * len(abstracts)})
    except Exception as e:
        st.error(f"PubMed query failed: {str(e)}")
        return pd.DataFrame(columns=["abstract", "source"])

def get_wikipedia_background(topic):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page = wiki_wiki.page(topic)
    if page.exists():
        return [{
            "source": "Wikipedia",
            "title": topic,
            "date": topic,
            "summary": page.summary[:500]
        }]
    return []

@st.cache_data
def fetch_arxiv_articles(query, max_results=5):
    try:
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
    except Exception as e:
        st.error(f"arXiv query failed: {str(e)}")
        return []

def build_merged_report(topic, pubmed_limit=5, arxiv_limit=5):
    pubmed = fetch_pubmed_articles(topic, max_results=pubmed_limit)
    arxiv_articles = fetch_arxiv_articles(topic, max_results=arxiv_limit)
    wiki = get_wikipedia_background(topic)
    return pubmed.to_dict('records') + arxiv_articles + wiki

def visualize_results(data):
    if not data:
        st.warning("No data to visualize")
        return
    
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

def answer_with_llm(question, abstracts):
    try:
        chat = ChatOpenAI(model="gpt-3.5-turbo")
        context = "\n\n".join(doc.get('summary', '') or doc.get('abstract', '') for doc in abstracts[:3])
        response = chat([HumanMessage(content=f"Context: {context}\n\nQuestion: {question}")])
        return response.content
    except Exception as e:
        st.error(f"Failed to generate answer: {str(e)}")
        return "Could not generate answer due to an error."

# Streamlit UI setup
st.set_page_config(page_title="Find Your Research", layout="wide")

background_url = "https://www.yomu.ai/_next/image?url=https%3A%2F%2Fmars-images.imgix.net%2Fseobot%2Fyomu.ai%2F66fddfacb73bfea48e23e839-f6ce70040dea2c7b011ccfe0680258d1.png%3Fauto%3Dcompress&w=1920&q=75"
st.markdown(
    f'''
    <style>
        .stApp {{
            background-image: url("{background_url}");
            background-size: cover;
            background-attachment: fixed;
        }}
    </style>
    ''',
    unsafe_allow_html=True
)

st.markdown("<h1 style='color: #4CAF50;'>🔬 Find Your Research</h1>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🧪 Select a Research Topic")
predefined_topics = ["AI in healthcare", "Thyroid cancer treatment", "CRISPR gene editing", "Drug repurposing"]
selected_topic = st.sidebar.selectbox("Choose a topic or type custom:", predefined_topics + ["Custom"])

custom_query = ""
if selected_topic == "Custom":
    custom_query = st.sidebar.text_input("Enter your custom topic:")
else:
    custom_query = selected_topic

question = st.text_input("Ask a research question:")

if question and custom_query:
    with st.spinner("🔎 Searching Sources and generating answer..."):
        try:
            abstracts = build_merged_report(custom_query)
            
            st.subheader("📄 Retrieved Abstracts")
            with st.expander("Click to show abstracts"):
                for doc in abstracts:
                    content = doc.get('abstract', doc.get('summary', ''))
                    st.markdown(f"**{doc.get('source')}**")
                    st.markdown(f"```\n{content[:500]}...\n```")
            
            st.subheader("📊 Source Distribution")
            visualize_results(abstracts)
            
            st.subheader("🤖 Generated Answer")
            answer = answer_with_llm(question, abstracts)
            st.markdown(answer)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    


