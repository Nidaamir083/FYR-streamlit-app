
import streamlit as st
from Bio import Entrez
import arxiv
import wikipedia
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage


# 🔧 Config
st.set_page_config(page_title="GenAI Scientific QA", layout="wide")

background_url = "https://tradebrains-wp.s3.ap-south-1.amazonaws.com/features/wp-content/uploads/2024/08/How-to-Avoid-Grammatical-Errors-in-Your-Research-Papers-1080x628.jpg"


# 🎨 Custom background
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

# 🧠 Title and header
st.markdown("<h1 style='color: #4CAF50;'>🔬 GenAI Scientific QA App</h1>", unsafe_allow_html=True)
st.sidebar.markdown("## 🧪 Select a Research Topic")

# 🔽 Dropdown to select topic
topics = ["Anaplastic Thyroid Cancer", "Drug Repurposing", "Thyroid Cancer AI", "PubMed NLP", "Other"]
selected_topic = st.sidebar.selectbox("Choose a scientific topic:", topics)

# 📩 Ask a question
question = st.text_input("Ask a research question:")

# 🧬 PubMed fetcher
def fetch_pubmed_abstracts(query, start_year=2020, end_year=2024, max_results=5):
    Entrez.email = "your-email@example.com"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results,
                            mindate=f"{start_year}/01/01", maxdate=f"{end_year}/12/31", datetype="pdat")
    record = Entrez.read(handle)
    ids = record["IdList"]
    abstracts = []
    for pmid in ids:
        fetch = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        abstracts.append(fetch.read())
    return abstracts

# 🤖 LLM-based answer (simplified)
def answer_with_llm(question, abstracts):
    chat = ChatOpenAI(model="gpt-3.5-turbo")
    context = "\n\n".join(abstracts[:3])  # use top 3 abstracts
    response = chat([HumanMessage(content=f"Context: {context}\n\nQuestion: {question}")])
    return response.content

# ▶️ Run pipeline
if question:
    with st.spinner("🔎 Searching PubMed and generating answer..."):
        abstracts = fetch_pubmed_abstracts(selected_topic)
        answer = answer_with_llm(question, abstracts)
        st.success("✅ Answer:")
        st.write(answer)
        with st.expander("📄 Show retrieved abstracts"):
            for abs in abstracts:
                st.markdown(f"<pre>{abs}</pre>", unsafe_allow_html=True)
