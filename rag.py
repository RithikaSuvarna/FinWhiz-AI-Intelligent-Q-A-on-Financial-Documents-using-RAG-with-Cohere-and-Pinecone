import os
import streamlit as st
import tempfile
import cohere
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
import time

# --- Set page config ---
st.set_page_config(page_title="FinWhiz AI", layout="centered")

# --- CSS styling for dark theme and mobile look ---
st.markdown("""
    <style>
        body {
            background-color: #0d1117;
            color: #c9d1d9;
        }
        .stApp {
            background-color: #0d1117;
        }
        .block-container {
            padding: 2rem 1rem;
        }
        .css-1aumxhk {
            background-color: #161b22;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        .stTextInput > div > div > input {
            background-color: #21262d;
            color: white;
        }
        .stButton > button {
            background-color: #238636;
            color: white;
            border-radius: 10px;
            padding: 0.5rem 1rem;
        }
        .message {
            background-color: #161b22;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("""
    <h2 style='text-align: center; color: #58a6ff;'>🤖 FinWhiz AI</h2>
    <p style='text-align: center; color: #8b949e;'>Ask your financial documents anything!</p>
""", unsafe_allow_html=True)

# --- Init environment variables ---
PINECONE_API_KEY = "pcsk_2gJsfh_D2y2B1ajCXntfoQJD1BF6ce1a82mi66hsgy5q8iuV9ngd9UL1461gRofH7y4FpA"
COHERE_API_KEY = "5v1OSlWJQGrp2XnhWOdFkCus29Q6iggjsOovoeKW"
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")

# --- Initialize Pinecone and Cohere ---
pc = Pinecone(api_key=PINECONE_API_KEY)
co = cohere.Client(COHERE_API_KEY)

# --- Helper: Extract text from PDF ---
def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# --- Helper: Get embeddings ---
def get_embedding(text):
    return co.embed(texts=[text], input_type="search_query", model="embed-english-v3.0").embeddings[0]

# --- Helper: Upload document and create index ---
def create_index_with_document(uploaded_file, index_name="finwhiz-index"):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
        )
    index = pc.Index(index_name)

    if uploaded_file.name.endswith(".pdf"):
        text = extract_text(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeds = co.embed(texts=chunks, input_type="search_document", model="embed-english-v3.0").embeddings

    vectors = [(f"chunk-{i}", embeds[i], {"text": chunks[i]}) for i in range(len(chunks))]
    index.upsert(vectors)
    return index

# --- Helper: Query index ---
def retrieve(query, index):
    query_embed = get_embedding(query)
    results = index.query(vector=query_embed, top_k=5, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]

# --- Helper: Generate answer ---
def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""
    Answer the question based on the context below:
    Context: {context}
    Question: {query}
    Answer:
    """
    res = co.chat(model="command-r-plus", message=prompt)
    return res.text

# --- UI Components ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload financial document (PDF or TXT)", type=["pdf", "txt"])

query = st.text_input("Ask a question")

if uploaded_file:
    with st.spinner("Indexing document..."):
        index = create_index_with_document(uploaded_file)

    if query:
        with st.spinner("Generating answer..."):
            context = retrieve(query, index)
            answer = generate_answer(query, context)

            st.session_state.chat_history.append((query, answer))
            #st.balloons()

# --- Display Chat History ---
if st.session_state.chat_history:
    st.markdown("<h4 style='color:#58a6ff;'>Chat History</h4>", unsafe_allow_html=True)
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Q: {q}"):
            st.markdown(f"<div class='message'>{a}</div>", unsafe_allow_html=True)
            st.download_button(
                label="Download Answer",
                data=a,
                file_name=f"finwhiz_response_{i+1}.txt",
                mime="text/plain"
            )