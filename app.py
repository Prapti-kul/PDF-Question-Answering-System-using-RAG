"""
PDF Question Answering using RAG (Retrieval-Augmented Generation)

Pipeline:
PDF -> text per page -> chunks -> embeddings -> FAISS index
Question -> embedding -> top-k similar chunks -> prompt -> local LLM (Ollama) -> answer

Run:
    ollama pull llama3
    streamlit run app.py
"""

import io

import faiss
import numpy as np
import requests
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

OLLAMA_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "all-MiniLM-L6-v2"

st.set_page_config(page_title="PDF Q&A using RAG", page_icon="📄", layout="wide")


# ---------------------------------------------------------------
# Step 1 : Load the embedding model (once)
# ---------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL)


# ---------------------------------------------------------------
# Step 2 : Extract text page by page (so we can cite page numbers)
# ---------------------------------------------------------------
def extract_pages(file_name, file_bytes):
    pages = []
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"source": file_name, "page": page_no, "text": text})
    except Exception as e:
        st.error(f"Could not read {file_name}: {e}")
    return pages


# ---------------------------------------------------------------
# Step 3 : Split text into overlapping chunks (tries to end on a sentence)
# ---------------------------------------------------------------
def split_into_chunks(pages, chunk_size, overlap):
    chunks = []

    for p in pages:
        text = " ".join(p["text"].split())  # clean extra spaces/newlines
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Prefer to cut at the end of a sentence
            if end < len(text):
                cut = text.rfind(". ", start + chunk_size // 2, end)
                if cut != -1:
                    end = cut + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "source": p["source"],
                    "page": p["page"],
                    "text": chunk_text
                })

            if end >= len(text):
                break

            start = max(end - overlap, start + 1)

    return chunks


# ---------------------------------------------------------------
# Step 4 : Build the FAISS vector index (cached, so it is not rebuilt
#          every time the user types a question)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_index(files_data, chunk_size, overlap):
    model = load_embedding_model()

    pages = []
    for file_name, file_bytes in files_data:
        pages.extend(extract_pages(file_name, file_bytes))

    chunks = split_into_chunks(pages, chunk_size, overlap)
    if not chunks:
        return None, [], 0

    texts = [c["text"] for c in chunks]

    # Normalised embeddings + inner product = cosine similarity
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks, len(pages)


# ---------------------------------------------------------------
# Step 5 : Retrieve the most relevant chunks for a question
# ---------------------------------------------------------------
def search_chunks(question, index, chunks, top_k, min_score):
    model = load_embedding_model()

    q_emb = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    scores, ids = index.search(q_emb, min(top_k, len(chunks)))

    results = []
    for score, i in zip(scores[0], ids[0]):
        if i != -1 and score >= min_score:
            results.append({**chunks[i], "score": float(score)})

    return results


# ---------------------------------------------------------------
# Step 6 : Ask the local LLM through Ollama
# ---------------------------------------------------------------
def ask_llm(question, context, history, model_name):
    history_text = ""
    for turn in history[-6:]:  # last 3 question/answer pairs
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['content']}\n"

    prompt = f"""You are a document question answering assistant.

Rules:
1. Answer ONLY from the context below.
2. Do not use outside knowledge.
3. If the answer is not in the context, say exactly:
   "I could not find the answer in the uploaded PDF."
4. Keep the answer clear and simple. Use bullet points when helpful.

Previous conversation:
{history_text}

Context:
{context}

Question:
{question}

Answer:"""

    payload = {"model": model_name, "prompt": prompt, "stream": False}

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        return "Ollama is not running. Start Ollama and run: `ollama pull " + model_name + "`"
    except requests.exceptions.Timeout:
        return "The model took too long to respond. Try a smaller model or a shorter question."
    except Exception as e:
        return f"Error while talking to the LLM: {e}"


# ---------------------------------------------------------------
# Sidebar : settings
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    model_name = st.text_input("Ollama model", value="llama3")
    chunk_size = st.slider("Chunk size (characters)", 200, 1500, 600, step=50)
    overlap = st.slider("Chunk overlap (characters)", 0, 300, 100, step=10)
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 8, 4)
    min_score = st.slider(
        "Minimum similarity score", 0.0, 0.8, 0.25, step=0.05,
        help="Chunks scoring below this are ignored. If nothing passes, "
             "the app says the answer was not found instead of guessing."
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Embeddings: all-MiniLM-L6-v2 | Vector DB: FAISS | LLM: Ollama")


# ---------------------------------------------------------------
# Main page
# ---------------------------------------------------------------
st.title("📄 Chat with your PDFs (RAG)")
st.write("Upload one or more PDFs and ask questions. Answers come only from your documents.")

uploaded_files = st.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not uploaded_files:
    st.info("Upload at least one PDF to begin.")
    st.stop()

# Convert uploads to hashable data so the index can be cached
files_data = tuple((f.name, f.getvalue()) for f in uploaded_files)

with st.spinner("Reading PDFs and building the vector index..."):
    index, chunks, page_count = build_index(files_data, chunk_size, overlap)

if index is None:
    st.error("No readable text found. Scanned PDFs need OCR, which this app does not do.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Files", len(uploaded_files))
c2.metric("Pages with text", page_count)
c3.metric("Chunks indexed", len(chunks))

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**{s['source']} - page {s['page']}** (score {s['score']:.2f})"
                    )
                    st.caption(s["text"])

# New question
question = st.chat_input("Ask a question about your PDFs")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            results = search_chunks(question, index, chunks, top_k, min_score)

        if not results:
            answer = "I could not find the answer in the uploaded PDF."
        else:
            context = "\n\n".join(
                f"[{r['source']} p.{r['page']}] {r['text']}" for r in results
            )
            with st.spinner("Generating answer..."):
                answer = ask_llm(
                    question, context, st.session_state.messages, model_name
                )

        st.write(answer)

        if results:
            with st.expander("Sources"):
                for s in results:
                    st.markdown(
                        f"**{s['source']} - page {s['page']}** (score {s['score']:.2f})"
                    )
                    st.caption(s["text"])

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": results}
    )
