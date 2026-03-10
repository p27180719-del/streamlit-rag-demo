
import streamlit as st
import PyPDF2

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings


# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="AI Assistant", layout="wide")

st.title("AI Assistant")
st.write("Upload PDF and chat with your document")


# -------------------------
# Session memory
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "store" not in st.session_state:
    st.session_state.store = None


# -------------------------
# Read PDF
# -------------------------
def read_pdf(file):

    reader = PyPDF2.PdfReader(file)

    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t

    return text


# -------------------------
# Split text
# -------------------------
def split_text(text):

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    docs = [Document(page_content=c) for c in chunks]

    return docs


# -------------------------
# Create FAISS store
# -------------------------
def create_store(docs):

    embeddings = FakeEmbeddings(size=32)

    store = FAISS.from_documents(
        docs,
        embeddings
    )

    return store


# -------------------------
# Upload PDF
# -------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    text = read_pdf(uploaded_file)

    docs = split_text(text)

    st.session_state.store = create_store(docs)

    st.success("PDF indexed successfully")


# -------------------------
# Chat input
# -------------------------
query = st.chat_input("Ask something from PDF")

if query:

    st.session_state.history.append(("user", query))

    if st.session_state.store is None:

        answer = "Please upload PDF first"

    else:

        retriever = st.session_state.store.as_retriever()

        results = retriever.invoke(query)

        context = "

".join([r.page_content for r in results])

        answer = "Answer from PDF:

" + context


    st.session_state.history.append(("bot", answer))


# -------------------------
# Show chat history
# -------------------------
for role, msg in st.session_state.history:

    if role == "user":
        st.chat_message("user").write(msg)

    else:
        st.chat_message("assistant").write(msg)
