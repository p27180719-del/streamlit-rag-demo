
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
import PyPDF2


st.title("PDF RAG Search (Free)")


# -----------------
# Read PDF
# -----------------
def read_pdf(file):

    pdf_reader = PyPDF2.PdfReader(file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text


# -----------------
# Split text
# -----------------
def split_text(text):

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    return [Document(page_content=c) for c in chunks]


# -----------------
# Vector store
# -----------------
def create_store(docs):

    embeddings = FakeEmbeddings(size=32)

    store = FAISS.from_documents(
        docs,
        embeddings
    )

    return store


# -----------------
# UI upload
# -----------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

query = st.text_input("Ask question")

if st.button("Search"):

    if uploaded_file is None:
        st.write("Upload PDF first")

    else:

        text = read_pdf(uploaded_file)

        docs = split_text(text)

        store = create_store(docs)

        retriever = store.as_retriever()

        results = retriever.invoke(query)

        answer = "\n".join([r.page_content for r in results])

        st.write("### Result")
        st.write(answer)
