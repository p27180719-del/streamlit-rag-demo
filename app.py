
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings


st.title("Free RAG Demo (No API Key)")


# -----------------
# Documents
# -----------------
def prepare_documents():

    docs = [
        "Azure Cloud Subscription costs 199 dollars per month",
        "Power BI Pro costs 9.99 dollars per user",
        "Microsoft 365 Business costs 12.50 per user",
        "FAISS is used for vector search",
        "RAG means Retrieval Augmented Generation",
    ]

    return docs


# -----------------
# Split
# -----------------
def split_documents(docs):

    splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0
    )

    chunks = splitter.split_text("\n".join(docs))

    return [Document(page_content=c) for c in chunks]


# -----------------
# Vector store
# -----------------
def create_vector_store(chunks):

    embeddings = FakeEmbeddings(size=32)

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


# -----------------
# Ask
# -----------------
def ask_rag(question):

    docs = prepare_documents()

    chunks = split_documents(docs)

    vector_store = create_vector_store(chunks)

    retriever = vector_store.as_retriever()

    results = retriever.invoke(question)

    context = "\n".join([r.page_content for r in results])

    return context


# -----------------
# UI
# -----------------
query = st.text_input("Ask question")

if st.button("Search"):

    answer = ask_rag(query)

    st.write("### Result")
    st.write(answer)
