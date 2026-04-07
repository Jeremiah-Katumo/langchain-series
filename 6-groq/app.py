import os
import time
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS


load_dotenv()

GROQ_API_KEY = os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model='llama2')
    st.session_state.loader = WebBaseLoader("https://fastapi.tiangolo.com/advanced/events/")
    st.session_state.docs = st.session_state.loader.load()
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
    st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

    st.session_state.vector_store = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
    
st.title('ChatGroq Demo')

llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template(
"""
Answer the following questions based on the context provided.
Think step by step before giving a detailed answer.
I will tip you $1000 is the user finds the answer helpfully.

<context>
{context}
</context>

Question: {input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = st.session_state.vector_store.as_retriever()

retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

prompt = st.text_input("Write your prompt as input here.")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response time: ", time.process_time() - start)
    st.write(response['answer'])
    
    # With a stremlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for idx, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("------------------------------")
            