import os
import time
from dotenv import load_dotenv
import streamlit as st 
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model='llama2') # Creating Embeddings 
    st.session_state.loader = PyPDFDirectoryLoader("../census") # Data Ingestion
    st.session_state.docs = st.session_state.loader.load() # Loading the documents into memory
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Chunk Creation
    st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) # Splitting the documents into chunks

    st.session_state.vector_store = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings) # Creating the vector database and storing the document chunks in it
    
    
st.title('ChatGroq Demo with Llama3')

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

prompt = ChatMessagePromptTemplate.from_template(
"""
Answer the following questions based on the context provided.
Think step by step before giving a detailed answer.
I will tip you $1000 is the user finds the answer helpfully.

<context>
{context}
</context>

Questions: {input}
"""
)

def vector_embedding():
    if "vector" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model='llama2') # Creating Embeddings 
        st.session_state.loader = PyPDFDirectoryLoader("../census") # Data Ingestion
        st.session_state.docs = st.session_state.loader.load() # Loading the documents into memory
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Chunk Creation
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) # Splitting the documents into chunks

        st.session_state.vector_store = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings) # Creating the vector database and storing the document chunks in it 

prompt1 = st.text_input("Enter your Question from Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Documents have been embedded successfully! DB is ready!")
    
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)
    
if prompt1:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})
    print("Response time: ", time.process_time() - start)
    st.write(response['answer'])
    
    with st.expander("Retrieved Documents"):
        for idx, doc in enumerate(retriever.get_relevant_documents(prompt1)):
            st.write(f"Document {idx+1}:")
            st.write(doc.page_content)
            st.write("------------------------------")
            
        for idx, doc in enumerate(response['context']):
            st.write(f"Context Document {idx+1}:")
            st.write(doc.page_content)
            st.write("------------------------------")