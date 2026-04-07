from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
import streamlit as st 
from dotenv import load_dotenv
import os

load_dotenv()


os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_HANDLER"] = "langchain_core.output_parsers.StrOutputParser"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that answers questions about the world."),
        ("human", "Question: {question}"),
    ]
)


st.title("Chatbot with LangChain and Streamlit")
input_text = st.text_input("Ask a question about the world:")


# When the user clicks the button, run the chain and display the answer
button_clicked = st.button("Get Answer")


if button_clicked:
    llm = OllamaLLM(model="llama2", temperature=0)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": input_text})
    
    if input_text:
        st.write(f"Answer: {answer}")