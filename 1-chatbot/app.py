from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_HANDLER"] = "langchain_core.output_parsers.StrOutputParser"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage("You are a helpful assistant that answers questions about the world."),
        HumanMessage("Question: {question}"),
    ]
)


st.title("Chatbot with LangChain and Streamlit")
input_text = st.text_input("Ask a question about the world:")


# When the user clicks the button, run the chain and display the answer
button_clicked = st.button("Get Answer")

if button_clicked:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": input_text})
    
    if input_text:
        st.write(f"Answer: {answer}")