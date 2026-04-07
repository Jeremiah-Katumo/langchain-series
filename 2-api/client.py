import requests
import streamlit as st


BASE_URL = "http://127.0.0.1:8000/api"


# API CALLS
def get_openai_response(question: str):
    url = f"{BASE_URL}/essay"
    payload = {"topic": question}

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json().get("essay", "No answer found.")
        else:
            return f"Error {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"

def get_ollama_response(poem: str):
    url = f"{BASE_URL}/poem"
    payload = {"poem": poem}

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json().get("summary", "No answer found.")
        else:
            return f"Error {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"


# STREAMLIT UI
st.title("Chatbot with LangChain API and Streamlit")

input_text1 = st.text_input("Write an essay on:")
button1_clicked = st.button("Get Essay")
if button1_clicked:

    if input_text1:
        openai_response = get_openai_response(input_text1)
        st.subheader("Essay Output")
        st.write(openai_response)
    else:
        st.warning("Please enter a topic for the essay.")
        
input_text2 = st.text_input("Summarize a poem:")
button2_clicked = st.button("Summarize Poem")
if button2_clicked:
    if input_text2:
        ollama_response = get_ollama_response(input_text2)
        st.subheader("Poem Summary")
        st.write(ollama_response)
    else:
        st.warning("Please enter a poem to summarize.")