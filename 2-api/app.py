from fastapi import FastAPI, status
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langserve import add_routes
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "")

app = FastAPI(
    root_path="/api",
    title="LangChain API",
    description="API using LangChain with OpenAI and Ollama",
)

# Models
class QuestionRequest(BaseModel):
    question: str

class EssayRequest(BaseModel):
    topic: str

class PoemRequest(BaseModel):
    poem: str


@app.get("/", status_code=status.HTTP_200_OK)
def read_root():
    return {"message": "Welcome! Use /ask, /essay, or /poem endpoints."}


# LLMs
openai_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
ollama_model = OllamaLLM(model="llama2", temperature=0)


# Prompts
ask_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Question: {question}"),
    ]
)

essay_prompt = ChatPromptTemplate.from_template(
    "Write an essay about {topic} with 100 words."
)

poem_prompt = ChatPromptTemplate.from_template(
    "Summarize the following poem in less than 50 words: {poem}"
)

parser = StrOutputParser()


# Chains
ask_chain = ask_prompt | openai_model | parser
essay_chain = essay_prompt | ollama_model | parser
poem_chain = poem_prompt | ollama_model | parser


# API Routes (Manual)
@app.post("/ask", status_code=status.HTTP_200_OK)
def ask_question(request: QuestionRequest):
    answer = ask_chain.invoke({"question": request.question})
    return {"answer": answer}


@app.post("/essay", status_code=status.HTTP_200_OK)
def write_essay(request: EssayRequest):
    essay = essay_chain.invoke({"topic": request.topic})
    return {"essay": essay}


@app.post("/poem", status_code=status.HTTP_200_OK)
def summarize_poem(request: PoemRequest):
    summary = poem_chain.invoke({"poem": request.poem})
    return {"summary": summary}


# LangServe Routes (Optional)
add_routes(app, ask_chain, path="/ask-chain")
add_routes(app, essay_chain, path="/essay-chain")
add_routes(app, poem_chain, path="/poem-chain")


# Run App
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)