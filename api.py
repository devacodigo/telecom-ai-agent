from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import wikipedia
import logging
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# -------------------------------------------------------
# Logging setup
# -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# FastAPI app setup
# -------------------------------------------------------
app = FastAPI()

# Why CORS: React runs on port 3000, FastAPI on port 8000
# Browsers block requests between different ports by default
# CORS tells the browser "it's okay, allow these requests"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# Build RAG knowledge base on startup
# -------------------------------------------------------
logger.info("Loading telecom policy document...")
loader = TextLoader("telecom_policy.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
logger.info(f"Knowledge base ready — {len(chunks)} chunks loaded")

# -------------------------------------------------------
# Groq client and tools
# -------------------------------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def search_policy(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

def search_wikipedia(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Could not find information: {str(e)}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search the telecom policy document for answers about refunds, data plans, network outages, contracts, roaming, and customer support.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The policy topic to search for"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for general knowledge about telecom technology, industry trends, or any real world information not covered in company policy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The topic to search for"}
                },
                "required": ["query"]
            }
        }
    }
]

tool_map = {
    "search_policy": search_policy,
    "search_wikipedia": search_wikipedia
}

SYSTEM_PROMPT = """You are a helpful customer support agent for Deutsche Telekom.
You help customers with questions about their plans, policies, billing, and network issues.
Always be polite, clear, and concise.
Use the search_policy tool for questions about company policies and plans.
Use the search_wikipedia tool for general technology or industry questions.
If you already know the answer from conversation history, answer directly without using tools."""

# -------------------------------------------------------
# Conversation memory — stored per session
# -------------------------------------------------------
conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# -------------------------------------------------------
# Request model
# Why Pydantic BaseModel: FastAPI uses this to validate
# incoming request data automatically
# -------------------------------------------------------
class ChatRequest(BaseModel):
    message: str

# -------------------------------------------------------
# The chat endpoint
# -------------------------------------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.message
    logger.info(f"Received message: {question}")

    conversation_history.append({
        "role": "user",
        "content": question
    })

    # Agent loop
    while True:
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=conversation_history,
                tools=tools,
                tool_choice="auto"
            )
        except Exception as e:
            logger.error(f"API error: {e}")
            return {"response": "I'm having technical difficulties. Please try again."}

        message = response.choices[0].message

        if message.tool_calls:
            conversation_history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                logger.info(f"Calling tool: {tool_name} with args: {tool_args}")

                try:
                    result = tool_map[tool_name](**tool_args)
                except Exception as e:
                    result = f"Tool {tool_name} failed: {str(e)}"
                    logger.error(f"Tool error: {e}")

                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            final_answer = message.content
            conversation_history.append({
                "role": "assistant",
                "content": final_answer
            })
            logger.info("Agent produced final answer")
            return {"response": final_answer}

# -------------------------------------------------------
# Health check endpoint
# Why: lets you verify the server is running
# -------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}