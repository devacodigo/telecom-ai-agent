from dotenv import load_dotenv
import os
import json
import time
import wikipedia
import logging
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
# Build RAG knowledge base
# Runs once when this file is imported or executed
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
# Groq client
# -------------------------------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------------------------------
# Tool functions
# -------------------------------------------------------
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
# Conversation memory
# Exposed so api.py can reset it if needed
# -------------------------------------------------------
conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# -------------------------------------------------------
# Core agent function
# This is what api.py and the terminal loop both call
# -------------------------------------------------------
def run_agent(question: str) -> str:
    logger.info(f"Processing question: {question}")

    conversation_history.append({
        "role": "user",
        "content": question
    })

    for attempt in range(3):
        try:
            while True:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=conversation_history,
                    tools=tools,
                    tool_choice="auto"
                )

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
                    return final_answer

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                wait_time = 2 ** attempt
                logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}")
                time.sleep(wait_time)
            elif "timeout" in error_msg.lower():
                logger.warning(f"Timeout. Retrying attempt {attempt + 1}")
            else:
                logger.error(f"API error on attempt {attempt + 1}: {error_msg}")

            if attempt == 2:
                logger.critical("All 3 retry attempts failed.")
                return "I'm sorry, I'm having technical difficulties. Please try again."

# -------------------------------------------------------
# Terminal interface
# Only runs when you execute: python telecom_agent.py
# NOT when api.py imports this file
# -------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Deutsche Telekom Customer Support Agent")
    print("="*50)
    print("Ask me anything. Type 'quit' to exit.\n")

    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        answer = run_agent(question)
        print(f"\nAgent: {answer}\n")