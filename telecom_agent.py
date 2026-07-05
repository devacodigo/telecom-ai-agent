from dotenv import load_dotenv
import os
from groq import Groq
import json
import wikipedia
import logging
import time

# RAG imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Set up logging
# Why: captures all agent activity with timestamps and severity levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),  # saves to file
        logging.StreamHandler()             # also prints to terminal
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------------------------------
# PART 1: BUILD THE RAG KNOWLEDGE BASE
# -------------------------------------------------------
# Why: We load the telecom policy document into a vector
# database so the agent can search it by meaning, not keywords

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
# k=2 means fetch the 2 most relevant chunks for each query

logger.info(f"Knowledge base ready — {len(chunks)} chunks loaded into FAISS")

# -------------------------------------------------------
# PART 2: DEFINE THE TOOLS
# -------------------------------------------------------
# These are the actions the agent can take
# The agent reads the descriptions to decide which tool to use

def search_policy(query: str) -> str:
    """
    Search the telecom policy document for answers about:
    refunds, data plans, network outages, contracts, roaming, support.
    Use this for ANY question about company policies or plans.
    """
    docs = retriever.invoke(query)
    # Join the relevant chunks into one string
    result = "\n".join([doc.page_content for doc in docs])
    return result

def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for general knowledge about telecom industry,
    technology concepts, or any real world information not in the policy.
    Use this for questions about technology, industry trends, or general facts.
    """
    try:
        result = wikipedia.summary(query, sentences=3)
        return result
    except Exception as e:
        return f"Could not find information: {str(e)}"

# -------------------------------------------------------
# PART 3: TELL THE LLM ABOUT THE TOOLS
# -------------------------------------------------------
# This is the menu we give to the LLM
# The LLM reads descriptions to decide which tool to call

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search the telecom policy document for answers about refunds, data plans, network outages, contracts, roaming, and customer support. Use this for ANY question about company policies or plans.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The policy topic to search for"
                    }
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
                    "query": {
                        "type": "string",
                        "description": "The topic to search Wikipedia for"
                    }
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

# -------------------------------------------------------
# PART 4: THE SYSTEM PROMPT
# -------------------------------------------------------
# Why: This tells the LLM who it is and how to behave
# It's the first message in every conversation

SYSTEM_PROMPT = """You are a helpful customer support agent for Deutsche Telekom.
You help customers with questions about their plans, policies, billing, and network issues.
Always be polite, clear, and concise.
Use the search_policy tool for questions about company policies and plans.
Use the search_wikipedia tool for general technology or industry questions.
If you already know the answer from conversation history, answer directly without using tools."""

# -------------------------------------------------------
# PART 5: THE AGENT LOOP WITH MEMORY
# -------------------------------------------------------
# Why memory: Unlike agent.py, this agent remembers the whole
# conversation. Each question builds on the previous ones.

# This list stores the full conversation history
# It starts with the system prompt and grows with each message
conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]



def run_agent(question: str, max_retries: int = 3):
    # Add user question to conversation history
    conversation_history.append({
        "role": "user",
        "content": question
    })
    
    logger.info(f"Processing question: {question}")
    
    # Agent loop
    while True:
        # Retry logic — try up to max_retries times
        response = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=conversation_history,
                    tools=tools,
                    tool_choice="auto",
                    timeout=30  # wait max 30 seconds for a response
                )
                break  # if successful, exit retry loop

            except Exception as e:
                error_msg = str(e)
                
                if "rate_limit" in error_msg.lower() or "429" in error_msg:
                    # Rate limit — wait and retry
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}")



                    time.sleep(wait_time)
                
                elif "timeout" in error_msg.lower():
                    # Timeout — retry immediately
                    logger.warning(f"Request timed out. Retrying attempt {attempt + 1}")
                
                else:
                    # Unknown error — log it and retry
                    logger.error(f"API error on attempt {attempt + 1}: {error_msg}")
                
                # If this was the last attempt, give up
                if attempt == max_retries - 1:
                    logger.critical("All 3 retry attempts failed. Giving up.")
                    conversation_history.append({
                        "role": "assistant",
                        "content": "I'm sorry, I'm having technical difficulties. Please try again."
                    })
                    return

        message = response.choices[0].message
        
        # LLM wants to call a tool
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
                
                # Tool error handling
                # Each tool is wrapped in try/except independently
                # so one failing tool doesn't crash the whole agent
                try:
                    result = tool_map[tool_name](**tool_args)
                except Exception as e:
                    # Tool failed — tell the LLM what happened
                    # so it can decide what to do next
                    result = f"Tool {tool_name} failed with error: {str(e)}. Please try a different approach."
                    logger.error(f"Tool {tool_name} failed: {e}")
                
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        
        # LLM has final answer
        else:
            conversation_history.append({
                "role": "assistant",
                "content": message.content
            })
            logger.info("Agent produced final answer")
            print(f"\nAgent: {message.content}\n")
            break
# -------------------------------------------------------
# PART 6: MAIN LOOP
# -------------------------------------------------------
print("\n" + "="*50)
print("Deutsche Telekom Customer Support Agent")
print("="*50)
print("Ask me anything about your plan, billing, or network.")
print("Type 'quit' to exit.\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    run_agent(question)