# AI Hackathon — Complete Documentation
## Days 1–4: From Zero to a Production AI Agent

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Tech Stack](#tech-stack)
3. [Day 1 — Multi-Turn Chatbot](#day-1--multi-turn-chatbot)
4. [Day 2 — RAG Pipeline](#day-2--rag-pipeline)
5. [Day 3 — AI Agent with Tools](#day-3--ai-agent-with-tools)
6. [Day 4 — Telecom Support Agent](#day-4--telecom-support-agent)
7. [How Everything Connects](#how-everything-connects)
8. [Key Concepts Glossary](#key-concepts-glossary)

---

## Project Overview

Over 4 days, you built three progressively more powerful AI systems from scratch, culminating in a production-grade telecom customer support agent. Each day built directly on the previous one.

| Day | File | What it does |
|---|---|---|
| 1 | `chat.py` | Multi-turn chatbot with memory |
| 2 | `rag.py` | Document Q&A using vector search |
| 3 | `agent.py` | AI agent that uses tools to answer |
| 4 | `telecom_agent.py` | Full telecom support agent combining all three |

---

## Tech Stack

| Tool | What it is | Why we use it |
|---|---|---|
| Python 3.13 | Programming language | Industry standard for AI development |
| LangChain | AI framework | Ready-made tools for building LLM apps |
| LangGraph | Agent framework | Modern way to build agentic workflows |
| Groq | LLM API provider | Free, fast, no daily quota issues |
| Gemini API | Google AI | Used for embeddings (converting text to numbers) |
| FAISS | Vector database | Stores and searches document embeddings locally |
| Wikipedia | Knowledge source | Real-time general knowledge tool |
| python-dotenv | Environment manager | Safely stores API keys outside code |

---

## Day 1 — Multi-Turn Chatbot

### What we built
A chatbot that remembers the full conversation history. Unlike a single-turn AI call, this one knows what was said earlier in the conversation.

### File: `chat.py`

```python
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

messages = [
    SystemMessage(content="You are a helpful assistant.")
]

print("Chatbot ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    messages.append(HumanMessage(content=user_input))
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print(f"AI: {response.content}\n")
```

### How it works

```
You type a message
        ↓
Message added to history list
        ↓
Entire history sent to Gemini
        ↓
Gemini reads full context and replies
        ↓
Reply added to history list
        ↓
Next message includes everything above
```

### Line by line explanation

| Code | What it does | Why |
|---|---|---|
| `load_dotenv()` | Reads `.env` file into memory | So `os.getenv()` can find your API key |
| `ChatGoogleGenerativeAI(...)` | Creates the Gemini model object | Selects which AI brain to use |
| `messages = [SystemMessage(...)]` | Creates conversation history with one instruction | Sets AI behaviour for the whole conversation |
| `messages.append(HumanMessage(...))` | Adds your message to history | Builds up conversation context |
| `llm.invoke(messages)` | Sends full history to Gemini | Gemini sees everything said so far |
| `messages.append(AIMessage(...))` | Adds AI reply to history | So AI remembers its own previous answers |

### Key concept: Why memory works
Every time you send a message, you send the **entire conversation history**, not just the latest message. Gemini reads everything and answers in context. This is why it can answer "what was my first question?" — because it received all previous messages.

### Message types
- `SystemMessage` — hidden instructions that shape AI behaviour. User never sees this.
- `HumanMessage` — what you type
- `AIMessage` — what the AI replies

---

## Day 2 — RAG Pipeline

### What we built
A system that reads a PDF document and answers questions from it. The AI doesn't use its general training knowledge — it only answers from your specific document.

### What is RAG?
RAG stands for **Retrieval Augmented Generation**.

- **Retrieval** — find the relevant parts of your document
- **Augmented** — add those parts to the AI's context
- **Generation** — AI generates an answer based on what it found

Think of it like an open-book exam. Instead of answering from memory, the AI looks up the relevant pages first.

### File: `rag.py`

```python
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

# Step 1 — Load PDF
loader = PyPDFLoader("Devashu_resume_July.pdf")
documents = loader.load()

# Step 2 — Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Step 3 — Store in vector database
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Step 4 — Build RAG chain
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
Context: {context}
Question: {question}
""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    answer = rag_chain.invoke(question)
    print(f"AI: {answer}\n")
```

### How it works — the 4 steps

**Step 1 — Load PDF**
Reads your PDF file page by page and converts it into text Python can work with.

**Step 2 — Split into chunks**
The AI can't read an entire document at once — it has a token limit. So we cut the document into overlapping pieces of 500 characters.

Why overlap? If an important sentence falls at the edge of a chunk, the 50-character overlap ensures it appears in both chunks so its meaning isn't lost.

```
Document: "...the refund policy states customers have 30 days..."
Chunk 1:  "...the refund policy states customers"
Chunk 2:  "customers have 30 days..."  ← overlap preserves context
```

**Step 3 — Store in vector database**
Each chunk gets converted into a list of numbers called an **embedding**. These numbers represent the *meaning* of the text, not just the words.

FAISS stores all these numbers. When you ask a question, your question also becomes numbers, and FAISS finds the chunks whose numbers are mathematically closest — meaning most semantically similar to your question.

This is why searching for "professional history" finds chunks about "work experience" — they mean the same thing so their numbers are similar.

**Step 4 — Build the RAG chain**
Connects everything into a pipeline using the `|` operator:

```
Your question
      ↓
Retriever finds relevant chunks → fills {context}
      ↓
Your question fills {question}
      ↓
Prompt combines both
      ↓
LLM generates answer from context only
      ↓
StrOutputParser extracts plain text
      ↓
Answer printed
```

### Why "Answer based only on context below" matters
This instruction stops the AI from using its general training knowledge. Without it, the AI might answer from memory even when your document says something different. This instruction forces grounding — answers must come from your document.

### Why we switched from Gemini to Groq for the LLM
Gemini's free tier has a daily request limit. We hit it during development. Groq is also free but has much higher limits. We kept Gemini only for embeddings because Groq doesn't offer an embedding model.

---

## Day 3 — AI Agent with Tools

### What we built
An agent that actively decides what to do — call a tool or answer directly — based on your question. It can recover from failed tool calls and retry with different inputs.

### What makes something an agent?
The key difference from chat.py and rag.py:

| | chat.py | rag.py | agent.py |
|---|---|---|---|
| Knowledge source | LLM training | Your document | Live tools |
| Decision making | None — always same steps | None — always searches | LLM decides what to do |
| Can recover from errors | No | No | Yes — retries with different approach |
| Loop type | Input → Output | Input → Output | Think → Act → Observe → Repeat |

### The ReAct Pattern
ReAct stands for **Reason + Act**. It's the core pattern of how agents work:

```
Question: "Who is Elon Musk?"
      ↓
Thought: "I need to look this up"
      ↓
Action: call search_wikipedia("Elon Musk")
      ↓
Observation: "Page not found"
      ↓
Thought: "That didn't work, try different query"
      ↓
Action: call search_wikipedia("Elon Musk biography")
      ↓
Observation: "Elon Musk is the wealthiest person..."
      ↓
Thought: "I have enough information now"
      ↓
Final Answer: "Elon Musk is..."
```

The agent loops through Think → Act → Observe until it has a complete answer.

### File: `agent.py`

```python
from dotenv import load_dotenv
import os
from groq import Groq
import json
import wikipedia

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Tool functions
def search_wikipedia(query: str) -> str:
    try:
        result = wikipedia.summary(query, sentences=3)
        return result
    except Exception as e:
        return f"Could not find information: {str(e)}"

def count_words(text: str) -> str:
    count = len(text.split())
    return f"The text contains {count} words."

# Tool descriptions for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for facts about any person, place or topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic to search for"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_words",
            "description": "Count the number of words in a text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to count words in"
                    }
                },
                "required": ["text"]
            }
        }
    }
]

tool_map = {
    "search_wikipedia": search_wikipedia,
    "count_words": count_words
}

def run_agent(question: str):
    messages = [{"role": "user", "content": question}]
    
    while True:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        if message.tool_calls:
            messages.append({
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
                result = tool_map[tool_name](**tool_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            print(f"\nAI: {message.content}\n")
            break
```

### Understanding the tools block

The tools list is a menu you give the LLM. Each entry has:

```python
"name"        → exact function name in your code
"description" → what the tool does — LLM reads this to decide WHEN to use it
"parameters"  → what inputs the function needs
  "properties"  → describes each input individually
    "type"        → what kind of value (string, integer, boolean)
    "description" → what this input means
  "required"    → which inputs are mandatory
```

The description is the most important field. If it's vague, the LLM makes wrong decisions about when to use the tool.

### The tool_map
```python
tool_map = {
    "search_wikipedia": search_wikipedia,
    "count_words": count_words
}
```

When the LLM says "call search_wikipedia", it gives us the name as a string. We can't call a string in Python. `tool_map` lets us look up the actual function using that string name.

### How the LLM decides which tool to call
The LLM doesn't "think" like a human. It pattern-matches your question against tool descriptions based on patterns learned during training. If your question sounds like it matches a description, that tool gets called.

This is why tool descriptions must be specific and clear — they are literally how the agent decides what to do.

### Why we bypassed LangChain in agent.py
LangChain's agent wrappers kept failing with Groq's tool calling format — causing `tool_use_failed` errors. By talking directly to Groq's API, we had full control over the exact format of tool calls, which fixed the errors. The tradeoff: more code to write, but full visibility into every step.

---

## Day 4 — Telecom Support Agent

### What we built
A complete customer support agent for Deutsche Telekom that combines everything from Days 1-3 into one production-ready system.

### Architecture

```
User Question
      ↓
   Agent Brain (LLM)
   reads conversation history + question
   decides what to do
        /              \
search_policy tool    search_wikipedia tool    answer directly
(policy document)     (general knowledge)      (from memory)
        \              /
    Tool result added to history
              ↓
         Final Answer
         added to history
              ↓
    Next question has full context
```

### What's new compared to agent.py

| Feature | agent.py | telecom_agent.py |
|---|---|---|
| Memory | Per question only | Full conversation history |
| Knowledge | Wikipedia only | Policy document + Wikipedia |
| Identity | Generic assistant | Deutsche Telekom support agent |
| Tools | 2 general tools | 2 domain-specific tools |
| System prompt | None | Yes — shapes all responses |

### File: `telecom_agent.py`

```python
from dotenv import load_dotenv
import os
from groq import Groq
import json
import wikipedia
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# PART 1: Build RAG knowledge base from policy document
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

# PART 2: Tool functions
def search_policy(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])

def search_wikipedia(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Could not find information: {str(e)}"

# PART 3: Tool descriptions
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
                    "query": {"type": "string", "description": "The topic to search Wikipedia for"}
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

# PART 4: System prompt + conversation memory
SYSTEM_PROMPT = """You are a helpful customer support agent for Deutsche Telekom.
You help customers with questions about their plans, policies, billing, and network issues.
Always be polite, clear, and concise.
Use the search_policy tool for questions about company policies and plans.
Use the search_wikipedia tool for general technology or industry questions.
If you already know the answer from conversation history, answer directly without using tools."""

conversation_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

# PART 5: Agent loop with memory
def run_agent(question: str):
    conversation_history.append({"role": "user", "content": question})
    
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
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in message.tool_calls
                ]
            })
            
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                result = tool_map[tool_name](**tool_args)
                
                conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            conversation_history.append({
                "role": "assistant",
                "content": message.content
            })
            print(f"\nAgent: {message.content}\n")
            break
```

### The system prompt — why it matters
The system prompt is the first message in `conversation_history`. The user never sees it but it shapes every single response. It tells the LLM:
- Who it is — Deutsche Telekom support agent
- How to behave — polite, clear, concise
- When to use each tool — explicit guidance
- When NOT to use tools — if answer is already in memory

Without a system prompt, the LLM behaves generically. With a good system prompt, it behaves like a trained support agent.

### Conversation memory across turns
Unlike `agent.py` where each question started fresh, `telecom_agent.py` has one `conversation_history` list that grows throughout the entire session. Every message, tool call, tool result, and answer gets added to it.

This is why when you asked about refunds in question 3, the agent didn't search again — the refund policy was already in the conversation history from question 1.

### search_policy vs search_wikipedia — how the agent chooses
The agent reads both descriptions and pattern-matches against your question:

- "What is the refund policy?" → matches "refunds, data plans... company policies" → `search_policy`
- "What is 5G technology?" → matches "general knowledge... technology... not in policy" → `search_wikipedia`
- "Thank you" → matches neither → answers directly from memory

---

## How Everything Connects

```
chat.py          →    Taught us: LLM calls, message history, multi-turn memory
    ↓
rag.py           →    Taught us: Document loading, chunking, embeddings, vector search
    ↓
agent.py         →    Taught us: Tool calling, ReAct loop, decision making, retry logic
    ↓
telecom_agent.py →    Combined all three: RAG as a tool + agent loop + persistent memory
```

Each file is not a replacement for the previous — it builds on top of it. `telecom_agent.py` uses concepts from all three previous days simultaneously.

---

## Key Concepts Glossary

| Term | What it means |
|---|---|
| **LLM** | Large Language Model — the AI brain (Gemini, Groq's Llama) |
| **Token** | A chunk of text the LLM processes — roughly 1 word = 1 token |
| **Embedding** | A list of numbers representing the meaning of a text |
| **Vector Database** | A database that stores embeddings and searches by meaning similarity |
| **FAISS** | Meta's vector database — stores embeddings locally, no server needed |
| **RAG** | Retrieval Augmented Generation — answer from documents, not memory |
| **Chunk** | A small piece of a document (500 chars in our case) |
| **Chunk overlap** | Shared characters between consecutive chunks to preserve context |
| **ReAct** | Reason + Act — the Think → Act → Observe agent loop pattern |
| **Tool** | A function the agent can call to get information or take action |
| **Tool description** | Text telling the LLM when and how to use a tool |
| **tool_map** | Dictionary mapping tool name strings to actual Python functions |
| **System prompt** | Hidden instruction shaping AI behaviour — user never sees it |
| **Conversation history** | Running list of all messages — how memory works |
| **Exponential backoff** | Waiting 1s, 2s, 4s between retries — prevents overwhelming APIs |
| **Intent inference** | LLM guessing what you meant ("Elon" → "Elon Musk") |

---

*Documentation covers Days 1–4. Day 5 covers error handling, evaluation, logging, and Docker.*
