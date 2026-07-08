# Deutsche Telekom Customer Support AI Agent

An intelligent customer support agent built for The Talent Hack hackathon.
Combines RAG, LLM orchestration, and agentic workflows to answer
customer queries about Deutsche Telekom's policies and services.

## Architecture
User Question
↓
Agent Brain (LLM - Groq)
/              
search_policy        search_wikipedia
(RAG over policy     (Wikipedia API for
document via         general knowledge)
FAISS vector DB)
\              /
Final Answer
(with memory across turns)

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (openai/gpt-oss-20b) |
| Embeddings | Google Gemini (gemini-embedding-001) |
| Vector Database | FAISS (local) |
| RAG Framework | LangChain |
| Agent Framework | Groq native tool calling |
| Containerization | Docker |

## Features

- RAG pipeline over telecom policy document
- Real-time Wikipedia search for general knowledge
- Persistent conversation memory across turns
- Automatic tool selection based on question type
- Error handling with exponential backoff retry
- Structured logging to agent.log
- 100% accuracy on 10-question evaluation suite
- Average response latency under 3 seconds

## Setup

### Prerequisites
- Python 3.13+
- Docker (optional)
- Google Gemini API key (free at aistudio.google.com)
- Groq API key (free at console.groq.com)

### Installation

1. Clone the repository
2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:


GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key

## Running the Agent

### Option 1 — Direct Python
```bash
python telecom_agent.py
```

### Option 2 — Docker
```bash
docker-compose up
```

## Running Evaluation
```bash
python evaluate.py
```
Expected output: 10/10 questions passed, 100% accuracy

## Project Structure

ai-hackathon/
├── telecom_agent.py      # Main agent — combines RAG + tools + memory
├── evaluate.py           # Evaluation script — measures accuracy
├── agent.py              # Standalone agent with Wikipedia + word count tools
├── rag.py                # RAG pipeline — document Q&A
├── chat.py               # Multi-turn chatbot
├── telecom_policy.txt    # Knowledge base — Deutsche Telekom policies
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Container orchestration
└── agent.log             # Runtime logs

## Sample Interactions

**Policy question:**

You: What is the refund policy?
Agent: Customers are eligible for a full refund within 30 days
of purchase. Refunds are processed within 5-7 business days.

**Technology question:**
You: What is 5G technology?
Agent: 5G is the fifth generation of wireless technology...
[retrieved from Wikipedia]

**Multi-turn memory:**

You: What are the data plans?
Agent: We have three plans — Basic (10GB, €15), Standard (50GB, €25),
Premium (Unlimited 5G, €45)
You: Which one has data rollover?
Agent: Data rollover is available on Standard and Premium plans.
[answered from memory — no tool call needed]

## Evaluation Results

| Category | Score |
|---|---|
| Refund policies | 2/2 |
| Data plans | 2/2 |
| Contract terms | 2/2 |
| Customer support | 1/1 |
| Roaming | 2/2 |
| Network outages | 1/1 |
| **Total** | **10/10 (100%)** |

## What I'd Build Next

- Streamlit UI for browser-based demo
- Support for multiple documents
- User authentication
- Response caching to reduce latency
- Fine-tuned evaluation with semantic similarity scoring