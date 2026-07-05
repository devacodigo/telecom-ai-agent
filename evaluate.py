from dotenv import load_dotenv
import os
from groq import Groq
import json
import wikipedia
import time
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

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
# PART 1: Build RAG knowledge base (same as telecom_agent.py)
# -------------------------------------------------------
print("Setting up knowledge base...")
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

# -------------------------------------------------------
# PART 2: Tools (same as telecom_agent.py)
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
            "description": "Search Wikipedia for general knowledge about telecom technology.",
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

# -------------------------------------------------------
# PART 3: Evaluation dataset
# Each entry has:
# - question: what we ask the agent
# - expected_keywords: words that MUST appear in a correct answer
# - category: which type of question this is
# -------------------------------------------------------
eval_dataset = [
    {
        "question": "How many days do I have to request a refund?",
        "expected_keywords": ["30"],
        "category": "refund"
    },
    {
        "question": "How long does a refund take to process?",
        "expected_keywords": ["5", "7"],
        "category": "refund"
    },
    {
        "question": "What is the price of the Premium Plan?",
        "expected_keywords": ["45"],
        "category": "plans"
    },
    {
        "question": "How much data does the Standard Plan include?",
        "expected_keywords": ["50"],
        "category": "plans"
    },
    {
        "question": "What is the early termination fee?",
        "expected_keywords": ["50"],
        "category": "contract"
    },
    {
        "question": "How long is the minimum contract period?",
        "expected_keywords": ["12"],
        "category": "contract"
    },
    {
        "question": "What is the support phone number?",
        "expected_keywords": ["0800"],
        "category": "support"
    },
    {
        "question": "How many countries support roaming?",
        "expected_keywords": ["50"],
        "category": "roaming"
    },
    {
        "question": "How much does the international roaming pass cost per day?",
        "expected_keywords": ["10"],
        "category": "roaming"
    },
    {
        "question": "What credit do I get if my network is down for more than 24 hours?",
        "expected_keywords": ["1 day", "credit", "bill"],
        "category": "outage"
    }
]

# -------------------------------------------------------
# PART 4: Single question agent
# Simplified version — no memory needed for evaluation
# each question is independent
# -------------------------------------------------------
def ask_agent(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a Deutsche Telekom support agent. Answer questions using the available tools."
        },
        {
            "role": "user",
            "content": question
        }
    ]

    for attempt in range(3):
        try:
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
                    return message.content

        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {str(e)}"

# -------------------------------------------------------
# PART 5: Run the evaluation
# -------------------------------------------------------
print("\n" + "="*60)
print("EVALUATION REPORT — Deutsche Telekom Support Agent")
print("="*60 + "\n")

results = []
passed = 0
failed = 0

for i, item in enumerate(eval_dataset):
    question = item["question"]
    keywords = item["expected_keywords"]
    category = item["category"]

    print(f"[{i+1}/10] {question}")

    # Get agent's answer
    start_time = time.time()
    answer = ask_agent(question)
    latency = round(time.time() - start_time, 2)

    # Check if all expected keywords appear in the answer
    # We check lowercase to make matching case-insensitive
    answer_lower = answer.lower()
    keyword_found = any(kw.lower() in answer_lower for kw in keywords)

    if keyword_found:
        status = "✅ PASS"
        passed += 1
    else:
        status = "❌ FAIL"
        failed += 1

    print(f"   Status   : {status}")
    print(f"   Latency  : {latency}s")
    print(f"   Expected : {keywords}")
    print(f"   Answer   : {answer[:120]}...")
    print()

    results.append({
        "question": question,
        "category": category,
        "status": "PASS" if keyword_found else "FAIL",
        "latency": latency,
        "answer": answer
    })

    # Small delay between questions to avoid rate limits
    time.sleep(1)

# -------------------------------------------------------
# PART 6: Summary
# -------------------------------------------------------
accuracy = round((passed / len(eval_dataset)) * 100)
avg_latency = round(sum(r["latency"] for r in results) / len(results), 2)

print("="*60)
print("SUMMARY")
print("="*60)
print(f"Total questions : {len(eval_dataset)}")
print(f"Passed          : {passed}")
print(f"Failed          : {failed}")
print(f"Accuracy        : {accuracy}%")
print(f"Avg latency     : {avg_latency}s")
print()

# Break down by category
categories = {}
for r in results:
    cat = next(item["category"] for item in eval_dataset if item["question"] == r["question"])
    if cat not in categories:
        categories[cat] = {"pass": 0, "total": 0}
    categories[cat]["total"] += 1
    if r["status"] == "PASS":
        categories[cat]["pass"] += 1

print("Results by category:")
for cat, data in categories.items():
    cat_accuracy = round((data["pass"] / data["total"]) * 100)
    print(f"  {cat:<12} : {data['pass']}/{data['total']} ({cat_accuracy}%)")

print("\nEvaluation complete.")