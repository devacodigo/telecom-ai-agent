from dotenv import load_dotenv
import os
from groq import Groq
import json
import wikipedia

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Step 1: Define tools as functions ---
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic."""
    try:
        result = wikipedia.summary(query, sentences=3)
        return result
    except Exception as e:
        return f"Could not find information: {str(e)}"

def count_words(text: str) -> str:
    """Count words in a text."""
    count = len(text.split())
    return f"The text contains {count} words."

# --- Step 2: Tell the LLM what tools exist ---
# This is how the LLM knows what tools it can call and what inputs they need
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for facts about any person, place or topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The topic to search for"}
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
                    "text": {"type": "string", "description": "The text to count words in"}
                },
                "required": ["text"]
            }
        }
    }
]

# Map tool names to actual functions
tool_map = {
    "search_wikipedia": search_wikipedia,
    "count_words": count_words
}

def run_agent(question: str):
    messages = [{"role": "user", "content": question}]
    
    print(f"\nThinking...")
    
    # Step 3: Agent loop — keep going until LLM stops calling tools
    while True:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        # If LLM wants to call a tool
        if message.tool_calls:
            # Add LLM's decision to message history
            messages.append({
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
            
            # Call each tool the LLM requested
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                print(f"Using tool: {tool_name} with input: {tool_args}")
                
                # Actually run the tool function
                result = tool_map[tool_name](**tool_args)
                print(f"Tool result: {result[:100]}...")
                
                # Add tool result to message history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        
        # If LLM has final answer, stop the loop
        else:
            print(f"\nAI: {message.content}\n")
            break

print("Agent ready! Ask me anything. Type 'quit' to exit.\n")

while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    run_agent(question)