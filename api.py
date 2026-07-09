from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from telecom_agent import run_agent, conversation_history, SYSTEM_PROMPT
import logging
from groq import Groq

# -------------------------------------------------------
# FastAPI app
# -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),  # saves to file
        logging.StreamHandler()             # also prints to terminal
    ]
)
logger = logging.getLogger(__name__)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# Request model
# -------------------------------------------------------
class ChatRequest(BaseModel):
    message: str

# -------------------------------------------------------
# Chat endpoint
# -------------------------------------------------------

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.message
    logger.info(f"Received message: {question}")

    conversation_history.append({
        "role": "user",
        "content": question
    })

    tools_used = []  # track which tools were called

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
            return {"response": "I'm having technical difficulties. Please try again.", "tools_used": []}

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

                # Track tool usage
                tools_used.append(tool_name)

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

            # Return answer + which tools were used
            return {
                "response": final_answer,
                "tools_used": tools_used
            }

# -------------------------------------------------------
# Reset endpoint
# Clears conversation history for a fresh session
# -------------------------------------------------------
@app.post("/reset")
async def reset():
    conversation_history.clear()
    conversation_history.append({"role": "system", "content": SYSTEM_PROMPT})
    return {"status": "conversation reset"}

# -------------------------------------------------------
# Health check
# -------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}