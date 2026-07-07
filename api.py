from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from telecom_agent import run_agent, conversation_history, SYSTEM_PROMPT

# -------------------------------------------------------
# FastAPI app
# -------------------------------------------------------
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
@app.post("/chat")
async def chat(request: ChatRequest):
    answer = run_agent(request.message)
    return {"response": answer}

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