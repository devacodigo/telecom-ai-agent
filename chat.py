from dotenv import load_dotenv
import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY is not set. Add it to your .env file.")
    raise SystemExit(1)

# Prefer models with separate free-tier quotas; fall back if one is unavailable.
MODELS = ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.0-flash"]
model_index = 0
llm = ChatGoogleGenerativeAI(model=MODELS[model_index], google_api_key=api_key)

messages = [SystemMessage(content="You are a helpful assistant.")]

print(f"Chatbot ready (model: {MODELS[model_index]}). Type 'quit' to exit.\n")


def friendly_error(exc: ChatGoogleGenerativeAIError) -> str:
    err = str(exc).lower()
    if "resource_exhausted" in err or "429" in err:
        return (
            "Quota exceeded for all tried models. Wait a minute and try again, "
            "or check usage at https://ai.dev/rate-limit"
        )
    if "not_found" in err or "404" in err:
        return "That model is not available on your API key."
    if "api key" in err or "permission" in err or "401" in err or "403" in err:
        return "Invalid or unauthorized API key. Check GOOGLE_API_KEY in your .env file."
    return f"API error: {exc}"


def invoke_with_fallback(msgs):
    global llm, model_index

    last_error = None
    for i in range(model_index, len(MODELS)):
        if i != model_index:
            llm = ChatGoogleGenerativeAI(model=MODELS[i], google_api_key=api_key)
            model_index = i
            print(f"Switched to model: {MODELS[i]}")

        try:
            return llm.invoke(msgs)
        except ChatGoogleGenerativeAIError as e:
            last_error = e
            err = str(e).lower()
            retryable = "not_found" in err or "404" in err or "resource_exhausted" in err or "429" in err
            if retryable and i < len(MODELS) - 1:
                continue
            raise

    raise last_error


while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    messages.append(HumanMessage(content=user_input))
    try:
        response = invoke_with_fallback(messages)
    except ChatGoogleGenerativeAIError as e:
        messages.pop()
        print(f"AI: {friendly_error(e)}\n")
        continue

    messages.append(AIMessage(content=response.content))
    print(f"AI: {response.content}\n")
