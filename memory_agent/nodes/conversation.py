import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from memory_agent.config import MODEL_NAME, OPENAI_API_KEY
from memory_agent.models import AgentState, UserMessage
from memory_agent.redis_client import redis_client

logger = logging.getLogger("ConversationNode")

# --- PATTERN 1: CONVERSATION ENTRY ---
def conversation_node(state: AgentState) -> dict:
    logger.info("[ConversationNode] Processing message")
    
    if not state.current_message:
        return {"last_response": "No message to process."}

    # Initialize LLM
    llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)
    
    # Use current prompt from state (built by PromptBuilderNode)
    system_prompt = state.current_prompt or "You are a helpful assistant."
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.current_message.content)
    ]
    
    # Generate response
    response = llm.invoke(messages)
    state.last_response = response.content
    
    # Publish to Redis stream for async extraction
    # --- PATTERN 2: ASYNC EXTRACTION TRIGGER ---
    redis_client.xadd_model("extraction:queue", state.current_message)
    logger.info(f"[ConversationNode] Published message {state.current_message.id} to extraction:queue")
    
    return {
        "last_response": state.last_response,
        "memory_version": state.memory_version
    }
