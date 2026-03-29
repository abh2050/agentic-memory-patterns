import logging
import threading
import time
from datetime import datetime
from typing import List
from langchain_openai import ChatOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
import openai

from memory_agent.config import MODEL_NAME, OPENAI_API_KEY, DEBOUNCE_SECONDS
from memory_agent.models import UserMessage, FactCandidate, FactExtractionResult, MemoryContext
from memory_agent.redis_client import redis_client

logger = logging.getLogger("ExtractionNode")

# --- PATTERN 3: DEBOUNCE EXTRACTION ---
class ExtractionManager:
    def __init__(self):
        self.timer = None
        self.lock = threading.Lock()
        self.llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)

    def handle_message(self, message: UserMessage):
        with self.lock:
            # Update last seen in Redis
            redis_client.set_raw("debounce:last_seen", datetime.utcnow().isoformat())
            
            if self.timer:
                self.timer.cancel()
            
            self.timer = threading.Timer(DEBOUNCE_SECONDS, self.process_queue)
            self.timer.start()
            logger.info(f"[ExtractionNode] Debounce timer reset for {DEBOUNCE_SECONDS}s")

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def extract_facts(self, transcript: str) -> List[FactCandidate]:
        structured_llm = self.llm.with_structured_output(FactExtractionResult)
        system_msg = "Extract discrete facts about the user from this conversation transcript. Return only facts you are confident about. Include a confidence score for each."
        result = structured_llm.invoke([
            ("system", system_msg),
            ("human", transcript)
        ])
        return result.facts

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def update_context(self, transcript: str, current: MemoryContext) -> MemoryContext:
        structured_llm = self.llm.with_structured_output(MemoryContext)
        system_msg = "Update the user context summary based on this transcript. Keep each field to 1-3 sentences. Preserve existing context unless the transcript clearly supersedes it."
        result = structured_llm.invoke([
            ("system", system_msg),
            ("human", f"Current Context: {current.model_dump_json()}\n\nTranscript: {transcript}")
        ])
        return result

    def process_queue(self):
        logger.info("[ExtractionNode] Timer fired, processing queue")
        # Read all pending messages from stream
        messages = redis_client.xread_models("extraction:queue", UserMessage)
        if not messages:
            return

        transcript = "\n".join([f"User: {m.content}" for m in messages])
        
        # Extract facts
        facts = self.extract_facts(transcript)
        logger.info(f"[ExtractionNode] Extracted {len(facts)} fact candidates")
        
        # In a real LangGraph, we'd pass this to memory_node. 
        # For this demo, we'll simulate the handoff or call memory_node logic.
        # The prompt says: "Passes extracted candidates to memory_node."
        # We'll use a direct call or another Redis stream if needed, 
        # but the graph definition will handle the node transition.
        # However, extraction_node runs in a separate thread.
        
        from memory_agent.nodes.memory import memory_node_internal
        memory_node_internal(facts, transcript)

extraction_manager = ExtractionManager()

def extraction_node(state: dict) -> dict:
    # This node is triggered by the background thread consuming from Redis
    # In the LangGraph, it might be a dummy or a placeholder if the thread handles it.
    # But the prompt says "Runs as a background LangGraph node triggered by a separate thread".
    # We'll implement the thread loop in main.py or here.
    pass
