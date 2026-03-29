import os
import json
import logging
import threading
from datetime import datetime
from memory_agent.config import MEMORY_FILE, MEMORY_TMP_FILE, CONFIDENCE_THRESHOLD, MAX_FACTS
from memory_agent.models import MemoryStore, FactCandidate, MemoryUpdatedEvent
from memory_agent.redis_client import redis_client

logger = logging.getLogger("MemoryNode")
lock = threading.Lock()

def load_memory() -> MemoryStore:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return MemoryStore.model_validate_json(f.read())
    return MemoryStore()

# --- PATTERN 4: ATOMIC PERSISTENCE ---
def save_memory_atomic(store: MemoryStore):
    """
    os.replace is atomic on POSIX systems.
    The lock protects against concurrent Python threads.
    The tmp-then-rename pattern ensures readers never observe a partial write.
    """
    with lock:
        data = store.model_dump_json(indent=2)
        with open(MEMORY_TMP_FILE, "w") as f:
            f.write(data)
        os.replace(MEMORY_TMP_FILE, MEMORY_FILE)

# --- PATTERN 5: EVICTION AND FILTERING ---
def memory_node_internal(new_facts: list[FactCandidate], transcript: str):
    logger.info(f"[MemoryNode] Processing {len(new_facts)} new fact candidates")
    
    store = load_memory()
    
    # 1. Confidence filtering
    accepted_facts = [f for f in new_facts if f.confidence >= CONFIDENCE_THRESHOLD]
    discarded_count = len(new_facts) - len(accepted_facts)
    
    # 2. Add to store
    store.facts.extend(accepted_facts)
    
    # 3. Enforce 100-fact cap (oldest-lowest-confidence eviction)
    evicted_count = 0
    if len(store.facts) > MAX_FACTS:
        # Sort by confidence ascending, then timestamp ascending (oldest first)
        store.facts.sort(key=lambda x: (x.confidence, x.timestamp))
        evicted_count = len(store.facts) - MAX_FACTS
        store.facts = store.facts[evicted_count:]
    
    # Write atomically
    save_memory_atomic(store)
    
    # Publish event
    event = MemoryUpdatedEvent(
        version=0, # Simplified for demo
        facts_added=len(accepted_facts),
        facts_discarded=discarded_count,
        facts_evicted=evicted_count
    )
    redis_client.xadd_model("memory:updates", event)
    logger.info(f"[MemoryNode] Updated memory. Added: {len(accepted_facts)}, Discarded: {discarded_count}, Evicted: {evicted_count}")

def memory_node(state: dict) -> dict:
    # In the graph, memory_node might be called if we use a synchronous flow.
    # But extraction is async. So this node might be a pass-through or triggered by extraction.
    return state
