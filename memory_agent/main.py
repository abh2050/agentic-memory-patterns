import os
import time
import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import List

from memory_agent.config import DEBOUNCE_SECONDS, MAX_FACTS, TOKEN_BUDGET
from memory_agent.models import AgentState, UserMessage, FactCandidate, MemoryStore, MemoryContext, MemoryHistory
from memory_agent.redis_client import redis_client
from memory_agent.graph import graph
from memory_agent.nodes.extraction import extraction_manager
from memory_agent.nodes.memory import save_memory_atomic, load_memory, memory_node_internal
from memory_agent.nodes.prompt_builder import prompt_builder_node

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

def run_act_1():
    print("\n[DEMO] ACT 1 — Debounce under burst load")
    redis_client.delete_stream("extraction:queue")
    
    # Mock extraction_manager.process_queue to count calls
    original_process = extraction_manager.process_queue
    call_count = 0
    def counted_process():
        nonlocal call_count
        call_count += 1
        original_process()
    
    extraction_manager.process_queue = counted_process
    
    for i in range(4):
        msg = UserMessage(content=f"Message {i}")
        # Simulate conversation_node publishing
        redis_client.xadd_model("extraction:queue", msg)
        extraction_manager.handle_message(msg)
        time.sleep(0.5)
    
    print(f"Redis stream depth before timer: {redis_client.get_stream_length('extraction:queue')}")
    
    # Wait for debounce
    time.sleep(DEBOUNCE_SECONDS + 1)
    
    print(f"Redis stream depth after timer: {redis_client.get_stream_length('extraction:queue')}")
    assert call_count == 1, f"Expected 1 extraction call, got {call_count}"
    print("ACT 1 PASSED")
    
    # Restore original
    extraction_manager.process_queue = original_process

def run_act_2():
    print("\n[DEMO] ACT 2 — Confidence filtering")
    # Clean memory
    if os.path.exists("memory.json"): os.remove("memory.json")
    
    candidates = [
        FactCandidate(category="test", value="Low confidence", confidence=0.1),
        FactCandidate(category="test", value="Medium confidence", confidence=0.5),
        FactCandidate(category="test", value="High confidence", confidence=0.8),
        FactCandidate(category="test", value="Very high confidence", confidence=0.95),
    ]
    
    memory_node_internal(candidates, "Transcript")
    
    store = load_memory()
    print(f"Accepted: {len(store.facts)}, Discarded: {len(candidates) - len(store.facts)}")
    for f in store.facts:
        print(f"Fact: {f.value}, Confidence: {f.confidence}")
        assert f.confidence >= 0.7, f"Fact with confidence {f.confidence} should have been filtered"
    
    assert len(store.facts) == 2
    print("ACT 2 PASSED")

def run_act_3():
    print("\n[DEMO] ACT 3 — Eviction under cap pressure")
    # Pre-load 98 facts
    facts = []
    base_time = datetime.utcnow() - timedelta(days=30)
    for i in range(98):
        facts.append(FactCandidate(
            category="test",
            value=f"Old fact {i}",
            confidence=0.71,
            timestamp=base_time + timedelta(minutes=i)
        ))
    
    store = MemoryStore(facts=facts)
    save_memory_atomic(store)
    
    # Add 5 new facts
    new_facts = [
        FactCandidate(category="test", value=f"New fact {i}", confidence=0.85 + (i*0.02))
        for i in range(5)
    ]
    
    memory_node_internal(new_facts, "Transcript")
    
    final_store = load_memory()
    print(f"Final fact count: {len(final_store.facts)}")
    assert len(final_store.facts) == 100
    
    # Check that oldest lowest confidence were removed
    # Oldest were "Old fact 0", "Old fact 1", "Old fact 2"
    fact_values = [f.value for f in final_store.facts]
    assert "Old fact 0" not in fact_values
    assert "Old fact 1" not in fact_values
    assert "Old fact 2" not in fact_values
    
    print("Before/After Table (Sample):")
    print("ID | Confidence | Timestamp")
    for f in final_store.facts[:5]:
        print(f"{str(f.id)[:8]} | {f.confidence:.2f} | {f.timestamp}")
    
    print("ACT 3 PASSED")

def run_act_4():
    print("\n[DEMO] ACT 4 — Token budget enforcement")
    # Load 60 facts
    facts = [
        FactCandidate(category="test", value=f"Fact {i} with some extra text to consume tokens " * 5, confidence=0.5 + (i*0.008))
        for i in range(60)
    ]
    store = MemoryStore(facts=facts)
    save_memory_atomic(store)
    
    state = AgentState()
    result = prompt_builder_node(state)
    
    prompt = result["current_prompt"]
    # Rough token count: words * 1.3
    tokens = int(len(prompt.split()) * 1.3)
    print(f"Estimated tokens: {tokens}")
    assert tokens <= TOKEN_BUDGET, f"Token count {tokens} exceeds budget {TOKEN_BUDGET}"
    
    # Check descending confidence
    lines = [l for l in prompt.split("\n") if l.startswith("- ")]
    confidences = []
    for l in lines:
        conf = float(l.split("Confidence: ")[1].split(")")[0])
        confidences.append(conf)
    
    assert confidences == sorted(confidences, reverse=True), "Facts not in descending confidence order"
    print("ACT 4 PASSED")

def run_act_5():
    print("\n[DEMO] ACT 5 — Concurrent write safety")
    
    def worker(thread_id):
        store = MemoryStore(context=MemoryContext(work=f"Thread {thread_id} was here"))
        save_memory_atomic(store)
        # print(f"Thread {thread_id} finished")

    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    final_store = load_memory()
    print(f"Winning context: {final_store.context.work}")
    assert "Thread" in final_store.context.work
    print("ACT 5 PASSED")

if __name__ == "__main__":
    try:
        run_act_1()
        run_act_2()
        run_act_3()
        run_act_4()
        run_act_5()
        print("\nALL ACTS PASSED SUCCESSFULLY!")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
