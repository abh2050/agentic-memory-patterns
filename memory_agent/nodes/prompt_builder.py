import logging
import tiktoken
from memory_agent.config import TOKEN_BUDGET, MODEL_NAME
from memory_agent.models import AgentState, MemoryStore, BuildReport
from memory_agent.nodes.memory import load_memory
from memory_agent.redis_client import redis_client

logger = logging.getLogger("PromptBuilderNode")

def count_tokens(text: str) -> int:
    # Simplified token estimation as requested: words * 1.3
    # Or use tiktoken for more accuracy if available.
    # The prompt says: "running token estimate (words * 1.3)"
    return int(len(text.split()) * 1.3)

# --- PATTERN 6: CONTEXT-AWARE PROMPT BUILDING ---
def prompt_builder_node(state: AgentState) -> dict:
    logger.info("[PromptBuilderNode] Building system prompt")
    
    store = load_memory()
    
    # Start building prompt
    prompt_parts = []
    
    # 1. Context layers
    context_str = f"Work: {store.context.work}\nPersonal: {store.context.personal}\nTop of Mind: {store.context.top_of_mind}"
    prompt_parts.append("## USER CONTEXT")
    prompt_parts.append(context_str)
    
    # 2. History layers
    history_str = f"Recent: {store.history.recent_months}\nEarlier: {store.history.earlier_context}\nLong-term: {store.history.long_term_background}"
    prompt_parts.append("\n## CONVERSATION HISTORY")
    prompt_parts.append(history_str)
    
    context_tokens = count_tokens(context_str)
    history_tokens = count_tokens(history_str)
    
    # 3. Facts (sorted by confidence descending)
    sorted_facts = sorted(store.facts, key=lambda x: x.confidence, reverse=True)
    
    prompt_parts.append("\n## KNOWN FACTS")
    
    facts_included = 0
    facts_excluded = 0
    current_tokens = context_tokens + history_tokens + count_tokens("\n".join(prompt_parts))
    
    for fact in sorted_facts:
        fact_str = f"- {fact.value} (Confidence: {fact.confidence:.2f})"
        fact_tokens = count_tokens(fact_str)
        
        if current_tokens + fact_tokens > TOKEN_BUDGET:
            facts_excluded += 1
            continue
        
        prompt_parts.append(fact_str)
        current_tokens += fact_tokens
        facts_included += 1
    
    final_prompt = "\n".join(prompt_parts)
    
    # Log report
    report = BuildReport(
        context_tokens=context_tokens,
        history_tokens=history_tokens,
        facts_included=facts_included,
        facts_excluded=facts_excluded,
        total_tokens=current_tokens
    )
    logger.info(f"[PromptBuilderNode] Build Report: {report.model_dump_json()}")
    
    # Set in Redis for shared state access if needed
    redis_client.set_raw("prompt:current", final_prompt)
    
    return {
        "current_prompt": final_prompt
    }
