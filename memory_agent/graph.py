from langgraph.graph import StateGraph, END
from memory_agent.models import AgentState
from memory_agent.nodes.conversation import conversation_node
from memory_agent.nodes.prompt_builder import prompt_builder_node
from memory_agent.nodes.extraction import extraction_node
from memory_agent.nodes.memory import memory_node

def create_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("conversation", conversation_node)
    workflow.add_node("prompt_builder", prompt_builder_node)
    
    # Define edges
    # conversation -> prompt_builder (for next turn)
    workflow.add_edge("conversation", "prompt_builder")
    workflow.add_edge("prompt_builder", END)

    # Entry point
    workflow.set_entry_point("conversation")

    return workflow.compile()

graph = create_graph()
