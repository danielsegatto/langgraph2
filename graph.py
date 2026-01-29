from langgraph.graph import StateGraph, START, END
from state import AgentState
from nodes import sentence_splitter_node, triple_translator_node, style_refiner_node

def create_app(llm):
    """
    Constructs the LangGraph workflow.
    """
    
    # 1. Internal Router Logic
    def router_logic(state: AgentState):
        return "refiner" if state.get("mode") == "refine" else "splitter"

    # 2. Initialize the Graph
    workflow = StateGraph(AgentState)

    # 3. Add Nodes 
    # We use lambda functions to pass the 'llm' into our node functions
    workflow.add_node("splitter", lambda state: sentence_splitter_node(state, llm))
    workflow.add_node("translator", lambda state: triple_translator_node(state, llm))
    workflow.add_node("refiner", lambda state: style_refiner_node(state, llm))

    # 4. Define the Paths
    workflow.add_conditional_edges(
        START, 
        router_logic, 
        {"splitter": "splitter", "refiner": "refiner"}
    )
    
    workflow.add_edge("splitter", "translator")
    workflow.add_edge("translator", END)
    workflow.add_edge("refiner", END)

    # 5. Compile the Application
    return workflow.compile()