from typing import TypedDict, List

class AgentState(TypedDict):
    """
    The state of the translation agent.
    
    original_text: The full text provided by the user.
    mode: Determines the path ('translate' or 'refine').
    refinement_text: The specific sentence selected for iteration.
    sentences: The individual Portuguese sentences after splitting.
    final_results: The list of English versions returned by the AI.
    """
    original_text: str
    mode: str           
    refinement_text: str 
    sentences: List[str]
    final_results: List[dict]