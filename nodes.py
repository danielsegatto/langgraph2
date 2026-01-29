import json
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

def sentence_splitter_node(state: AgentState, llm):
    """Breaks raw text into individual sentences."""
    system_instructions = (
        "You are a linguistic tool. Your task is to break the provided text into individual sentences. "
        "Respond ONLY with a JSON object. "
        "Example Output: {\"sentences\": [\"Sentence A\", \"Sentence B\"]}"
    )
    response = llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content=f"SENTENCE: '{state['original_text']}'")
    ])
    data = json.loads(response.content)
    return {"sentences": data.get("sentences", [])}

def triple_translator_node(state: AgentState, llm):
    """Generates three English translations for each sentence."""
    system_instructions = (
        "You are a professional translator and expert polyglot. For each Portuguese sentence provided, create 3 English versions. "
        "Choose the 3 best stylistic variations that fit the context of the sentence. "
        "Respond ONLY with a JSON object. No numbers inside strings. "
        "Example Output: {\"results\": [{\"original\": \"...\", \"versions\": [\"V1\", \"V2\", \"V3\"]}]}"
    )
    input_list = "\n".join(state['sentences'])
    response = llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content=f"TRANSLATE: {input_list}")
    ])
    data = json.loads(response.content)
    return {"final_results": data.get("results", [])}

def style_refiner_node(state: AgentState, llm):
    """Provides three new variations of a selected English sentence."""
    system_instructions = (
        "You are a writing coach. The user has selected a specific English sentence and wants more variations of it. "
        "Generate 3 NEW natural English variations. Do NOT add numbers."
        "Respond ONLY with a JSON object. "
        "Example Output: {\"results\": [{\"original\": \"...\", \"versions\": [\"Alt 1\", \"Alt 2\", \"Alt 3\"]}]}"
    )
    response = llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content=f"REFINE: {state['refinement_text']}")
    ])
    data = json.loads(response.content)
    return {"final_results": data.get("results", [])}