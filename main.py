import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
llm = ChatGroq(
    temperature=0,
    model_name="openai/gpt-oss-120b", 
    groq_api_key=os.getenv("GROQ_API_KEY") 
)

# 1. We add 'mode' and 'refinement_text' to the State
class AgentState(TypedDict):
    original_text: str
    mode: str           # "translate" or "refine"
    refinement_text: str # The specific sentence the user wants to iterate on
    sentences: List[str]
    final_results: List[dict]

# --- NODE 1: SENTENCE SPLITTER ---
def sentence_splitter_node(state: AgentState):
    system_instructions = (
        "You are a linguistic tool. Your task is to break the provided text into individual sentences. "
        "Return each sentence on a new line. Do not add numbers, bullet points, or any introductory text."
    )
    response = llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content=state['original_text'])
    ])
    content = response.content
    if isinstance(content, list): content = content[0].get('text', '')
    sentence_list = [s.strip() for s in content.split('\n') if s.strip()]
    return {"sentences": sentence_list}

# --- NODE 2: TRIPLE TRANSLATOR ---
def triple_translator_node(state: AgentState):
    system_instructions = (
        "You are a professional translator and expert polyglot. For each Portuguese sentence provided, create 3 English versions. "
        "Choose the 3 best stylistic variations that fit the context of the sentence. "
        "Return the versions as a plain list of strings. Do NOT add numbers (1., 2., etc.) inside the strings."
        "\n\nRespond ONLY with a JSON array of objects. Structure:"
        '\n[{"original": "...", "versions": ["Version A", "Version B", "Version C"]}]'
    )
    input_list = "\n".join(state['sentences'])
    response = llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content=f"Sentences to translate:\n{input_list}")
    ])
    raw_content = response.content.replace("```json", "").replace("```", "").strip()
    try:
        return {"final_results": json.loads(raw_content)}
    except:
        return {"final_results": []}

# --- NODE 3: STYLE REFINER ---
def style_refiner_node(state: AgentState):
    # This node takes ONE English sentence and finds 3 new ways to say it
    system_instructions = (
        "You are a writing coach. The user has selected a specific English sentence and wants more variations of it. "
        "Generate 3 NEW natural English variations. Do NOT add numbers."
        "\n\nRespond ONLY with a JSON array containing one object. Structure:"
        '\n[{"original": "...", "versions": ["Variation 1", "Variation 2", "Variation 3"]}]'
    )
    response = llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content=f"Provide variations for: {state['refinement_text']}")
    ])
    raw_content = response.content.replace("```json", "").replace("```", "").strip()
    try:
        return {"final_results": json.loads(raw_content)}
    except:
        return {"final_results": []}

# --- THE STATIC ROUTER (No AI involved) ---
def router_logic(state: AgentState):
    if state.get("mode") == "refine":
        return "refiner"
    return "splitter"

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)

workflow.add_node("splitter", sentence_splitter_node)
workflow.add_node("translator", triple_translator_node)
workflow.add_node("refiner", style_refiner_node)

# We use the router_logic to decide where to go immediately after START
workflow.add_conditional_edges(
    START,
    router_logic,
    {
        "splitter": "splitter",
        "refiner": "refiner"
    }
)

workflow.add_edge("splitter", "translator")
workflow.add_edge("translator", END)
workflow.add_edge("refiner", END)

app = workflow.compile()

# --- INTERACTIVE LOOP ---
compiled_list = []

print(">>> Translation Agent Ready. <<<")

while True:
    user_input = input("\n =>: ")

    if user_input.lower() in ["q", "exit"]:
        break

    # We start every new text in "translate" mode
    result = app.invoke({"original_text": user_input, "mode": "translate"})

    # We keep the current view in a variable so we can "refine" it
    current_results = result['final_results']

    while True:
        for entry in current_results:
            versions = entry.get('versions', [])
            for idx, version in enumerate(versions, 1):
                print(f"[{idx}] {version}")
        
        print("\nOPTIONS: [1,2.. to Save] | [r1, r2.. to Refine] | [Enter for New Text]")
        action = input(" Action: ").strip().lower()

        if not action: break # Exit inner loop for new text

        # If user wants to REFINE
        if action.startswith('r'):
            try:
                # Get the index (e.g., 'r1' -> index 0)
                idx = int(action[1:]) - 1
                selected_for_refinement = current_results[0]['versions'][idx]
                
                # INVOKE GRAPH IN REFINE MODE
                refine_result = app.invoke({
                    "refinement_text": selected_for_refinement, 
                    "mode": "refine"
                })
                # Update current view with the 3 new versions
                current_results = refine_result['final_results']
            except:
                print("Invalid refinement selection.")
        
        # If user wants to SAVE
        elif action[0].isdigit():
            indices = [int(x.strip()) for x in action.split(',') if x.strip().isdigit()]
            for index in indices:
                versions = current_results[0]['versions']
                if 1 <= index <= len(versions):
                    compiled_list.append(versions[index-1])
                    print(f"----> Saved: {versions[index-1][:30]}/n")
            
            

# Show final list
print("\n" + "="*40)
for i, item in enumerate(compiled_list, 1): print(f"{i}. {item}")