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
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_kwargs={"response_format": {"type": "json_object"}} 
)

class AgentState(TypedDict):
    original_text: str
    mode: str           
    refinement_text: str 
    sentences: List[str]
    final_results: List[dict]

# --- NODO 1: SENTENCE SPLITTER ---
def sentence_splitter_node(state: AgentState):
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

# --- NODO 2: TRIPLE TRANSLATOR ---
def triple_translator_node(state: AgentState):
    system_instructions = (
        "You are a professional translator and expert polyglot. For each Portuguese sentence provided, create 3 English versions. "
        "Choose the 3 best stylistic variations that fit the context of the sentence. "
        "Respond ONLY with a JSON object. No numbers inside strings. "
        "Example Output: {\"results\": [{\"original\": \"...\", \"versions\": [\"V1\", \"V2\", \"V3\"]}]}"
    )
    input_list = "\n".join(state['sentences'])
    response = llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content=f"Translate: {input_list}")
    ])
    data = json.loads(response.content)
    return {"final_results": data.get("results", [])}

# --- NODO 3: STYLE REFINER ---
def style_refiner_node(state: AgentState):
    system_instructions = (
        "You are a writing coach. The user has selected a specific English sentence and wants more variations of it. "
        "Generate 3 NEW natural English variations. Do NOT add numbers."
        "Respond ONLY with a JSON object. "
        "Example Output: {\"results\": [{\"original\": \"...\", \"versions\": [\"Alt 1\", \"Alt 2\", \"Alt 3\"]}]}"
    )
    response = llm.invoke([
        SystemMessage(content=system_instructions),
        HumanMessage(content=f"Refine: {state['refinement_text']}")
    ])
    data = json.loads(response.content)
    return {"final_results": data.get("results", [])}
def router_logic(state: AgentState):
    return "refiner" if state.get("mode") == "refine" else "splitter"
workflow = StateGraph(AgentState)

workflow.add_node("splitter", sentence_splitter_node)
workflow.add_node("translator", triple_translator_node)
workflow.add_node("refiner", style_refiner_node)
workflow.add_conditional_edges(START, router_logic, {"splitter": "splitter", "refiner": "refiner"})
workflow.add_edge("splitter", "translator")
workflow.add_edge("translator", END)
workflow.add_edge("refiner", END)

app = workflow.compile()

compiled_list = []

while True:
    user_input = input("\n =>: ").strip()

    if user_input.lower() in ["q", "exit", "sair"]:
        print("="*40)
        for i, item in enumerate(compiled_list, 1):
            print(f"{item}")
        break

    result = app.invoke({"original_text": user_input, "mode": "translate"})
    current_results = result['final_results']

    while True:
        for entry in current_results:
            versions = entry.get('versions', [])
            for idx, version in enumerate(versions, 1):
                print(f"   [{idx}] {version}")
        
        print("\nOPTIONS: [1,2.. to Save] | [r1, r2.. to Refine] | [Enter for New Text]")
        action = input(" Ação: ").strip().lower()

        if not action: break

        if action.startswith('r'):
            try:
                idx = int(action[1:]) - 1
                selected = current_results[0]['versions'][idx]
                refine_res = app.invoke({"refinement_text": selected, "mode": "refine"})
                current_results = refine_res['final_results']
            except:
                print("Escolha de refinamento inválida.")
        
        elif action[0].isdigit():
            indices = [int(x.strip()) for x in action.split(',') if x.strip().isdigit()]
            for index in indices:
                versions = current_results[0]['versions']
                if 1 <= index <= len(versions):
                    compiled_list.append(versions[index-1])
                    print(f"   -> Salvo: {versions[index-1][:40]}...")
