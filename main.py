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

class AgentState(TypedDict):
    original_text: str
    sentences: List[str]
    final_results: List[dict]

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
    if isinstance(content, list):
        content = content[0].get('text', '')
    
    sentence_list = [s.strip() for s in content.split('\n') if s.strip()]
    return {"sentences": sentence_list}

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
        translated_data = json.loads(raw_content)
        return {"final_results": translated_data}
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {"final_results": []}

workflow = StateGraph(AgentState)

workflow.add_node("splitter", sentence_splitter_node)
workflow.add_node("translator", triple_translator_node)

workflow.add_edge(START, "splitter")
workflow.add_edge("splitter", "translator")
workflow.add_edge("translator", END)

app = workflow.compile()

compiled_list = []

while True:
    user_input = input("\n =>: ")

    if user_input.lower() in ["q", "exit"]:
        print("="*40)
        for i, item in enumerate(compiled_list, 1):
            print(f"{i}. {item}")
        print("\nClosing...")
        break

    result = app.invoke({"original_text": user_input})

    for entry in result['final_results']:
        versions = entry.get('versions', [])
        
        for idx, version in enumerate(versions, 1):
            print(f"[{idx}] {version}")
        
        choice = input("\n Select versions to save or Enter to skip: ")
        
        if choice.strip():
            indices = [int(x.strip()) for x in choice.split(',') if x.strip().isdigit()]
            
            for index in indices:
                if 1 <= index <= len(versions):
                    selected_text = versions[index-1]
                    compiled_list.append(selected_text)
                    print(f"-> Added to list: {selected_text[:40]}...")
        
        print("-" * 20)