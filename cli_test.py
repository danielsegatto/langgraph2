import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from graph import create_app

# 1. Environment and Configuration
load_dotenv()

llm = ChatGroq(
    temperature=0,
    model_name="openai/gpt-oss-120b", 
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_kwargs={"response_format": {"type": "json_object"}} 
)

# 2. Build the Application
app = create_app(llm)

# 3. Execution Logic
compiled_list = []

while True:
    user_input = input("\n =>: ").strip()

    if user_input.lower() in ["q", "exit", "sair"]:
        print("\n" + "="*40)
        for i, item in enumerate(compiled_list, 1):
            print(f"{item}")
        print("="*40)
        break

    # Run the graph in Translation Mode
    result = app.invoke({"original_text": user_input, "mode": "translate"})
    print(result['final_results'])
    current_results = result['final_results']

    while True:
        for entry in current_results:
            # Displays the current focus (original sentence) and its 3 variations
            versions = entry.get('versions', [])
            for idx, version in enumerate(versions, 1):
                print(f"   [{idx}] {version}")
        
        print("\nOPTIONS: [1,2.. to Save] | [r1, r2.. to Refine] | [Enter for New Text]")
        action = input(" Action: ").strip().lower()

        if not action: 
            break

        # REFINEMENT: Re-runs the graph in Refine Mode
        if action.startswith('r'):
            try:
                idx = int(action[1:]) - 1
                selected = current_results[0]['versions'][idx]

                refine_res = app.invoke({"refinement_text": selected, "mode": "refine"})
                
                # Updates the loop with the 3 fresh refinements
                current_results = refine_res['final_results']
            except Exception:
                print("Invalid refinement choice.")
        
        # SAVING: Adds selected versions to the final list
        elif action[0].isdigit():
            indices = [int(x.strip()) for x in action.split(',') if x.strip().isdigit()]
            for index in indices:
                versions = current_results[0]['versions']
                if 1 <= index <= len(versions):
                    saved_text = versions[index-1]
                    compiled_list.append(saved_text)
                    print(f"   -> Saved: {saved_text[:40]}...")