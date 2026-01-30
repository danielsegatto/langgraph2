import requests
import ui  

BASE_URL = "https://friendly-lamp-rr9j746jxjqfvjx-8000.app.github.dev"

def translate_api(text):
    try:
        response = requests.post(f"{BASE_URL}/translate", json={"text": text})
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return []

def refine_api(sentence):
    try:
        response = requests.post(f"{BASE_URL}/refine", json={"selected_sentence": sentence})
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return []

def interaction_loop(current_results, compiled_list):
    while current_results:
        ui.display_results(current_results)
        
        print("\nOPTIONS: [1,2.. to Save] | [r1, r2.. to Refine] | [Enter for New Text]")
        action = input(" Action: ").strip().lower()

        if not action: 
            break

        if action.startswith('r'):
            text_to_refine = ui.get_refinement_text(action, current_results)
            if text_to_refine:
                current_results = refine_api(text_to_refine)
            else:
                print("(use r1, r2...).")
        
        elif action[0].isdigit():
            ui.process_save_action(action, current_results, compiled_list)

# --- Entry Point ---
if __name__ == "__main__":
    saved_sentences = []
    print("\n>>> API Client Ready. Type 'q' to exit. <<<")
    
    while True:
        user_input = input("\n =>: ").strip()
        if user_input.lower() in ["q", "exit", "sair"]:
            break
        
        results = translate_api(user_input)
        interaction_loop(results, saved_sentences)
    
ui.display_final_list(saved_sentences)
