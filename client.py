import requests

# Configuration
BASE_URL = "https://friendly-lamp-rr9j746jxjqfvjx-8000.app.github.dev/"

def translate_api(text):
    """Calls the /translate endpoint."""
    try:
        response = requests.post(f"{BASE_URL}/translate", json={"text": text})
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return []

def refine_api(sentence):
    """Calls the /refine endpoint."""
    try:
        response = requests.post(f"{BASE_URL}/refine", json={"selected_sentence": sentence})
        response.raise_for_status()
        return response.json().get("results", [])
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return []

def display_results(current_results):
    """Handles only the visual display of the translations."""
    for entry in current_results:
        print(f"\nFOCUS: {entry.get('original')}")
        versions = entry.get('versions', [])
        for idx, version in enumerate(versions, 1):
            print(f"   [{idx}] {version}")

def handle_refinement(action, current_results):
    """Processes the refinement logic and returns the new results."""
    try:
        idx = int(action[1:]) - 1
        # Assumes working with the first entry's versions
        selected = current_results[0]['versions'][idx]
        print(f"--- Refining via API: '{selected[:30]}...' ---")
        return refine_api(selected)
    except (ValueError, IndexError):
        print("Invalid refinement choice. Use 'r1', 'r2', etc.")
        return current_results

def handle_saving(action, current_results, compiled_list):
    """Processes the saving logic and updates the compiled list."""
    try:
        indices = [int(x.strip()) for x in action.split(',') if x.strip().isdigit()]
        for index in indices:
            versions = current_results[0]['versions']
            if 1 <= index <= len(versions):
                saved_text = versions[index-1]
                compiled_list.append(saved_text)
                print(f"   -> Saved locally: {saved_text[:40]}...")
    except Exception:
        print("Invalid selection.")

def interaction_loop(current_results, compiled_list):
    """The modularized version of your refinement/save loop."""
    while current_results:
        display_results(current_results)
        
        print("\nOPTIONS: [1,2.. to Save] | [r1, r2.. to Refine] | [Enter for New Text]")
        action = input(" Action: ").strip().lower()

        if not action: 
            break

        if action.startswith('r'):
            current_results = handle_refinement(action, current_results)
        
        elif action and action[0].isdigit():
            handle_saving(action, current_results, compiled_list)

# Main entry point
if __name__ == "__main__":
    saved_sentences = []
    print("\n>>> API Client Ready. Type 'q' to exit. <<<")
    
    while True:
        user_input = input("\n =>: ").strip()
        if user_input.lower() in ["q", "exit"]:
            break
        
        results = translate_api(user_input)
        interaction_loop(results, saved_sentences)
    
    print("\nFINAL SAVED LIST:", saved_sentences)