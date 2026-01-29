def display_results(current_results):
    for entry in current_results:
        print(f"\nFOCUS: {entry.get('original')}")
        versions = entry.get('versions', [])
        for idx, version in enumerate(versions, 1):
            print(f"   [{idx}] {version}")

def get_refinement_text(action, current_results):
    try:
        idx = int(action[1:]) - 1
        versions = current_results[0]['versions']
        if 0 <= idx < len(versions):
            return versions[idx]
    except (ValueError, IndexError):
        pass
    return None

def process_save_action(action, current_results, compiled_list):
    try:
        indices = [int(x.strip()) for x in action.split(',') if x.strip().isdigit()]
        for index in indices:
            versions = current_results[0]['versions']
            if 1 <= index <= len(versions):
                saved_text = versions[index-1]
                compiled_list.append(saved_text)
                print(f"   -> Salvo localmente: {saved_text[:40]}...")
    except Exception:
        print("Seleção inválida.")