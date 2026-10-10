"""Select an installed generation model from the local Ollama server."""
import requests

OLLAMA_URL = "http://127.0.0.1:11434"


def select_model():
    """Return the exact installed model name, or explain why selection cannot start."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise ValueError(f"Cannot list local Ollama models. Check that Ollama is running. Details: {exc}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ValueError("Ollama returned an invalid model list. No run was started.")
    names = []
    for entry in data["models"]:
        if not isinstance(entry, dict):
            raise ValueError("Ollama returned an invalid model list. No run was started.")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Ollama returned an invalid model name. No run was started.")
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("No models are installed in local Ollama. Install a model with 'ollama pull <model-name>', then try again.")

    print("\nInstalled Ollama models:")
    for number, name in enumerate(names, 1):
        print(f"{number}. {name}")
    while True:
        choice = input(f"Choose a model (1-{len(names)}): ").strip()
        if choice.isdecimal() and 1 <= int(choice) <= len(names):
            selected = names[int(choice) - 1]
            print(f"Selected model: {selected}")
            return selected
        print(f"Invalid selection. Enter a number from 1 to {len(names)}.")
