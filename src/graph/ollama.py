import subprocess


def call_ollama(prompt: str, model: str = "qwen2.5:3b") -> str:
    r = subprocess.run(["ollama", "run", model, prompt], capture_output=True, text=True)
    return (r.stdout or "").strip()
    