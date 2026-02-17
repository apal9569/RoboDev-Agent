import json
import os
from pathlib import Path
from google import genai

CONFIG_PATH = Path.home() / ".robodev" / "llm_config.json"


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


class GeminiClient:
    def __init__(self, api_key: str = None, model: str = None):
        config = _load_config()
        gemini_cfg = config.get("gemini", {})
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or gemini_cfg.get("api_key")
        )
        self.model = model or gemini_cfg.get("model", "gemini-2.5-flash")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY env var:\n"
                '  export GEMINI_API_KEY="YOUR_KEY"  # add to ~/.bashrc'
            )
        self.client = genai.Client(api_key=self.api_key)

    def chat(self, prompt, timeout: int = 300, task: str = None) -> str:
        if isinstance(prompt, list):
            parts = []
            for msg in prompt:
                parts.append(f"{msg['role']}: {msg['content']}")
            text = "\n\n".join(parts)
        else:
            text = prompt

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=text,
            )
            print(f"🤖 Model: {self.model} (Gemini) | Task: {task or 'default'}")
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}") from e