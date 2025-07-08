from langchain_core.language_models.llms import LLM
import requests
from typing import Optional, List, Any

class DeepSeekLLM(LLM):
    model: str = "deepseek-chat"
    api_key: str = ""
    api_url: str = "https://api.deepseek.com/v1/chat/completions"

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] 