from typing import Any
import requests

class ModelNotFoundError(RuntimeError):
    """The requested model is not installed in Ollama."""


class OllamaClient:
    def __init__(
            self,
            base_url: str,
            model: str,
            timeout: int,
            num_ctx: int = 8192,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx

    def generate(
            self,
            prompt: str,
            system: str = "",
            json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }

        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )

            if getattr(response, "status_code", 200) == 404:
                raise ModelNotFoundError(
                    f"Ollama model '{self.model}' was not found. "
                    f"Install it with: ollama pull {self.model}"
                )

            response.raise_for_status()

            return str(
                response.json().get("response", "")
            ).strip()

        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama connection failed: {exc}") from exc

