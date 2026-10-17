import pytest
import requests

import app.llm as llm_module
from app.llm import ModelNotFoundError, OllamaClient


class FakeResponse:
    def __init__(self, text="  hello  "):
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": self.text}


def with_fake_post(fake):
    original = llm_module.requests.post
    llm_module.requests.post = fake
    return original


def test_generate_sends_temperature_zero_and_explicit_context_window():
    seen = {}

    def fake_post(url, json, timeout):
        seen.update(url=url, payload=json, timeout=timeout)
        return FakeResponse()

    original = with_fake_post(fake_post)

    try:
        client = OllamaClient("http://host:11434/", "qwen", 30, 4096)
        result = client.generate("question", system="sys", json_mode=True)
    finally:
        llm_module.requests.post = original

    assert result == "hello"
    assert seen["url"] == "http://host:11434/api/generate"
    assert seen["payload"]["options"] == {"temperature": 0, "num_ctx": 4096}
    assert seen["payload"]["format"] == "json"
    assert seen["payload"]["model"] == "qwen"
    assert seen["timeout"] == 30


def test_json_format_is_only_requested_in_json_mode():
    seen = {}

    def fake_post(url, json, timeout):
        seen.update(payload=json)
        return FakeResponse()

    original = with_fake_post(fake_post)

    try:
        OllamaClient("http://h", "m", 5).generate("q")
    finally:
        llm_module.requests.post = original

    assert "format" not in seen["payload"]
    assert seen["payload"]["options"]["num_ctx"] == 8192  # default


def test_connection_errors_become_runtime_errors():
    def fake_post(url, json, timeout):
        raise requests.ConnectionError("refused")

    original = with_fake_post(fake_post)

    try:
        with pytest.raises(RuntimeError):
            OllamaClient("http://h", "m", 5).generate("q")
    finally:
        llm_module.requests.post = original


def test_missing_model_raises_a_clear_error_with_the_pull_command():
    class NotFound(FakeResponse):
        status_code = 404

    original = with_fake_post(lambda url, json, timeout: NotFound())

    try:
        with pytest.raises(ModelNotFoundError) as error:
            OllamaClient("http://h", "qwen2.5:14b", 5).generate("q")
    finally:
        llm_module.requests.post = original

    assert "ollama pull qwen2.5:14b" in str(error.value)
    assert isinstance(error.value, RuntimeError)  # the API maps it to a 502
