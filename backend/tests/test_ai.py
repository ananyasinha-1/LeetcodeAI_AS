import re

import pytest
from dotenv import load_dotenv

from ai.provider_manager import ProviderManager

load_dotenv()


# ---------------------------
# Provider tests
# ---------------------------

@pytest.mark.parametrize(
    "provider_name",
    ["gemini", "openai", "perplexity"],
)
def test_provider_generation(monkeypatch, provider_name):

    monkeypatch.setenv("AI_PROVIDER", provider_name)

    manager = ProviderManager()

    response = manager.generate("Write one short sentence about Python.")

    print(f"\n[{provider_name}] Response: {response}")

    assert isinstance(response, str)
    assert response.strip() != ""
    assert len(response.strip()) > 10

    assert re.search(r"\bpython\b", response, re.IGNORECASE)

    error_patterns = [
        r"invalid api key",
        r"unauthorized",
        r"authentication",
        r"rate limit",
        r"quota",
        r"forbidden",
        r"error",
        r"failed",
    ]

    for pattern in error_patterns:
        assert not re.search(pattern, response, re.IGNORECASE)
