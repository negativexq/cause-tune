"""Provider-neutral and explicitly opt-in provider clients for Experiment 03C."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol

from .rendering import RenderPlan, build_render_prompt


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ProviderRawResponse:
    content: str
    usage: ProviderUsage = ProviderUsage()
    request_id: str | None = None

    def content_hash(self) -> str:
        return sha256(self.content.encode("utf-8")).hexdigest()


class ProviderError(RuntimeError):
    """A provider failure that may be retried by the pilot orchestrator."""

    def __init__(self, message: str, *, retryable: bool = True, category: str = "provider_error") -> None:
        self.retryable = retryable
        self.category = category
        super().__init__(message)


class ProviderClient(Protocol):
    adapter_version: str
    provider_name: str
    model_identifier: str

    def generate(self, prompt: str) -> ProviderRawResponse:
        """Generate one structured response for a rendered prompt."""


class FakeTelemetryProvider:
    """Deterministic local provider used by tests and the default dry-run."""

    adapter_version = "fake-provider-03c-v1"
    provider_name = "fake"

    def __init__(self, model_identifier: str = "fake-telemetry-renderer-v1") -> None:
        self.model_identifier = model_identifier
        self.request_count = 0

    def generate(self, prompt: str) -> ProviderRawResponse:
        self.request_count += 1
        marker = '"render_contract": "03c-telemetry-surface-v1"'
        start = prompt.find("{")
        if start < 0 or marker not in prompt:
            raise ProviderError("fake provider received a prompt without the render contract", retryable=False)
        provider_payload = json.loads(prompt[start:])
        family = provider_payload["renderer_family"]
        runtime = provider_payload["runtime_id"]
        style = provider_payload["style"]
        observations: list[dict[str, Any]] = []
        for index, entry in enumerate(provider_payload["evidence"], start=1):
            if entry["may_be_omitted"]:
                continue
            attributes = {item["name"]: item["value"] for item in entry["required_attributes"]}
            if family == "concise_ops_v1":
                prefix = "ops signal"
            elif family == "verbose_enterprise_v1":
                prefix = "enterprise observation"
            else:
                prefix = "runtime-native signal"
            attribute_text = "; ".join(f"{key}={value}" for key, value in sorted(attributes.items()))
            message = f"{prefix} for {entry['source_component']}: {attribute_text or style}"
            if family == "ecosystem_native_v1":
                message += f" ({runtime.replace('_', '/')})"
            digest = sha256(f"{prompt}|{entry['evidence_id']}".encode("utf-8")).hexdigest()
            observations.append(
                {
                    "evidence_id": entry["evidence_id"],
                    "kind": entry["permitted_surface_types"][0],
                    "component_id": entry["source_component"],
                    "timestamp_offset_seconds": int(digest[:4], 16) % 3600 + index,
                    "text": message,
                    "value": attributes or None,
                }
            )
        content = json.dumps({"observations": observations}, ensure_ascii=False, sort_keys=True)
        token_count = len(content.split())
        return ProviderRawResponse(content, ProviderUsage(len(prompt.split()), token_count, len(prompt.split()) + token_count))


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Configuration for a generic OpenAI-compatible chat-completions endpoint."""

    model_identifier: str
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    default_base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 60.0
    max_retries: int = 2
    temperature: float = 0.0
    max_output_tokens: int = 2048

    def __post_init__(self) -> None:
        if not self.model_identifier.strip():
            raise ValueError("live provider model identifier is required")
        if self.timeout_seconds <= 0 or self.max_retries < 0 or self.max_output_tokens <= 0:
            raise ValueError("invalid live provider bounds")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")


class OpenAICompatibleProvider:
    """One opt-in stdlib adapter for OpenAI-compatible providers.

    The adapter never persists credentials.  Network retries are bounded here;
    semantic retries remain the responsibility of the pilot runner.
    """

    adapter_version = "openai-compatible-provider-03c-v1"
    provider_name = "openai_compatible"

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config
        self.model_identifier = config.model_identifier

    def generate(self, prompt: str) -> ProviderRawResponse:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ProviderError(f"missing provider credential environment variable: {self.config.api_key_env}", retryable=False, category="credential_error")
        base_url = os.environ.get(self.config.base_url_env, self.config.default_base_url).rstrip("/")
        payload = {
            "model": self.config.model_identifier,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ProviderError("provider returned non-string message content", retryable=False, category="provider_protocol_error")
                usage_payload = body.get("usage") or {}
                usage = ProviderUsage(
                    usage_payload.get("prompt_tokens"),
                    usage_payload.get("completion_tokens"),
                    usage_payload.get("total_tokens"),
                )
                return ProviderRawResponse(content, usage, body.get("id"))
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt >= self.config.max_retries:
                    raise ProviderError(f"provider HTTP error {exc.code}", retryable=retryable, category="provider_http_error") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
                if attempt >= self.config.max_retries:
                    raise ProviderError(f"provider request failed: {type(exc).__name__}", category="provider_network_error") from exc
            if attempt < self.config.max_retries:
                time.sleep(min(2**attempt, 4))
        raise ProviderError("provider request exhausted its retry budget", category="provider_network_error")


def prompt_for_plan(plan: RenderPlan) -> str:
    """Keep prompt construction in the rendering boundary, not the adapter."""

    return build_render_prompt(plan)
