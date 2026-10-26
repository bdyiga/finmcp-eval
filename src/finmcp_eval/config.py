"""Settings read from environment variables. Secrets are never logged or echoed."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ConfigError(RuntimeError):
    """A required setting is missing or invalid. Messages never include secret values."""


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, default)
    return value.strip() if isinstance(value, str) else value


def _float(name: str, default: float, lo: float, hi: float) -> float:
    raw = _env(name)
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number") from None
    if not lo <= value <= hi:
        raise ConfigError(f"{name} must be between {lo} and {hi}")
    return value


def _int(name: str, default: int, lo: int, hi: int) -> int:
    return int(_float(name, default, lo, hi))


# --- SEC --------------------------------------------------------------------


def sec_user_agent() -> str:
    """Build the User-Agent SEC requires: a name and a contact email."""
    name = _env("SEC_CONTACT_NAME") or ""
    email = _env("SEC_CONTACT_EMAIL") or ""
    if not name or not email:
        raise ConfigError(
            "SEC_CONTACT_NAME and SEC_CONTACT_EMAIL must be set (SEC requires a contact in the User-Agent)"
        )
    if not _EMAIL_RE.fullmatch(email):
        raise ConfigError("SEC_CONTACT_EMAIL does not look like an email address")
    agent = f"{name} {email}"
    if len(agent) > 200 or any(ch in agent for ch in "\r\n\t") or not agent.isprintable():
        raise ConfigError("SEC_CONTACT_NAME/SEC_CONTACT_EMAIL contain invalid characters or are too long")
    return agent


def cache_dir() -> Path:
    return Path(_env("FINMCP_CACHE_DIR") or "cache").resolve()


def cache_ttl_days() -> float:
    return _float("FINMCP_CACHE_TTL_DAYS", 30.0, 0.0, 3650.0)


# Variables forwarded to the MCP server subprocess. The MCP SDK passes only a
# small safe set of variables by default, so the model API key never reaches it.
SERVER_ENV_VARS = ("SEC_CONTACT_NAME", "SEC_CONTACT_EMAIL", "FINMCP_CACHE_DIR", "FINMCP_CACHE_TTL_DAYS")


def server_env() -> dict[str, str]:
    env = {k: os.environ[k] for k in SERVER_ENV_VARS if os.environ.get(k)}
    env["FINMCP_CACHE_DIR"] = str(cache_dir())
    return env


# --- Models -----------------------------------------------------------------


@dataclass(frozen=True)
class GenerationSettings:
    """Identical in both evaluation modes."""

    temperature: float
    max_output_tokens: int
    seed: int


def generation_settings() -> GenerationSettings:
    return GenerationSettings(
        temperature=_float("MODEL_TEMPERATURE", 0.0, 0.0, 2.0),
        max_output_tokens=_int("MODEL_MAX_OUTPUT_TOKENS", 2048, 64, 65536),
        seed=_int("MODEL_SEED", 1234, 0, 2**31 - 1),
    )


def gemini_api_key() -> str:
    key = _env("GEMINI_API_KEY") or ""
    if not key:
        raise ConfigError("GEMINI_API_KEY is not set (see .env.example)")
    return key


def gemini_model() -> str:
    return _model_name("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def gemini_min_interval() -> float:
    """Seconds between Gemini requests, to stay under free-tier per-minute limits."""
    return _float("GEMINI_MIN_INTERVAL_SECONDS", 6.0, 0.0, 600.0)


def ollama_model() -> str:
    return _model_name("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def ollama_host() -> str:
    """Ollama must be local: only loopback hosts are accepted."""
    host = (_env("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    parsed = urlparse(host)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigError("OLLAMA_HOST must be a local http URL such as http://127.0.0.1:11434")
    if parsed.path not in ("", "/") or parsed.query or parsed.username:
        raise ConfigError("OLLAMA_HOST must not contain a path, query, or credentials")
    return host


def _model_name(var: str, default: str) -> str:
    name = _env(var) or default
    if not re.fullmatch(r"[A-Za-z0-9._:/\-]{1,100}", name):
        raise ConfigError(f"{var} contains unexpected characters")
    return name
