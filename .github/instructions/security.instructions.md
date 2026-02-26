---
applyTo: "**/*"
---

# Security Guidelines

- **No hardcoded secrets** — never hardcode API keys, tokens, or credentials in source code. All secrets must come from environment variables or the plugin config object.
- **No sensitive logging** — do not log API keys, authentication tokens, or full embedding vectors. Redact or omit sensitive values in log output.
- **Input validation** — validate and sanitize user input before passing it to external APIs (Voyage, OpenAI, Jina, LanceDB).
- **URL normalization** — always use `normalizeBaseUrl()` from `src/utils.ts` when handling base URLs for API endpoints. This prevents SSRF and ensures consistent URL formatting.
