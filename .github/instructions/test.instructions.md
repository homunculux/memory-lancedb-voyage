---
applyTo: "test/**/*.ts"
---

# Testing Conventions

- **Test runner** — uses the Node.js built-in test runner (`node:test`). Import `describe`, `it`, `before`, `after`, `mock` from `node:test`.
- **Assertions** — use `node:assert/strict` for all assertions (`assert.strictEqual`, `assert.deepStrictEqual`, `assert.throws`, etc.).
- **Mock external APIs** — always mock calls to external services (Voyage AI, OpenAI, Jina). Never make real API calls in unit tests.
- **File naming** — unit tests live in `test/unit.test.ts`.
- **Test isolation** — each test must be independent. No shared mutable state between tests. Set up and tear down within each test or `describe` block.
- **Descriptive names** — use `describe`/`it` blocks with clear, human-readable descriptions of the behavior under test.
