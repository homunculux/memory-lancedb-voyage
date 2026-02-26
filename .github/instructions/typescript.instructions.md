---
applyTo: "**/*.ts"
---

# TypeScript Conventions

- **Strict mode** — all code must compile under TypeScript strict mode (`tsc --noEmit`).
- **Explicit types** — always provide explicit type annotations for function parameters and return types.
- **`interface` over `type`** — prefer `interface` for object shapes; use `type` for unions, intersections, and mapped types.
- **No `any`** — never use `any`. Use `unknown` when the type is uncertain and narrow with type guards.
- **Error handling** — always handle promise rejections. Use `try/catch` with specific error types rather than generic catches.
- **Imports** — use relative paths within `src/` (e.g., `./utils`, `./config`). Prefer named exports over default exports.
