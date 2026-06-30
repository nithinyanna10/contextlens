# Code Style Rules — contextlens

> Claude reads this file before writing any new module in this repo.

## Python

- **Version:** 3.11+ — use match/case, `tomllib`, and `Self` type where appropriate
- **Async:** `async def` for all route handlers, agent methods, and service methods that do I/O
- **Type hints:** mandatory on all function signatures — no bare `Any` without a comment
- **Imports:** stdlib → third-party → local, separated by blank lines. No wildcard imports.
- **Config:** always via `app/config.py` Settings — never `os.environ.get()` inline
- **Logging:** `from loguru import logger` — no `print()`, no `logging.getLogger()`
- **Pydantic:** v2 syntax — `model_config = ConfigDict(...)`, not `class Config`
- **FastAPI:** dependency injection via `Depends()` — no global mutable state

## Naming

- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions / variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private helpers: `_leading_underscore`

## Structure Rules

- One class or one logical group of functions per file
- Services in `services/` — no LLM calls, no agent calls directly from routes
- Agents in `agents/` — no DB calls, no direct HTTP calls to external APIs
- Tools in `agents/tools/` — stateless, single async `run()` method
- Prompts in `prompts/` — never inline in service or agent files

## Formatting

- Black-compatible line length: 100 chars
- Docstrings: one-line for simple functions, Google style for complex ones
- Comments explain *why*, not *what*

## Frontend (Next.js / TypeScript)

- TypeScript strict mode — no `any`
- Components: functional with hooks — no class components
- File naming: `PascalCase.tsx` for components, `camelCase.ts` for utilities
- API calls only through `src/lib/api.ts` — never `fetch()` inline in components
- Tailwind for styling — no inline `style={{}}` props
