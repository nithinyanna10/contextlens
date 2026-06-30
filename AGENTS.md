# AGENTS.md — contextlens

> Agent architecture rules for this repo. Read alongside CLAUDE.md at the start of every session.

## Agent Roles and Boundaries

Each agent has exactly one responsibility. If you're writing code that crosses these lines, stop and ask.

### `agents/document_grader.py`
**Role:** Score retrieved documents for relevance to the query.
- Input: `(query: str, documents: list[Document]) → list[GradedDocument]`
- Does NOT retrieve documents — that's `app/components/hybrid_retriever.py`
- Does NOT route to other sources — that's `adaptive_router.py`
- Returns a relevance score (0.0–1.0) and a brief reason string per document

### `agents/query_decomposer.py`
**Role:** Break complex multi-part queries into atomic sub-questions.
- Input: `(query: str) → list[str]` (list of sub-questions)
- Does NOT retrieve anything
- Does NOT decide which tool to use
- Single-hop queries should pass through unchanged (don't over-decompose)

### `agents/adaptive_router.py`
**Role:** Decide which retrieval tool(s) to call for a given query.
- Input: `(query: str, context: RouterContext) → list[ToolCall]`
- Does NOT execute the tools — it returns a list of `ToolCall` objects
- Does NOT grade documents
- Available tools registered in `agents/tools/` — router reads the tool registry

### `agents/tools/`
**Role:** Pluggable, stateless tool definitions.
- `vector_search.py` — ChromaDB / pgvector similarity search
- `web_search.py` — web retrieval (Tavily / SerpAPI / etc.)
- `code_search.py` — code-specific retrieval (GitHub, local codebase)
- Each tool: `async def run(query: str, **kwargs) → list[Document]`
- Tools are stateless — no memory, no routing decisions

---

## Data Contracts

All agents use these shared Pydantic models (define in `app/models.py`):

```python
class Document(BaseModel):
    id: str
    content: str
    source: str
    metadata: dict[str, Any] = {}

class GradedDocument(Document):
    relevance_score: float  # 0.0–1.0
    reason: str

class ToolCall(BaseModel):
    tool_name: str          # matches key in tool registry
    query: str
    kwargs: dict[str, Any] = {}

class RouterContext(BaseModel):
    query: str
    conversation_history: list[dict] = []
    available_tools: list[str] = []
```

---

## Trace ID Rule

Every agent method signature must accept `trace_id: str` as a keyword argument:

```python
async def grade(self, query: str, documents: list[Document], *, trace_id: str) -> list[GradedDocument]:
    ...
```

Log the `trace_id` at entry and exit of each agent call using `observability/tracer.py`.

---

## Error Handling

- Agents raise — services catch
- Use specific exception types, not bare `Exception`
- On retrieval failure: raise `RetrievalError` (defined in `app/models.py`)
- On grading failure: log the error with `trace_id` and return the document with `relevance_score=0.0`
- On routing failure: fall back to `vector_search` as the default tool

---

## Evaluation

When adding a new agent behavior or changing routing logic:
1. Add a representative case to `evaluation/golden_dataset.json`
2. Run `python evaluation/offline_eval.py` and confirm the score does not regress
3. Commit the eval results in `evaluation/eval_results/` alongside the code change
