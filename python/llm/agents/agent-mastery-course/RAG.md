# Local Guide RAG Demo

This backend ships with an optional retrieval-augmented generation (RAG) flow that powers the `local_agent` suggestions. It stays dormant until you opt in.

## 1. Enable it

1. Copy `backend/.env.example` to `backend/.env` if you have not already.
2. Set the feature flag in your `.env`:
   ```bash
   ENABLE_RAG=1
   ```
3. Provide embeddings + chat credentials. With OpenAI, add:
   ```bash
   OPENAI_API_KEY=sk-...
   # Optional: override the embedder (defaults to text-embedding-3-small)
   # OPENAI_EMBED_MODEL=text-embedding-3-small
   ```
4. Restart the FastAPI server so the retriever loads the documents and embeddings.

Set `ENABLE_RAG=0` to fall back to the original heuristic local guide responses.

## 2. What it does

- On startup the app reads the curated snippets in `backend/data/local_guides.json` and, when the flag is on, indexes them into a LangChain in-memory vector store using the configured OpenAI embedding model.
- At runtime the `local_agent` runs a LangChain retriever (so you'll see dedicated embedding + retrieval spans in Arize) to fetch the top matches for the user's destination/interests and injects the numbered excerpts into the prompt context alongside inline citations.
- Retrieval metadata (hits, sources, similarity) is added to the agent trace so Arize/OpenInference dashboards show exactly what powered each recommendation.
- If embeddings are unavailable (e.g., missing key), the retriever automatically falls back to simple keyword scoring so the demo still runs.

That's all that is required for a live, inspectable RAG hop inside the LangGraph trip planner.
