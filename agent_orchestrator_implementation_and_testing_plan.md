# Agent Orchestrator Microservice: Implementation & Testing Plan

This document provides a comprehensive, actionable plan for building, integrating, and testing the new agent-based orchestrator microservice for your collaborative RAG system. It is based on the workflow and requirements described in `new_workflow.md`.

---

## 1. Overview

The Agent Orchestrator is a microservice that coordinates multiple specialized agents (Dispatch, Visual, Textual, Verification, Response) to answer user queries by leveraging both text-based and VLM-based RAG backends. It exposes a single API endpoint for the frontend, manages collection context, and synthesizes multi-modal answers with verification and citations.

---

## 2. Microservice Architecture

### **Agents & Responsibilities**
- **Dispatch Agent:**
  - Receives user query and collection name.
  - Fetches collection summary/context (via summary API or MongoDB).
  - Packages the query and context into a Task Package.
- **Visual Agent:**
  - Calls VLM RAG API (`POST /api/query`).
  - Retrieves top-k visual evidence and generates a visual answer.
- **Textual Agent:**
  - Calls Text RAG API (`POST /api/v1/chat/query`).
  - Retrieves top-k text chunks and generates a textual answer.
- **Verification & Synthesis Agent:**
  - Cross-examines visual and textual answers for consistency and evidence grounding.
  - Synthesizes a verified context block with confidence scores and combined evidence.
- **Response Generation Agent:**
  - Formats the final answer, integrates citations, and prepares it for display.

### **Service Design**
- **Framework:** FastAPI (recommended for async, easy API definition, and testing)
- **Deployment:** Standalone service (Dockerized or virtualenv)
- **Config:** Reads backend URLs, MongoDB, and thresholds from `.env` or config file
- **Logging/Monitoring:** Add structured logging and error handling for all agent steps

---

## 3. API Design

### **Main Endpoint**
- `POST /api/agent_query`
  - **Input:** `{ "query": <str>, "collection": <str> }`
  - **Output:** `{ "answer": <str>, "citations": [...], "evidence": {...}, "confidence": <float> }`

### **(Optional) Collection Summary Endpoint**
- `GET /api/collection_summary?collection=<str>`
  - Returns summary/context for a collection (if not already present in text/VLM backends)

---

## 4. Implementation Steps

1. **Project Setup**
   - Create new FastAPI project (e.g., `agent_orchestrator/`)
   - Add dependencies: `fastapi`, `httpx`, `pydantic`, `python-dotenv`, `pytest`, etc.
2. **Config & Environment**
   - Load backend URLs, MongoDB URI, and other settings from `.env`
3. **Agent Classes/Modules**
   - Implement each agent as a class or function (Dispatch, Visual, Textual, Verification, Response)
   - Use async HTTP calls for parallel retrieval (httpx.AsyncClient)
4. **Orchestration Logic**
   - Main endpoint receives query, invokes Dispatch Agent
   - In parallel, Visual and Textual Agents call their respective backends
   - Verification Agent cross-examines and synthesizes
   - Response Agent formats and returns the answer
5. **Error Handling & Logging**
   - Add robust error handling for all API calls and agent steps
   - Log all actions, errors, and timings for monitoring
6. **Testing (see below)**
7. **Dockerization/Deployment**
   - Add Dockerfile and deployment scripts if needed

---

## 5. Testing Plan

### **Unit Tests**
- Test each agent in isolation with mocked backend responses
- Test error cases (backend unavailable, malformed responses, etc.)
- Test confidence scoring and fallback logic in Verification Agent

### **Integration Tests**
- Use dummy/mock text and VLM backends to simulate end-to-end flow
- Test main endpoint with various queries and collections
- Validate output structure, evidence, and confidence scores

### **Manual/Frontend Testing**
- Connect orchestrator to the Streamlit frontend
- Run real queries and verify correct, timely, and consistent answers
- Check evidence/citations and error handling in UI

### **Test Automation**
- Use `pytest` for all tests
- Add CI workflow for automated testing on push/PR

---

## 6. Example Test Cases

- Query with only text evidence (VLM returns nothing)
- Query with only visual evidence (text returns nothing)
- Both agents return answers, but with conflicting evidence
- Both agents agree and provide strong, consistent evidence
- Backend API returns error or times out
- Collection summary missing or malformed

---

## 7. Monitoring & Logging (Optional, but recommended)

- Log all agent actions, timings, and errors
- Add Prometheus/Grafana metrics for request counts, latencies, error rates
- Alert on backend failures or orchestrator errors

---

## 8. References
- See `new_workflow.md` for full workflow and API details
- See `agent_orchestrator_build_and_test_plan.md` for additional implementation notes

---

**This plan provides a clear, actionable roadmap for building, integrating, and testing your new agent orchestrator microservice.**
