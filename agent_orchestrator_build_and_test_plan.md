# Agent Orchestrator Microservice: Comprehensive Build & Testing Plan

This document provides a step-by-step plan for designing, implementing, and robustly testing the collaborative agent-based microservice system for Retrieval-Augmented Generation (RAG) with both text and visual (VLM) backends.

---

## 1. System Overview

- **Goal:** Build a microservice (Agent Orchestrator) that coordinates multiple specialized agents to answer user queries using both text and visual RAG backends.
- **Key Features:**
  - Modular agent workflow (Dispatch, Visual, Textual, Verification, Response)
  - Parallel retrieval from text and VLM backends
  - Evidence-based answer synthesis and verification
  - API-driven, stateless, and scalable

---

## 2. Architecture & Technology Choices

- **Service Type:** FastAPI (recommended) or Flask microservice
- **Agent Orchestration:** CrewAI (CrewChain) or LangChain (CrewAI preferred for agent workflow)
- **Communication:** REST API (JSON)
- **Backends:**
  - Text RAG: Existing API (`/api/v1/chat/query`, `/api/v1/collections`, etc.)
  - VLM RAG: Existing API (`/api/query`, `/api/index/pdf`, etc.)
- **Database:** MongoDB (for collection summaries, if needed)
- **Testing:** pytest, unittest, requests-mock, httpx, pytest-asyncio

---

## 3. Implementation Plan

### 3.1. Project Setup
- Create a new directory for the agent orchestrator microservice.
- Initialize a Python virtual environment.
- Install dependencies:
  - FastAPI, uvicorn, pydantic
  - CrewAI or LangChain
  - requests, python-dotenv
  - pytest, httpx, requests-mock (for testing)
- Set up `.env` for config (API URLs, MongoDB URI, etc.)

### 3.2. Core Microservice Structure
- `main.py`: FastAPI app entrypoint
- `agents/`: Agent classes (Dispatch, Visual, Textual, Verification, Response)
- `services/`: API client wrappers for text and VLM backends
- `schemas.py`: Pydantic models for requests/responses
- `config.py`: Environment/config loading
- `tests/`: Unit and integration tests

### 3.3. Agent Implementation
- **Dispatch Agent:**
  - Receives user query and collection name
  - Fetches collection summary/context
  - Packages task for downstream agents
- **Visual Agent:**
  - Calls VLM RAG API in parallel
  - Returns visual evidence and answer
- **Textual Agent:**
  - Calls Text RAG API in parallel
  - Returns text evidence and answer
- **Verification & Synthesis Agent:**
  - Receives both answers/evidence
  - Checks for consistency, evidence grounding
  - Synthesizes a verified context with confidence scores
- **Response Agent:**
  - Formats the final answer with citations for frontend

### 3.4. Orchestrator Logic
- Expose a single `/query` endpoint
- On request:
  1. Dispatch Agent prepares context
  2. Visual and Textual Agents run in parallel (async)
  3. Verification Agent combines and validates results
  4. Response Agent formats and returns the answer

### 3.5. API Integration
- Use robust error handling and timeouts for backend API calls
- Log all agent actions and API responses for traceability

---

## 4. Testing Plan

### 4.1. Unit Testing
- **Agent Logic:**
  - Mock backend API responses
  - Test each agent in isolation (input → output)
- **API Clients:**
  - Mock requests to text/VLM APIs
  - Test error handling, retries, and edge cases
- **Config Loading:**
  - Test .env and config parsing

### 4.2. Integration Testing
- **End-to-End Query Flow:**
  - Use httpx/requests-mock to simulate backend APIs
  - Test full orchestrator workflow (from `/query` to final answer)
  - Validate parallel execution and fallback logic
- **Failure Scenarios:**
  - Simulate backend timeouts, partial failures
  - Ensure graceful degradation and fallback to available evidence

### 4.3. Manual & Exploratory Testing
- Connect to real text and VLM backends in a staging environment
- Test with diverse queries, document types, and collections
- Validate evidence grounding and answer formatting

---

## 5. Deployment & Operations
- Containerize with Docker (optional)
- Provide a sample `.env` and README for setup
- Add logging and health check endpoints
- Monitor API latency and error rates

---

## 6. References & Further Reading
- [CrewAI Documentation](https://docs.crewai.com/)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [pytest Docs](https://docs.pytest.org/)

---

**This plan ensures a robust, modular, and testable agent orchestrator microservice for collaborative RAG.**
