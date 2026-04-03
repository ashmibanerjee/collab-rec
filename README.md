# Collab-Rec: An LLM-based Agentic Framework for Tourism Recommendations

**Authors:** Ashmi Banerjee, Fitri Nur Aisyah, Adithi Satish, Wolfgang Woerndl, Yashar Deldjoo

**Paper URL:** [https://arxiv.org/pdf/2508.15030](https://arxiv.org/pdf/2508.15030)

## Abstract
We propose Collab-REC, a multi-agent framework designed to counteract popularity bias and enhance diversity in tourism recommendations. Three LLM-based agents - **Personalization**, **Popularity**, and **Sustainability** - generate city suggestions from complementary perspectives. A non-LLM moderator merges and refines proposals via multi-round negotiation, ensuring each agent's viewpoint is incorporated while penalizing repeated or spurious responses. Experiments on European city queries show improved diversity and relevance compared to single-agent baselines.

## Features
- Multi-agent recommendation generation (personalization, popularity, sustainability)
- Multi-round negotiation and moderation pipeline
- FastAPI endpoint for serving negotiation runs
- Local model support via vLLM (GPU) and cloud model support via ADK integrations

## Repository Artifacts
- Core source code and experiment scripts
- Prompt templates for all agents
- Input data and analysis notebooks

## Reproducibility Checklist

### 1) Install dependencies with uv
This repository uses `uv` for fast, reproducible dependency management via `pyproject.toml` and `uv.lock`.

Install `uv` (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Sync project dependencies (including dev):

```bash
uv sync --dev
```


### 2) Configure environment variables
Create a local env file:

```bash
cp .env.docker.example .env.docker
```

Minimum for Gemini-based runs:
- `GOOGLE_API_KEY`

Optional (needed for Vertex-backed models, for example Claude via Vertex):
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `VERTEXAI_LOCATION`

## Inference Backends

### Option A: Local GPU models with vLLM (ADK-compatible)
Current code-level routing in `src/adk/agents/agent.py`:
- `model_name=gemma-4b` -> `google/gemma-3-4b-it` at `http://localhost:8000/v1`
- `model_name=gemma-12b` -> `google/gemma-3-12b-it` at `http://localhost:8001/v1`

Serve Gemma 4B:

```bash
vllm serve google/gemma-3-4b-it \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192
```

Serve Gemma 12B (dual GPU example):

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve google/gemma-3-12b-it \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192
```

Reference tutorial for local Gemma + ADK + GPU details:
- https://medium.com/google-developer-experts/tutorial-running-googles-gemma-4b-locally-with-google-adk-and-dual-a40-gpus-5a421fbc6ef9

### Option B: Cloud models
Use `model_name=gemini-2.5-flash` (or other configured cloud route) when calling the API.

## Deployment Matrix

| Mode | API Start | Model Backend |
|---|---|---|
| Local (host) | `uv run uvicorn src.server.main:app --host 0.0.0.0 --port 8005` | Local `vllm` or cloud |
| Docker (single container) | `docker run --rm -p 8005:8005 --env-file .env.docker collab-rec-api` | Cloud by default; local `vllm` requires network config |
| Docker Compose | `docker compose up --build` | Cloud by default; local `vllm` requires same network + URL update |

## Run the API

Build Docker image (once per change):

```bash
docker build -t collab-rec-api .
```

Run locally with uv:

```bash
uv run uvicorn src.server.main:app --host 0.0.0.0 --port 8005
```


Run in Docker:

```bash
docker run --rm -p 8005:8005 --env-file .env.docker collab-rec-api
```

Run with Docker Compose:

```bash
set -a
source .env.docker
set +a
docker compose up --build
```

## API Usage
Endpoint: `POST /run-negotiation-pipeline` (implemented in `src/server/endpoints.py`)

OpenAPI docs:
- `http://localhost:8005/docs`

Example call (swap `model_name` as needed):

```bash
curl -X POST "http://localhost:8005/run-negotiation-pipeline?rounds=3&min_rounds=1&model_name=gemma-4b" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Plan a 5-day Europe trip with local food and walkable cities",
    "filters": {
      "budget": "medium",
      "month": "May",
      "interests": "food",
      "popularity": "low",
      "aqi": "good",
      "walkability": "great",
      "seasonality": "medium"
    }
  }'
```

## Shutdown

```bash
docker compose down
```

For `docker run` or local `uvicorn`, stop with `Ctrl+C`.

## Docker + Local vLLM Networking Note
The local model URLs in code use `localhost`. If the API runs in a container and vLLM runs on the host, networking may require additional configuration. A straightforward setup is:
- run both API and vLLM on the host, or
- run both in the same Docker network and adjust base URLs accordingly.

## Citation
If you use this work, please cite:

```bibtex
@article{banerjee2025collab,
  title={Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism},
  author={Ashmi Banerjee and Adithi Satish and Fitri Nur Aisyah and Wolfgang Wörndl and Yashar Deldjoo},
  url={https://arxiv.org/abs/2508.15030},
  year={2025}
}
```
