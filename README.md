# Collab-Rec: An LLM-based Agentic Framework for Tourism Recommendations

**Authors:** Ashmi Banerjee, Fitri Nur Aisyah, Adithi Satish, Wolfgang Wörndl, Yashar Deldjoo

**Paper URL:** [https://arxiv.org/pdf/2508.15030](https://arxiv.org/pdf/2508.15030)

**Abstract:**  
We propose Collab-REC, a multi-agent framework designed to counteract popularity bias and enhance diversity in tourism recommendations. In our setting, three LLM-based agents— **Personalization**, **Popularity**, and **Sustainability** — generate city suggestions from complementary perspectives. A non-LLM moderator then merges and refines these proposals via multi-round negotiation, ensuring each agent's viewpoint is incorporated while penalizing spurious or repeated responses.
Experiments on European city queries demonstrate that Collab-REC enhances diversity and overall relevance compared to a single-agent baseline, surfacing lesser-visited locales that are often overlooked. This balanced, context-aware approach addresses over-tourism and better aligns with constraints provided by the user, highlighting the promise of multi-stakeholder collaboration in LLM-driven recommender systems
## Features
- Multi-agent design addressing personalization, popularity, and sustainability
- Iterative negotiation for balanced, context-aware recommendations
- Reduced popularity bias and surfacing of lesser-visited locales
- Reproducible pipeline

## Artifacts
The repository includes:
- Code and scripts
- Example prompts for each agent
- Datasets used in experiments

## Dependency Management (Poetry)

This project now uses `poetry` as the source of truth for dependencies (`pyproject.toml`).

Install dependencies locally:

```bash
poetry install
```

Run the API locally (without Docker):

```bash
poetry run uvicorn src.server.main:app --host 0.0.0.0 --port 8005
```

## Run with Docker (Local Deployment)

The API exposes the multi-agent, multi-round negotiation endpoint at:
`POST /run-negotiation-pipeline` (Code in: `src/server/endpoints.py`)

### 1) Prepare environment variables

Create a local env file for Docker:

```bash
cp .env.docker.example .env.docker
```

Fill at least:
- `GOOGLE_API_KEY` for Gemini-based runs.

If you use Vertex-backed models (for example Claude via Vertex), also set:
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION`
- `VERTEXAI_LOCATION`

### 2) Build and run the API container

The Docker image installs dependencies with `poetry` from `pyproject.toml`.

```bash
docker build -t collab-rec-api .
docker run --rm -p 8005:8005 --env-file .env.docker collab-rec-api
```

Or with Docker Compose:

```bash
set -a
source .env.docker
set +a
docker compose up --build
```

### 3) Call the negotiation endpoint

OpenAPI docs:
- `http://localhost:8005/docs`

Example request:

```bash
curl -X POST "http://localhost:8005/run-negotiation-pipeline?rounds=3&min_rounds=1&model_name=gemini-2.5-flash" \
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

### 4) Stop the app

```bash
docker compose down
```

For `docker run`, stop with `Ctrl+C` in the running terminal.


## Citation
If you use this work, please cite the paper:

```
@article{banerjee2025collab,
  title={Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism},
  author={Banerjee, Ashmi and Aisyah, Fitri Nur and Satish, Adithi and W{\"o}rndl, Wolfgang and Deldjoo, Yashar},
  journal={arXiv preprint arXiv:2508.15030},
  year={2025}}
```
