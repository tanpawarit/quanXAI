## Quick Start

### Prerequisites
- Python 3.12+

## Docker Deployment

### Clone & Configure
```bash
git clone https://github.com/tanpawarit/quanXAI.git
cd quanXAI

# Copy environment file and add ENV
cp .env.example .env
```

### 1. Running ingestion *before* starting the API avoids database locking issues.
```bash
docker-compose run --rm app python -m src.cli.ingest --sync
```

### 2. Build & Start
```bash
docker-compose up --build -d
```

### 3. Access API
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Useful Commands
```bash
# View logs
docker logs -f quanxai-api

# Stop
docker-compose down

# Rebuild
docker-compose up --build -d
```

---

## Test the API

### Product Catalog RAG
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What wireless headphones do we have in stock?"}'
```

### Web Search (Market Trends)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Current market price for noise-cancelling headphones?"}'
```

### Price Analysis
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which products have lowest profit margins?"}'
```

### Multi-Tool Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Should we adjust AudioMax headphones pricing vs competitors?"}'
```

---

### Limitations & Future Improvements
```markdown
Limitations & Future Improvements

What's Not Implemented:

- Sequential Agent Execution: The Agent Graph runs product_qa and market_analysis steps sequentially, even if they could be executed in parallel to reduce latency.
- Memory-Limited Ingestion: The DataIngester loads the entire CSV dataset into memory before processing. This will cause Out-Of-Memory (OOM) errors with large datasets (e.g., >100MB files).
- Lack of Resiliency: There is no exponential backoff or retry mechanism implemented for external API calls (OpenAI, Tavily) or database connections.
- Agent Request ID: For each request, should include a user identifier or a session ID for production.

What I Would Improve:

- Add Observability & Tracing: Integrate tools like LangSmith or Langfuse to trace the agent's decision-making process and debug complex multi-step queries.
- Semantic Caching: Add a caching layer (e.g., Redis) to store and retrieve similar queries, reducing costs and latency.
- Session Memory: Implement multi-turn conversation (session memory) using Redis so the agent can remember context from previous interactions and provide faster responses.
- Scalable Infrastructure: Migrate from Milvus Lite and SQLite to standalone Milvus Cluster and PostgreSQL to handle millions of vectors and concurrent users efficiently.

What I Learned:

- Balancing Determinism & Agency: Learned when to enforce strict logic (Price Analysis must be 100% accurate) versus when to let the Agent be creative (Market Research and conversational synthesis).
- Agent Design: Realized the importance of clear tool selection, effective structured prompts, and structured output for successful agent orchestration.
- Design Patterns & Architecture: Gained hands-on experience with Architecture and principles, which were previously unfamiliar territory. This project helped solidify my understanding of separation of concerns and dependency inversion. 
```
