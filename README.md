# HelixRAG

## 🎯 Features
	•	FastAPI backend with automatic API documentation
	•	Supabase integration with pgvector for vector similarity search
	•	Multi-AI Provider support (Aliyun, OpenAI & Anthropic)
	•	Aliyun Qwen as the recommended default provider
	•	Vector embeddings with semantic search
	•	Citation-based answers with source tracking
	•	Enhanced reranking with DashScopeRerank (Qwen)
	•	Bilingual query augmentation (CN⇄EN) to improve recall
	•	Strict citation prompting + server-side citation fallback
	•	File uploads and directory ingestion with LlamaIndex readers

## 🏗️ Architecture

```
HelixRAG/
├── app/
│   ├── core/           # Infrastructure (config, database)
│   ├── models/         # Pydantic data models
│   ├── services/       # Business logic (RAG, embeddings)
│   └── main.py         # FastAPI application
├── sql/
│   └── init_supabase.sql  # Database initialization script
└── requirements.txt
```

## 🚀 Quick Start Guide

**Complete setup from clone to asking questions in ~10 minutes**

---

### Prerequisites
- Python 3.12 (Tencent Cloud Ubuntu typically ships 3.12 by default)
- Supabase account
- Aliyun API key 
- Anthropic API key (optional, for Claude)

---

### Step 1: Clone and Install Dependencies (Tencent Cloud / Ubuntu)

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y python3 python3-venv python3-dev build-essential git curl

git clone https://github.com/ZhaoYi-10-13/HelixRAG.git
cd HelixRAG

python3 -m venv venv_helixrag
source venv_helixrag/bin/activate

# pip config set global.index-url http://mirrors.tencentyun.com/pypi/simple
# pip config set install.trusted-host mirrors.tencentyun.com

pip install -r requirements.txt
```

---

### Step 2: Get API Keys (5 minutes)

**Supabase Setup:**
1. Go to [supabase.com](https://supabase.com) and create a new project
2. Wait for project to be ready (~2 minutes)
3. Go to **Settings** → **API** and copy:
   - **Project URL** (e.g., `https://abc123.supabase.co`)
   - **Anon public key** (starts with `eyJ...`)
   - **Service role secret key** (starts with `eyJ...`)

**(Optional) OpenAI / Anthropic Setup:**
You can also configure OPENAI_API_KEY or ANTHROPIC_API_KEY as backup providers.

---

### Step 3: Configure Environment

```bash
cp .env.example .env
# Create and edit your environment file
nano .env
```

Put the following into `.env` (fill in real values where needed):
```env
# -------- Supabase --------
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here

# -------- AI Provider --------
AI_PROVIDER=aliyun

# Aliyun Qwen (default)
ALIYUN_API_KEY=your_aliyun_key_here
ALIYUN_CHAT_MODEL=qwen-plus
ALIYUN_EMBED_MODEL=text-embedding-v4

# -------- App Config --------
ENVIRONMENT=development
LOG_LEVEL=INFO
ENABLE_RERANKING=true
RERANK_TOP_N=6
MAX_FILE_SIZE_MB=50
SUPPORTED_FILE_TYPES=pdf,docx,txt,md,html,csv
```

---

### Step 4: Initialize Database (2 minutes)

1. **Open Supabase Dashboard** → **SQL Editor**
2. **Click "New query"**
3. **Copy entire contents** of `sql/init_supabase.sql`
4. **Paste and click "Run"**

✅ This creates everything needed:
- pgvector extension
- `rag_chunks` table with VECTOR(1024)
- Performance indexes
- Vector search functions
- RLS policies for future auth

---

### Step 5: Test Setup (Optional)

```bash
# Test your complete setup
python tests/test_setup.py
```

This verifies:
- ✅ Dependencies installed
- ✅ API keys configured
- ✅ Database connected
- ✅ Schema initialized
- ✅ RAG pipeline working
- 
---

### Step 6: Start the Server

```bash
# Local development
uvicorn app.main:app --reload --port 8000

# Server exposure (e.g., Tencent Cloud ECS)
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Or use port 80 (may require sudo privileges)
# uvicorn app.main:app --host 0.0.0.0 --port 80
```

Note: Binding to port 80 on Linux may require elevated privileges. If you encounter a permission error, run with `sudo` or prefer port 8000.

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

On a server (public IP):
- http://<your-public-ip>:8000
- Docs: http://<your-public-ip>:8000/docs

## 📁 Project Structure

```
HelixRAG/
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── database.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── default_documents.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── entities.py
│   │   ├── requests.py
│   │   └── responses.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── chunker.py
│   │   ├── document_parser.py
│   │   ├── embedding.py
│   │   ├── rag.py
│   │   └── rerank.py
│   ├── __init__.py
│   └── main.py
├── sql/
│   └── init_supabase.sql
├── static/
├── examples/
├── tools/
├── tests/
│   └── test_setup.py
├── main.py                 # Legacy entrypoint forwarding to app.main:app
├── requirements.txt
├── README.md
└── .env.example
```

## 🔧 Enhancements and Optimizations

Below is a concrete, implementation-level comparison that explains the improvements and their impact.

- Reranking
  - Previous: Pure vector-similarity ranking, susceptible to long-text noise and near-synonym interference.
  - New: Integrates DashScopeRerank (Qwen official Rerank). After vector retrieval, results are re-ordered using cross-model signals.
    - Code: `app/services/rerank.py`
    - Key implementation:
      - `DashScopeRerank(top_n=..., return_documents=True, api_key=settings.dashscope_api_key)`
      - Also set `dashscope.api_key = settings.dashscope_api_key` so the SDK can access the key globally
    - Effect: Logs show “Reranked N results to N top results”. Especially improves relevance when Chinese queries hit English documents.

- Query Recall (CN⇄EN)
  - Previous: Single-language, direct vectorization of the original query; insufficient recall across languages or varied phrasings.
  - New: `_augment_queries` performs lightweight synonym expansion and bidirectional CN⇄EN augmentation (e.g., 退货/退款/退换/政策/shipping/size), runs retrieval per “augmented query”, deduplicates, then feeds into Rerank.
    - Code: `app/services/rag.py`, `RAGService._augment_queries`
    - Key implementation:
      - CN→EN and EN→CN keyword dictionaries plus domain phrase boosts (e.g., add “退货 政策 退款”)
      - Retrieval flow: multi-query retrieval → merge & deduplicate → Rerank → take top-N as context
    - Effect: Robust hits for queries like “退款政策”, “free shipping”, “shoe size”; significantly better cross-language recall.

- Citations (consistency and fallback)
  - Previous: Model might not output parseable citations; UI sometimes lacked sources.
  - New:
    - Stronger prompting: `app/services/chat.py` `SYSTEM_PROMPT` requires adding `[chunk_id]` citations; context format `[chunk_id] content` helps the model cite correctly.
    - Fallback: In `app/services/rag.py`, if `_extract_citations` returns empty, use `top_doc_ids` from the current context as `citations`, ensuring sources are always presented.
    - Effect: Answers are traceable; even when the model occasionally omits citations, the backend provides stable source annotations.

- Parsing & Ingestion
  - Previous: Basic parsing with limited format coverage.
  - New: Unified parsing via LlamaIndex file readers for PDF / DOCX / TXT / MD / HTML / CSV, automatic metadata extraction, and chunking per your configured `chunk_size/overlap`, followed by embedding and upsert.
    - Code: `app/services/document_parser.py`, and endpoints `/upload` and `/process-directory`
    - Effect: “Out-of-the-box” multi-format ingestion, more robust parsing, and an automated pipeline (parse → chunk → embed → upsert).

- Prompting & Generation
  - New: Provide context in the form `[chunk_id] content`; prompt explicitly instructs “answer only from context; if unknown, output: I don't know.”
  - Code: `app/services/chat.py`
  - Effect: Reduces hallucinations, stabilizes citations, and keeps answers focused.

- Observability & Regression
  - Logging pipeline: Embedding → Vector Search (RPC/fallback) → Rerank → Chat → citations/latency. Each key step has INFO logs.
  - Batch tests: scripts in `tests/` validate typical queries (returns/shipping/size/customer service), enabling quick regression.

