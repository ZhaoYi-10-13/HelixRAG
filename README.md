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
- Python 3.11+
- Supabase account
- Aliyun API key 
- Anthropic API key (optional, for Claude)

---

### Step 1: Clone and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/ZhaoYi-10-13/HelixRAG.git
cd HelixRAG

# Create virtual environment
python3.11 -m venv venv_helixrag
source venv_helixrag/bin/activate  # On Windows: venv_helixrag\Scripts\activate

# Install dependencies
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

**Aliyun Setup (recommended):**
1.	Go to [Aliyun Bailian Console](https://bailian.console.aliyun.com)
2.	Login and get the API for large models (LLM API Key)
3.	Copy your ALIYUN_API_KEY

**(Optional) OpenAI / Anthropic Setup:**
You can also configure OPENAI_API_KEY or ANTHROPIC_API_KEY as backup providers.

---

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your real API keys
nano .env  # or use your preferred editor
```

Update .env with your values:
```env
# -------- Supabase --------
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=your_anon_key_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here

# -------- AI Provider --------
AI_PROVIDER=aliyun

# Aliyun Qwen (default)
ALIYUN_API_KEY=sk-your_aliyun_key_here
ALIYUN_CHAT_MODEL=qwen-plus
ALIYUN_EMBED_MODEL=text-embedding-v4

# Optional：OpenAI
OPENAI_API_KEY=sk-your_openai_key_here
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_EMBED_MODEL=text-embedding-3-large

# Optional：Anthropic
ANTHROPIC_API_KEY=your_anthropic_key_here
ANTHROPIC_CHAT_MODEL=claude-3-5-sonnet-20241022

# -------- App Config --------
ENVIRONMENT=development
LOG_LEVEL=INFO
```

---

### Step 4: Initialize Database (2 minutes)

1. **Open Supabase Dashboard** → **SQL Editor**
2. **Click "New query"**
3. **Copy entire contents** of `sql/init_supabase.sql`
4. **Paste and click "Run"**

✅ This creates everything needed:
- pgvector extension
- `rag_chunks` table with VECTOR(3072) for latest embeddings
- Performance indexes
- Vector search functions
- RLS policies for future auth

---

### Step 5: Test Setup (Optional)

```bash
# Test your complete setup
python test_setup.py
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
uvicorn main:app --reload --port 8000
uvicorn main:app --host 0.0.0.0 --port 80 # For Aliyun HTTP Service
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 API Usage

### Health Check
```bash
curl http://localhost:8000/healthz
```

### Seed Knowledge Base
```bash
# Seed with default documents
curl -X POST http://localhost:8000/seed

# Or seed with custom documents
curl -X POST http://localhost:8000/seed \
  -H "Content-Type: application/json" \
  -d '{
    "docs": [
      {
        "chunk_id": "policy_returns_v1#window",
        "source": "https://help.example.com/returns",
        "text": "You can return unworn items within 30 days of purchase..."
      }
    ]
  }'
```

### Ask Questions (RAG)
```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can I return shoes after 30 days?",
    "top_k": 6
  }'
```

**Example Response:**
```json
{
  "text": "Based on our return policy, you can return unworn shoes within 30 days of purchase [policy_returns_v1#window]. Items must be in original condition...",
  "citations": ["policy_returns_v1#window", "policy_returns_v1#conditions"],
  "debug": {
    "top_doc_ids": ["policy_returns_v1#window", "policy_returns_v1#conditions"],
    "latency_ms": 1250
  }
}
```

## 🔬 New vs Original Techniques

Below is a concrete, implementation-level comparison，并说明升级带来的效果：

- Reranking（重排序）
  - 原技术：纯向量相似度排序，容易被长文本噪声或近义干扰影响。
  - 新技术：接入 DashScopeRerank（通义千问官方 Rerank），在向量检索后使用跨模型信号进行二次重排。
    - 代码位置：`app/services/rerank.py`
    - 关键实现：
      - `DashScopeRerank(top_n=..., return_documents=True, api_key=settings.dashscope_api_key)`
      - 同时设置 `dashscope.api_key = settings.dashscope_api_key` 确保 SDK 全局可见
    - 效果：日志可见 “Reranked N results to N top results”，中文查询命中英文文档时相关性显著提升。

- Query Recall（召回增强，CN⇄EN 双向）
  - 原技术：单一语种/原始查询直接向量化检索，遇到跨语种或多表达时召回不足。
  - 新技术：`_augment_queries` 对查询做轻量同义词与中英双向扩展（退货/退款/退换/政策/shipping/size 等），对每个“扩展查询”分别检索，合并去重后交给 Rerank。
    - 代码位置：`app/services/rag.py` 中 `RAGService._augment_queries`
    - 关键实现：
      - 中文→英文、英文→中文关键词字典 + 域短语增强（例如附加 “退货 政策 退款”）
      - 检索流程：多查询检索 → 聚合去重 → Rerank → 取前若干作为上下文
    - 效果：中文问“退款政策”、英文问“free shipping”“shoe size”等均稳定命中；跨语召回率显著提升。

- Citations（引用一致性与兜底）
  - 原技术：模型不一定输出可解析的引用，前端偶尔没有来源展示。
  - 新技术：
    - 强化提示词：`app/services/chat.py` 中的 `SYSTEM_PROMPT` 明确要求在答案中添加 `[chunk_id]` 引用；上下文格式为 `[chunk_id] 内容`，便于模型引用。
    - 兜底策略：`app/services/rag.py` 中若 `_extract_citations` 为空，则使用当前上下文的 `top_doc_ids` 作为 `citations` 返回，保证前端永远有来源可展示。
    - 效果：答案可溯源；即使模型偶发遗漏，后端也能稳定提供来源标注。

- Parsing & Ingestion（文档解析与入库）
  - 原技术：基础解析，格式覆盖有限。
  - 新技术：基于 LlamaIndex 文件 Reader，统一解析 PDF / DOCX / TXT / MD / HTML / CSV，自动抽取元数据，并按你配置的 `chunk_size/overlap` 切分后嵌入、入库。
    - 代码位置：`app/services/document_parser.py`、`/upload` 与 `/process-directory` 端点
    - 效果：多格式文档“开箱即用”，解析更稳健，入库链路（解析→分块→嵌入→upsert）端到端自动化。

- Prompting & Generation（提示与生成）
  - 新增：将上下文按 `[chunk_id] 内容` 提供给模型；提示中明确“仅根据上下文回答；若无答案请输出 I don't know.”
  - 代码位置：`app/services/chat.py`
  - 效果：减少幻觉，引用更稳定，回答更聚焦。

- 可观测性与回归验证
  - 日志链路：Embedding → Vector Search（含 RPC/fallback）→ Rerank → Chat → citations/latency，关键步骤均有 INFO 日志。
  - 批量测试：`tests/` 下提供快速脚本验证退货/运费/尺码/客服等典型问法，便于回归。

# 🚀 阿里云部署服务指南

**以下步骤展示如何在阿里云 ECS 上快速部署 FastAPI 项目并开放公网访问。**

---

## 1. 购买与登录 ECS
1. 打开阿里云个人主页，点击 **立即体验**。  
2. 在顶部导航栏中选择 **产品** → **云服务器 ECS** → **立即购买**。  
3. 选择最低配置（最低仅需 **99 元 / 年**），也可以选择试用（但会有服务限制）。  
4. 下单完成后进入 **控制台**，点击 **远程连接**，即可自动进入 **Terminal 界面**。

---

## 2. 克隆项目与环境配置
在基础的ECS服务器中下载可能需要一段时间
```bash  
# 更新系统包
sudo apt update && sudo apt upgrade -y

# 安装 Python 3.11 及其工具
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils

# 安装常用工具
sudo apt install -y git curl build-essential

# 克隆项目
git clone https://github.com/ZhaoYi-10-13/HelixRAG.git
cd HelixRAG

# 创建虚拟环境
python3.11 -m venv venv_helixrag
source venv_helixrag/bin/activate

# 安装依赖
pip install -r requirements.txt
```
之后的步骤和前面是一样的，确保测试通过即可

---

## 3. 启动服务

使用以下命令之一启动服务（根据需要选择端口）：

```bash
uvicorn main:app --host 0.0.0.0 --port 80
# 或者
uvicorn main:app --host 0.0.0.0 --port 8000
```
	•	8000 端口：开发与调试推荐
	•	80 端口：生产环境推荐（无需加端口号）

---

## 4. 配置防火墙 (UFW)

检查并放行端口：

```bash
# 查看状态
sudo ufw status

# 开启 UFW（如果未启用）
sudo ufw enable

# 放行 SSH + HTTP/HTTPS + 开发端口
sudo ufw allow 80/tcp
sudo ufw allow 8000/tcp

# 重载规则
sudo ufw reload
```

---

## 5. 配置安全组

进入阿里云控制台 → ECS → 网络与安全组 → 安全组：
	1.	选择 添加入方向规则
	2.	协议类型选择：
	•	自定义 TCP（端口 8000）
	•	或者 Web HTTP 流量（端口 80）
	3.	访问来源：0.0.0.0/0（允许任何位置访问）
	4.	优先级和授权策略保持默认
	5.	点击 确定 保存

---

## 6. 检查服务运行状态

在云服务器终端运行：
```bash
# 如果跑在 8000 端口
curl http://127.0.0.1:8000/healthz

# 如果跑在 80 端口
curl http://127.0.0.1/healthz
```
正常输出示例：
```bash
{"status":"degraded","database_connected":false}
```
说明服务已启动成功（提示 degraded 是正常的，因为数据库服务器在国外）。

---

## 7. 浏览器访问
	1.	回到阿里云 控制台 → 网络与安全组
	2.	在左侧找到 公网 IP 并复制
	3.	在浏览器输入：

	•	如果是 8000 端口：

http://<公网IP>:8000/chat


	•	如果是 80 端口：

http://<公网IP>/chat

即可进入聊天页面 🎉

如果你觉得页面太单一，可以查看[这个项目](https://github.com/ZhaoYi-10-13/Agent-ui.git)去了解如何基于 React + Vite 构建一个首页从而做到更精美！
