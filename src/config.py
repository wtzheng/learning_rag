"""
Centralized configuration for the RAG knowledge base.
All tunable parameters live here - experiment by changing these values.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Document Loading ──
DATA_DIR = "data/raw"

# ── Chunking ──
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
CHUNK_STRATEGY = "sentence"  # "sentence" | "jieba" | "semantic"

# ── Embedding ──
EMBED_MODEL = "BAAI/bge-m3"  # switch to "BAAI/bge-small-zh-v1.5" for faster loading (~30MB)
EMBED_NORMALIZE = True
HF_ENDPOINT = os.getenv("HF_ENDPOINT", None)

# ── Vector Store ──
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "rag_knowledge_base"

# ── Retrieval ──
TOP_K = 5
RETRIEVAL_MODE = "hybrid"  # "dense" | "hybrid"
HYBRID_ALPHA = 0.5         # 0.0 = pure BM25, 1.0 = pure dense
RRF_K = 60                 # RRF fusion constant
RERANK_ENABLED = False
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K = 5

# ── LLM ──
LLM_PROVIDER = "deepseek"  # "deepseek" | "bailian"
LLM_TEMPERATURE = 0.3

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_THINKING = False  # True = thinking/reasoning mode

BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY", "")
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BAILIAN_MODEL = "qwen-max"

# ── Observability ──
LOG_LEVEL = "INFO"
