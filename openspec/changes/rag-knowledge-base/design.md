## Context

个人学习用 RAG 知识库。用户文档以 PDF（电子版为主）和 TXT 为主，需要智能问答。项目定位是学习型，强调模块化、可实验、可观测。

### 技术选型

| 决策 | 选择 | 理由 |
|------|------|------|
| 框架 | LlamaIndex | RAG 原生设计，概念清晰 |
| Embedding | BGE-M3 (本地) | 免费、离线、多语言 |
| 向量库 | Chroma | 轻量、零配置、持久化 |
| 检索 | Hybrid (Dense + BM25) | 中文场景召回率更高 |
| LLM | DeepSeek + 百炼 | 统一接口按需切换 |
| 文档加载 | PyMuPDF | 电子版 PDF 首选 |

### 项目结构

```
learning_rag/
├── data/
│   ├── raw/               ← 文档原料
│   └── chroma_db/         ← Chroma 持久化
├── src/
│   ├── config.py           ← 所有配置收拢
│   ├── loader.py           ← 文档读取
│   ├── chunking.py         ← 切分策略 (多策略 + 统一接口)
│   ├── embedding.py        ← BGE-M3 嵌入
│   ├── vector_store.py     ← Chroma 读写
│   ├── retrieval.py        ← 检索逻辑 (dense / hybrid)
│   ├── llm.py              ← LLM 统一接口
│   ├── pipeline.py         ← 串联 index() / query()
│   ├── prompts.py          ← Prompt 模板
│   └── cli.py              ← 命令行入口
├── notebooks/
│   ├── 01_加载与探索.ipynb
│   ├── 02_Chunking实验.ipynb
│   └── 03_端到端管线.ipynb
├── .env
└── pyproject.toml
```

## Goals / Non-Goals

**Goals:**
- 完整可运行的 RAG 管线，支持索引和查询
- 多种 Chunking 策略可切换对比
- Dense + Sparse 混合检索
- DeepSeek / 百炼 LLM 双 provider 支持
- CLI 交互 + Notebook 实验
- 所有配置集中管理，方便调参实验

**Non-Goals:**
- 生产级部署（高并发、高可用）
- Web UI（初始版本只做 CLI）
- OCR 管线（仅做失败检测和日志，不做完整 OCR）
- 多用户/权限系统
- 流式输出

## Decisions

### 1. 框架选择：LlamaIndex

LlamaIndex 相比 LangChain 对 RAG 场景更专注，概念更少（Document/Node/Index/Engine），更贴合"学习 RAG"这个目标。

### 2. 模块依赖方向

```
loader.py → chunking.py → embedding.py → vector_store.py
                                              │
                                         retrieval.py
                                              │
                                         pipeline.py
                                              │
                                           llm.py
                                              │
                                           cli.py
```

依赖单向，无循环，每个文件可独立调试。

### 3. Chunking 策略

三种策略通过统一 `BaseChunker` 接口暴露：
- **SentenceSplitter**：LlamaIndex 内置，按字符+标点边界，适合大部分文本
- **JiebaChunker**：先分词再组块，保证中文词不被切碎
- **SemanticChunker**：检测句子间 embedding 相似度突降点，自动识别主题边界

### 4. Hybrid 检索

Chroma 只支持向量检索。BM25 索引在内存中构建（`rank_bm25` + `jieba`），两者通过 RRF（Reciprocal Rank Fusion）融合。RRF 选择理由：向量分数和 BM25 分数在不同量级，RRF 只依赖排名，天然对齐。

### 5. LLM 统一接口

DeepSeek 和百炼都是 OpenAI 兼容接口，用 `openai` 库统一调用，通过 provider 参数切换。支持自动路由：简单问题 → DeepSeek（便宜快速），复杂问题 → 百炼（中文更强）。

### 6. Config 集中管理

```python
# config.py - 所有可调参数在一个文件
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
CHUNK_STRATEGY = "sentence"      # sentence | jieba | semantic
EMBED_MODEL = "BAAI/bge-m3"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "rag_knowledge_base"
TOP_K = 5
RETRIEVAL_MODE = "hybrid"         # dense | hybrid
RERANK_ENABLED = False
LLM_PROVIDER = "deepseek"        # deepseek | bailian
LLM_TEMPERATURE = 0.3
```

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|----------|
| BGE-M3 模型 ~2.2GB，首次下载耗时 | 使用 sentence-transformers 缓存，仅下载一次 |
| Chroma 不支持 BM25，需要双索引 | BM25 索引轻量（内存），重建成本低 |
| 中文 PDF 提取质量不稳定 | 先用 PyMuPDF，检测低文本量页面并告警 |
| Jieba 分词对专业术语可能不准 | 支持用户自定义词典扩展 |
| DeepSeek / 百炼 API 可能变更 | 通过 OpenAI 兼容接口解耦，变更影响局部化 |
