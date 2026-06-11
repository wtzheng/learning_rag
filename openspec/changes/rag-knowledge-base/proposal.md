## Why

个人学习需要一个 RAG 知识库系统，用于对本地 PDF/TXT 文档进行智能问答。现有工具要么太重（生产级部署），要么太黑盒（套壳产品看不到内部机制）。本项目从零构建一个可理解的 RAG 系统，目的是深入理解每一环节的工作原理，同时产出可用的工具。

## What Changes

- 基于 LlamaIndex 框架搭建完整的 RAG 索引与查询管线
- 支持本地 Embedding 模型（BGE-M3）进行文档向量化
- 支持 Chroma 作为持久化向量数据库
- 支持 PDF（电子版为主）和 TXT 文档的加载与处理
- 支持多种 Chunking 策略并可切换对比（按句 / 按词 / 语义切分）
- 支持 Hybrid 检索（向量检索 + BM25 关键词检索）
- 统一 LLM 接口，支持 DeepSeek API 和阿里云百炼 API 切换
- 提供 CLI 入口用于查询交互
- 提供 Jupyter Notebook 用于分阶段实验和可视化对比

## Capabilities

### New Capabilities

- `document-loading`: 从本地目录加载 PDF 和 TXT 文档，提取文本内容，附加来源元数据
- `text-chunking`: 将文档切分为可检索的文本块，支持多种策略（SentenceSplitter、Jieba 分词感知、语义切分）
- `local-embedding`: 基于 BGE-M3 本地嵌入模型，将文本块转换为向量
- `vector-storage`: 基于 Chroma 的向量存储与相似度检索，支持元数据过滤
- `hybrid-retrieval`: 向量检索 + BM25 关键词检索的混合检索，支持 RRF 排名融合
- `llm-inference`: 统一的大语言模型接口，支持 DeepSeek 和百炼 API 切换
- `query-pipeline`: 编排完整的 RAG 查询流程，包括检索、prompt 组装、LLM 生成

### Modified Capabilities

*暂无*

## Impact

- 新增 Python 项目依赖：llama-index, chromadb, sentence-transformers, rank-bm25, jieba, pymupdf
- 新增本地模型文件：BAAI/bge-m3（~2.2GB 首次下载）
- 新增数据目录 `data/raw/` 用于存放文档，`data/chroma_db/` 用于向量库持久化
- 需配置环境变量 `DEEPSEEK_API_KEY` 和 `BAILIAN_API_KEY`
