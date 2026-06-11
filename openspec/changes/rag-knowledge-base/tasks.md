## 1. 项目初始化与骨架

- [x] 1.1 初始化 Python 项目（pyproject.toml），安装所有依赖（llama-index, chromadb, sentence-transformers, rank-bm25, jieba, pymupdf, python-dotenv, openai）
- [x] 1.2 创建 `data/raw/` 和 `data/chroma_db/` 目录
- [x] 1.3 创建 `.env` 模板文件（DEEPSEEK_API_KEY, BAILIAN_API_KEY）
- [x] 1.4 创建 `src/config.py`，收拢所有可调参数（chunk_size, embed_model, top_k, llm_provider 等）

## 2. 文档加载模块

- [x] 2.1 实现 `src/loader.py`：支持 PDF（PyMuPDF）和 TXT 文件加载，返回统一 Document 结构（text + metadata）
- [x] 2.2 实现目录扫描：遍历 `data/raw/` 自动发现所有 PDF/TXT，过滤不支持的文件类型
- [x] 2.3 实现 PDF 低文本量检测：当页面提取文本 < 50 字符时标记警告

## 3. Chunking 模块

- [x] 3.1 定义 `Chunk` 数据类（text, index, metadata）
- [x] 3.2 定义 `BaseChunker` 抽象基类及 `SentenceChunker` 实现（基于 LlamaIndex SentenceSplitter，支持中文标点边界）
- [x] 3.3 实现 `JiebaChunker`：先 jieba 分词，按词边界组块，保证词不被切碎
- [x] 3.4 实现 `SemanticChunker`：计算相邻句子 embedding 相似度，检测语义突降点作为切分边界
- [x] 3.5 实现 `get_chunker()` 工厂函数，支持通过配置名切换策略

## 4. Embedding 模块

- [x] 4.1 实现 `src/embedding.py`：加载 BAAI/bge-m3 模型，提供 embed_texts() 批量推理接口
- [x] 4.2 确保输出向量 L2 归一化

## 5. 向量存储模块

- [x] 5.1 实现 `src/vector_store.py`：初始化 Chroma 客户端，持久化到 `data/chroma_db/`
- [x] 5.2 实现 add_chunks() 批量入库
- [x] 5.3 实现 similarity_search() 余弦相似度检索，返回 chunk + score + metadata
- [x] 5.4 实现 clear() 清空重建
- [x] 5.5 集成 index 管线：loader → chunker → embedder → vector_store

## 6. 检索模块

- [x] 6.1 实现 `src/retrieval.py`：纯向量检索（直接调用 Chroma similarity_search）
- [x] 6.2 实现 BM25 索引构建：所有 chunk 经 jieba 分词后建立 BM25Okapi 索引
- [x] 6.3 实现 Hybrid 检索：dense + sparse → RRF 融合（支持 alpha 参数）
- [x] 6.4 可选：实现 Reranker 集成（BAAI/bge-reranker-v2-m3）

## 7. LLM 模块

- [x] 7.1 实现 `src/llm.py`：统一 LLM 类，通过 OpenAI 兼容接口调用 DeepSeek 和百炼
- [x] 7.2 实现自动路由：简单查询 → DeepSeek，复杂查询 → 百炼
- [x] 7.3 实现 API 错误处理：超时、鉴权失败、限流的优雅降级

## 8. Prompt 模块

- [x] 8.1 实现 `src/prompts.py`：QA 模板（来源引用 + 无幻觉约束）
- [x] 8.2 实现 Summary 模板和 Code 模板
- [x] 8.3 实现 `detect_question_type()`：根据关键词自动选择模板

## 9. 管线编排

- [x] 9.1 实现 `src/pipeline.py`：index() 函数（加载 → 切分 → 嵌入 → 存储全流程）
- [x] 9.2 实现 query() 函数（嵌入 → 检索 → 组 prompt → LLM 生成）
- [x] 9.3 实现 `src/cli.py`：命令行入口，支持 `python -m src.cli index` 和 `python -m src.cli query "问题"`

## 10. Notebook 实验

- [x] 10.1 创建 `notebooks/01_加载与探索.ipynb`：展示文档加载后结构、PDF 提取质量、metadata 示例
- [x] 10.2 创建 `notebooks/02_Chunking实验.ipynb`：对比三种策略在不同 chunk_size/overlap 下的 chunk 差异
- [x] 10.3 创建 `notebooks/03_端到端管线.ipynb`：完整跑通 index + query，对比纯向量 vs hybrid 检索效果

## 11. 实验与对比

- [x] 11.1 准备 3 个测试问题（具体型、抽象型、术语型），固定评估集
- [x] 11.2 运行 Chunking 对比实验（E1~E5），观察 chunk 数量和语义完整性
- [x] 11.3 运行检索对比实验（纯向量 vs hybrid），观察召回位置和回答质量
- [x] 11.4 运行交叉实验（chunking × 检索），记录各组合效果差异
