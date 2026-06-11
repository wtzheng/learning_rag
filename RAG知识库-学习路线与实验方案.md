# RAG 知识库 — 学习路线与实验方案

> 项目: `learning_rag` / 个人学习用 / 非产品级

---

## 一、架构总览

```
                        ┌──────────────────┐
                        │   文档集           │
                        │  (PDF / TXT)      │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │    Loader         │
                        └────────┬─────────┘
                                 │
               ┌─────────────────▼──────────────────┐
               │         A. Chunking 层              │
               │  ┌─────────┐  ┌────────┐  ┌──────┐ │
               │  │ 固定大小 │  │按句切分│  │按段落│ │
               │  └────┬────┘  └───┬────┘  └──┬───┘ │
               │       └─────┬─────┘──────────┘     │
               └─────────────────┬──────────────────┘
                                 │
               ┌─────────────────▼──────────────────┐
               │      Embedding (BGE-M3 / 本地)     │
               └─────────────────┬──────────────────┘
                                 │
               ┌─────────────────▼──────────────────┐
               │         Chroma 向量库               │
               │      (chunks + vectors + metadata)  │
               └─────────────────┬──────────────────┘
                                 │
               ┌─────────────────▼──────────────────┐
               │         B. Retriever 层             │
               │  ┌─────────┐  ┌──────────┐         │
               │  │ 向量检索 │  │ BM25     │         │
               │  │(Chroma) │  │(jieba分词)│         │
               │  └────┬────┘  └────┬─────┘         │
               │       └────┬───────┘                │
               │       hybrid fusion                 │
               │            │                        │
               │       ┌────▼────┐                   │
               │       │Reranker │ (可选，优化阶段)   │
               │       └────┬────┘                   │
               └─────────────────┬──────────────────┘
                                 │
               ┌─────────────────▼──────────────────┐
               │    Prompt Assembly + LLM            │
               │  DeepSeek API  ←→  百炼 API         │
               └────────────────────────────────────┘
```

---

## 二、核心决策

| 决策点 | 选择 | 备注 |
|--------|------|------|
| Embedding | 本地模型 (BGE-M3) | 用 `HuggingFaceEmbedding` 加载 |
| 向量库 | Chroma | 轻量、零配置、持久化 |
| 框架 | LlamaIndex | RAG 原生设计，概念清晰 |
| LLM | DeepSeek + 百炼 | 统一接口，按配置切换 |
| 文档类型 | PDF（电子版为主）+ TXT | OCR 兜底扫描件 |
| 用途 | 个人知识库 + 学习 | 不走生产级部署 |

---

## 三、项目结构

```
learning_rag/
│
├── data/
│   ├── raw/               ← 文档原料（丢 PDF/TXT 进来）
│   └── chroma_db/         ← Chroma 持久化目录（自动生成）
│
├── src/
│   ├── config.py           ← 所有配置收拢（chunk_size、模型名、API Key 等）
│   ├── loader.py           ← 文档读取（PDF / TXT）
│   ├── chunking.py         ← 切分策略（多策略 + 统一接口）
│   ├── embedding.py        ← 嵌入模型（BGE-M3 加载 + 推理）
│   ├── vector_store.py     ← Chroma 读写（索引 + 检索）
│   ├── retrieval.py        ← 检索逻辑（向量 / hybrid 等）
│   ├── llm.py              ← LLM 统一接口（DeepSeek / 百炼）
│   ├── pipeline.py         ← 串联：index() / query()
│   ├── prompts.py          ← Prompt 模板
│   └── cli.py              ← 命令行入口
│
├── notebooks/
│   ├── 01_加载与探索.ipynb       ← 文档解析后长什么样
│   ├── 02_Chunking实验.ipynb     ← 对比不同切分策略
│   └── 03_端到端管线.ipynb       ← 完整跑通 + 检索实验
│
├── .env                         ← API Keys
├── pyproject.toml               ← 依赖管理
└── README.md
```

### 模块依赖关系

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

依赖单向、无循环，每个文件可单独调试。

---

## 四、Chunking 线

### 4.1 问题本质

Chunking 在"每个 chunk 信息完整"和"chunk 足够聚焦"之间找平衡。

```
文档段落                          Chunk 序列
┌──────────────────┐     ┌──────────────────────┐
│ Attention 是一种  │     │ Chunk A: Attention是 │
│ 机制，它允许模型   │     │ 一种机制...          │
│ 在生成时关注输入   │ ──▶ ├──────────────────────┤
│ 序列的不同部分...  │     │ Chunk B: 它允许模型  │
│                   │     │ 在生成时关注输入序列...│
│ "与传统 RNN       │     ├──────────────────────┤
│ 不同..."          │     │ Chunk C: 与传统RNN   │
└──────────────────┘     │ 不同...               │
                         └──────────────────────┘
检索 "Attention是什么？" → 命中 A 还是 B？结果截然不同
```

### 4.2 三种切分策略

#### 方案 A：SentenceSplitter (按句/按字符)

```python
from llama_index.core.node_parser import SentenceSplitter
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=128,
    separator="。！？；\n\n",
    paragraph_separator="\n\n"
)
```

| 优点 | 缺点 |
|------|------|
| 简单、零额外依赖 | 对"伪句号"（3.14、file.txt）误切 |
| LlamaIndex 原生支持 | 无标点文本段无效 |

#### 方案 B：Jieba 分词感知

```
原始文本: "深度学习在自然语言处理中取得了显著成效"
    分词: "深度 学习 在 自然语言 处理 中 取得 了 显著 成效"

固定字符切:      Jieba 感知切:
"深度学习在自然语"    "深度学习在自然语言处理中"
"言处理中取得了显"    "取得了显著成效"
← 语义被切碎         ← 完整词边界
```

| 优点 | 缺点 |
|------|------|
| 中文保真度更高 | 多了 jieba 依赖 |
| 专业术语可加自定义词典 | 混合英文/代码场景更复杂 |

#### 方案 C：语义切分 (Semantic Chunking)

```
核心思路: 检测句子间 embedding 相似度的"突降点"

语义相似度曲线:
       ╲╱
        ╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱╱╲╱╲╱╲╱╲╱╲╱╲╱╲╱
        ─────────────────────────────────────────▶
        引言────▶背景────▶方法────▶实验────▶结论
                     ↑语义突降处 = 切分点
```

| 优点 | 缺点 |
|------|------|
| 理论上最准，自动检测主题边界 | 需要跑 embedding 才知道怎么切，计算量大 |
| 对内容连续的文档可能过切或欠切 | 阈值需要调试 |

### 4.3 Chunking 接口设计

```python
@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict      # source, page, heading, chunk_index, ...

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict) -> list[Chunk]: ...

class SentenceChunker(BaseChunker): ...
class JiebaChunker(BaseChunker): ...
class SemanticChunker(BaseChunker): ...

def get_chunker(name: str, **kwargs) -> BaseChunker:
    chunkers = {"sentence": SentenceChunker, "jieba": JiebaChunker, "semantic": SemanticChunker}
    return chunkers[name](**kwargs)
```

使用方式：

```python
for name in ["sentence", "jieba", "semantic"]:
    chunker = get_chunker(name, chunk_size=512)
    chunks = chunker.chunk(doc_text, {"source": "paper.pdf"})
    print(f"{name}: {len(chunks)} chunks")
```

### 4.4 Chunking 元数据结构

```python
chunk = {
    "text": "Attention 是一种机制...",
    "metadata": {
        "source":       "transformer_paper.pdf",
        "page":         3,
        "heading":      "3.1 Attention 机制",
        "chunk_index":  4,
        "total_chunks": 12,
    }
}
```

### 4.5 Chunking 实验设计

```
实验维度:
                    chunk_size
                 256────512────1024
                 │  ┌────┬────┐
    strategy     │  │    │    │
   sentence ─────┤  ├────┼────┤
   jieba    ─────┤  ├────┼────┤
   semantic ─────┘  └────┴────┘

                    overlap
                   0───64───128───256
                   ┌────┬────┬────┐
                   │    │    │    │
                   └────┴────┴────┘
```

核心对比实验：

| 实验 | strategy | chunk_size | overlap | 目的 |
|------|----------|------------|---------|------|
| E1 | sentence | 512 | 128 | 基线 |
| E2 | jieba | 512 | 128 | vs E1，看中文保真度 |
| E3 | semantic | 自动 | 0 | vs E1，看语义边界效果 |
| E4 | sentence | 256 | 64 | vs E1，看小 chunk 优劣 |
| E5 | sentence | 1024 | 256 | vs E1，看大 chunk 优劣 |

每个实验需要观察：
- chunk 数量差异
- 是否有语义被切碎的情况
- 对同一个问题的检索命中位置

---

## 五、检索管线

### 5.1 混合检索（Hybrid Search）

Chroma 原生只支持向量检索。解决方案：**双检索器 + RRF Fusion**。

```
         Query
           │
    ┌──────┴──────┐
    ▼              ▼
 Chroma           BM25
 (向量检索)       (关键词检索)
    │              │
    └──────┬───────┘
           ▼
    RRF Fusion
    (排名融合)
           │
           ▼
        Top-N
```

#### BM25 对中文的处理

```python
import jieba
from rank_bm25 import BM25Okapi

# 建立索引
tokenized_chunks = [list(jieba.cut(chunk.text)) for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)

# 检索
query_tokens = list(jieba.cut(query))
scores = bm25.get_scores(query_tokens)
```

#### RRF (Reciprocal Rank Fusion)

RRF 只关心排名而不关心分数（向量分数和 BM25 分数不在同一量级）：

```python
def _rrf_fusion(self, dense, sparse, k=60):
    scores = {}
    for rank, doc in enumerate(dense):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (rank + k)
    for rank, doc in enumerate(sparse):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (rank + k)
    return sorted(scores.items(), key=lambda x: -x[1])
```

#### alpha 融合权重

```
alpha=1.0  → 纯向量检索（语义匹配）
alpha=0.0  → 纯 BM25（关键词匹配）
alpha=0.5  → 各一半
alpha=0.7  → 偏向语义
```

| 检索方式 | 强项 | 弱项 |
|----------|------|------|
| 纯向量 (dense) | 语义匹配。"Attention 机制的原理" → 找到讲 Attention 的段落 | 专有名词/缩写（MHA、FFN）可能跑偏 |
| BM25 (sparse) | 关键词匹配。"MHA" → 找到含 MHA 的段落 | 同义词/不同表述（"多头注意力" vs "MHA"）找不到 |
| Hybrid | 两者互补，中文场景通常高 5-15% 召回 | 多一套索引，多一次检索 |

### 5.2 Reranker（优化阶段）

```
第一轮（粗筛）：hybrid top-20
第二轮（精排）：Reranker 对 20 个候选逐一打分 → top-5
```

LlamaIndex 集成方式：

```python
from llama_index.postprocessor import SentenceTransformerRerank
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-v2-m3",
    top_k=5
)
```

| Reranker 模型 | 大小 | 中文效果 |
|---------------|------|----------|
| BAAI/bge-reranker-v2-m3 | ~2.2GB | 很好 |
| BAAI/bge-reranker-v2-minicpm-layerwise | ~500MB | 好 |
| cross-encoder/ms-marco-MiniLM | ~80MB | 英文优先，中文一般 |

**建议**：先不加 Reranker，跑通基础管线后作为优化阶段再加。

### 5.3 LLM 层设计

统一接口，按配置切换 DeepSeek / 百炼：

```python
class LLM:
    def __init__(self, provider="deepseek"):
        self.provider = provider
        if provider == "deepseek":
            self.client = OpenAI(api_key=..., base_url="https://api.deepseek.com")
            self.model = "deepseek-chat"
        elif provider == "bailian":
            self.client = OpenAI(api_key=..., base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.model = "qwen-max"

    def query(self, prompt: str, **kwargs) -> str: ...
```

#### DeepSeek vs 百炼差异

| 维度 | DeepSeek | 百炼 (Qwen) |
|------|----------|-------------|
| 上下文 | 128K | 32K-128K（看模型） |
| 价格 | ~¥1/M tokens | ~¥2/M tokens |
| 中文 | 好 | 优秀 |
| 速度 | 较快 | 中等 |

#### 自动路由策略

```python
def query_with_context(self, question, context):
    if len(question) < 20 and len(context) < 2000:
        return self.query(prompt, provider="deepseek")  # 简单→便宜
    else:
        return self.query(prompt, provider="bailian")   # 复杂→中文更强
```

### 5.4 Prompt 模板

```python
# src/prompts.py

QA_TEMPLATE = """你是一个知识库问答助手。

请基于以下文档内容回答问题。如果文档中没有足够信息，请明确说"文档中未找到相关信息"。

文档内容：
{context}

问题：{question}

要求：
- 回答要尽可能准确，基于文档内容
- 如果回答引用文档内容，请标注来源文件名
- 不要添加文档中不存在的信息

回答："""

SUMMARY_TEMPLATE = """请总结以下文档内容的核心要点。

文档内容：
{context}

要求：
- 用要点形式列出核心结论
- 每个要点一句话
- 保持原文的关键信息不丢失

总结："""
```

模板选择可自动检测问题类型：

```python
def detect_question_type(question: str) -> str:
    if any(w in question for w in ["代码", "实现", "函数", "代码示例"]):
        return "code"
    if any(w in question for w in ["总结", "概括", "归纳", "要点"]):
        return "summary"
    return "qa"
```

---

## 六、交叉实验（Chunking × 检索）

### 6.1 联动效应

```
chunk_size=256              chunk_size=1024
    │                            │
    ▼                            ▼
┌────────────┐              ┌────────────┐
│ 24 chunks  │              │  6 chunks  │
│ 每个聚焦    │              │ 每个信息   │
│ 但信息可能  │              │ 完整但     │
│ 不完整     │              │ 噪音多     │
└──────┬────┘              └──────┬────┘
       │                          │
       ▼                          ▼
检索 "注意力机制"          检索 "注意力机制"
  → 命中 chunk 3:           → 命中 chunk 2:
    "注意力机制定义..."          整节"注意力机制与发展"
  ↑ 精准                     ↑ 信息更多但也带回不相关细节
```

### 6.2 交叉实验矩阵

```
固定问题 + 固定检索策略，变化 chunking 策略

问题: "Transformer 和 RNN 的核心区别是什么？"
检索: hybrid top-5
chunking: [sentence/512, jieba/512, semantic]

对比:
  - 三个策略返回的 chunk 内容有什么不同？
  - 哪个 chunk 包含了最完整的对比信息？
  - LLM 用这些 chunk 生成的答案质量差异？
```

### 6.3 检索实验矩阵

```
固定 chunking + 固定问题，变化检索策略

                   检索方式
              ┌──────────────────────┐
              │  纯向量  │  hybrid  │
    ┌─────────┼──────────┼──────────┤
    │ 问题A   │ 结果A1   │ 结果A2   │
    │ (具体)  │          │          │
    ├─────────┼──────────┼──────────┤
    │ 问题B   │ 结果B1   │ 结果B2   │
    │ (抽象)  │          │          │
    ├─────────┼──────────┼──────────┤
    │ 问题C   │ 结果C1   │ 结果C2   │
    │ (术语)  │          │          │
    └─────────┴──────────┴──────────┘
```

### 6.4 测试问题

```
Q1（具体）: "什么是注意力机制？"          → 期望：准确定义
Q2（抽象）: "Transformer 的核心创新是什么？" → 期望：综合多个段落
Q3（术语）: "MHA 和 FFN 分别是什么？"       → 期望：术语解释
```

---

## 七、分阶段学习路线

### 阶段 1：搭框架

```
目标: 跑通"文档→chunk→向量→存储"全流程

涉及文件:
  config.py → loader.py → chunking.py → embedding.py → vector_store.py

验证方式:
  notebook 01_加载与探索.ipynb
  notebook 02_Chunking实验.ipynb
```

### 阶段 2：Chunking 实验

```
目标: 理解 chunk_size / overlap / strategy 的影响

实验内容:
  - 对比 E1~E5
  - 找到适合你文档集的策略
  - 理解中文切分的特殊性
```

### 阶段 3：检索管线

```
目标: 跑通完整 RAG 管线

涉及:
  retrieval.py（纯向量 → hybrid）
  llm.py（DeepSeek / 百炼）
  prompts.py

验证方式:
  notebook 03_端到端管线.ipynb
  notebook 额外：检索实验矩阵
```

### 阶段 4：端到端 + 交叉实验

```
目标: 系统对比，理解各环节如何联动

实验内容:
  - Chunking × 检索 交叉实验
  - 调优参数
  - 分析失败案例
```

### 阶段 5：进阶（可选）

```
- 添加 Reranker
- Query 改写 / 扩展
- 多轮对话 / 历史记忆
- Web UI（Gradio / Streamlit）
- OCR 管线（PaddleOCR 兜底扫描件）
```

---

## 八、技术栈总览

| 组件 | 技术选型 | 备注 |
|------|----------|------|
| 框架 | LlamaIndex | Python |
| 嵌入模型 | BAAI/bge-m3 | 本地运行，~2.2GB |
| 向量库 | Chroma | 持久化到本地目录 |
| BM25 | rank_bm25 + jieba | 中文分词后建索引 |
| Reranker | BAAI/bge-reranker-v2-m3 | 优化阶段引入 |
| LLM | DeepSeek API + 百炼 API | OpenAI 兼容接口 |
| PDF 解析 | PyMuPDF (fitz) | 电子版 PDF |
| OCR | PaddleOCR | 扫描件兜底 |
| 依赖管理 | pip / poetry / uv | 未定 |

---

> 本文档是探索阶段的产出，后续实现时可据此创建 OpenSpec change proposal，将架构决策、设计细节、任务拆解正式化。
