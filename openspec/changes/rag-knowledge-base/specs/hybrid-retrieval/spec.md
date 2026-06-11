## ADDED Requirements

### Requirement: Dense retrieval via Chroma
The system SHALL retrieve candidate chunks using Chroma's built-in vector similarity search.

#### Scenario: Dense retrieval
- **WHEN** a query embedding is submitted
- **THEN** the system SHALL return the top-N most semantically similar chunks from Chroma

### Requirement: Sparse retrieval via BM25
The system SHALL build and query a BM25 index over all chunks using jieba tokenization for Chinese text.

#### Scenario: BM25 index construction
- **WHEN** chunks are indexed
- **THEN** each chunk SHALL be tokenized with jieba and added to a BM25Okapi index

#### Scenario: BM25 retrieval
- **WHEN** a query is submitted
- **THEN** the query SHALL be tokenized with jieba and scored against the BM25 index, returning top-N results

### Requirement: Hybrid fusion via RRF
The system SHALL fuse dense and sparse results using Reciprocal Rank Fusion (RRF) with configurable k constant (default: 60).

#### Scenario: RRF fusion
- **WHEN** both dense and sparse retrieval results are available
- **THEN** the system SHALL compute RRF scores, merge and deduplicate, and return a unified ranking

#### Scenario: Configurable alpha
- **WHEN** the fusion strategy is configured with `alpha=1.0`
- **THEN** results SHALL be purely from dense retrieval (BM25 results ignored)

### Requirement: Reranker support (optional)
The system SHALL support an optional reranking step via `BAAI/bge-reranker-v2-m3` after hybrid retrieval, selectable via configuration.

#### Scenario: Reranker enabled
- **WHEN** reranker is enabled in config and hybrid results are available
- **THEN** the system SHALL rerank the candidate chunks and return the top-k after reranking
