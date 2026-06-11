## ADDED Requirements

### Requirement: Support multiple chunking strategies
The system SHALL provide at least three chunking strategies: SentenceSplitter (default), Jieba-aware chunking, and semantic chunking.

#### Scenario: SentenceSplitter chunking
- **WHEN** using SentenceSplitter with `chunk_size=512` and `chunk_overlap=128`
- **THEN** the document text SHALL be split into chunks with Chinese sentence boundaries respected (。！？；), each chunk not exceeding `chunk_size`, and adjacent chunks overlapping by `chunk_overlap` characters

#### Scenario: Jieba-aware chunking
- **WHEN** using Jieba-aware chunking
- **THEN** the text SHALL first be segmented by jieba, then grouped into chunks at word boundaries so that no word is split across chunks

#### Scenario: Semantic chunking
- **WHEN** using semantic chunking
- **THEN** the system SHALL compute embedding similarity between adjacent sentences and detect semantic shift boundaries as split points

### Requirement: Configurable chunk parameters
Chunk size and overlap SHALL be configurable via `config.py` and overridable per call.

#### Scenario: Override chunk parameters
- **WHEN** a chunker is instantiated with custom `chunk_size=256` and `chunk_overlap=64`
- **THEN** all output chunks SHALL respect the overridden values

### Requirement: Chunk metadata attachment
Every chunk SHALL carry metadata: source document name, chunk index, total chunk count, page number (if available).

#### Scenario: Chunk metadata
- **WHEN** a chunk is created
- **THEN** its metadata SHALL include `source`, `chunk_index`, `total_chunks`, and optionally `page_label`
