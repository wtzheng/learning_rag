## ADDED Requirements

### Requirement: Persist vectors to Chroma
The system SHALL store document chunks, their embeddings, and metadata in a Chroma collection persisted to disk at `data/chroma_db/`.

#### Scenario: Store chunks
- **WHEN** chunks with embeddings and metadata are passed to the vector store
- **THEN** they SHALL be added to a Chroma collection and persisted to disk

#### Scenario: Collection name
- **WHEN** initializing the vector store
- **THEN** the Chroma collection SHALL be named by a configurable value (default: `rag_knowledge_base`)

### Requirement: Similarity search
The vector store SHALL support cosine similarity search, returning top-k results with scores.

#### Scenario: Basic similarity search
- **WHEN** a query embedding is provided with `top_k=5`
- **THEN** the system SHALL return the 5 most similar chunks ranked by cosine distance, each with its metadata and similarity score

### Requirement: Incremental indexing
The system SHALL support adding new documents without re-indexing the entire collection.

#### Scenario: Add new documents
- **WHEN** new chunks are added to an existing Chroma collection
- **THEN** existing data SHALL remain unchanged and new chunks SHALL be appended

### Requirement: Clear and rebuild
The system SHALL provide a way to clear the entire vector store and re-index from scratch.

#### Scenario: Clear collection
- **WHEN** a reset command is issued
- **THEN** all chunks and vectors SHALL be removed from the collection and the persistence directory
