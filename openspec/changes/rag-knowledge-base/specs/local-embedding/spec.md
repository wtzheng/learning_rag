## ADDED Requirements

### Requirement: Load BGE-M3 embedding model locally
The system SHALL load BAAI/bge-m3 via HuggingFaceEmbedding (sentence-transformers) for local inference.

#### Scenario: Model loading
- **WHEN** the system initializes the embedding module
- **THEN** it SHALL load BAAI/bge-m3 from HuggingFace cache or download if not present

#### Scenario: Embed text chunks
- **WHEN** given a list of text strings
- **THEN** the system SHALL return their corresponding vector embeddings as numpy arrays

### Requirement: Configurable embedding model
The embedding model name SHALL be configurable in `config.py` so it can be swapped without code changes.

#### Scenario: Swap embedding model
- **WHEN** the model name in config is changed to another HuggingFace model (e.g., `BAAI/bge-small-zh-v1.5`)
- **THEN** the system SHALL load and use the new model without other code changes

### Requirement: Normalize embeddings
All output embeddings SHALL be L2-normalized for cosine similarity compatibility.

#### Scenario: Normalization
- **WHEN** embedding vectors are produced
- **THEN** each vector SHALL have L2 norm of 1.0 (within floating-point tolerance)
