## ADDED Requirements

### Requirement: End-to-end query pipeline
The system SHALL orchestrate the full RAG query pipeline: embed query → retrieve chunks → assemble prompt → generate answer.

#### Scenario: Full RAG query
- **WHEN** a user submits a question
- **THEN** the system SHALL embed it, retrieve top-k chunks from Chroma (or hybrid retriever), build a prompt with context, send to LLM, and return the answer

### Requirement: Multiple prompt templates
The system SHALL support at least three prompt templates: QA (default), Summary, and Code Explanation. Template selection SHALL be automatic based on question keywords, or manually overridable.

#### Scenario: QA template
- **WHEN** the question does not contain summary or code keywords
- **THEN** the system SHALL use the QA template which instructs the LLM to answer based on context and cite sources

#### Scenario: Summary template
- **WHEN** the question contains keywords like "总结" or "概括"
- **THEN** the system SHALL use the Summary template which instructs the LLM to output bullet-point summaries

### Requirement: Source citation
The system SHALL instruct the LLM to cite source filenames when referencing specific chunks.

#### Scenario: Answer with citation
- **WHEN** the LLM uses information from a specific chunk in its answer
- **THEN** the system SHALL encourage it to include the source filename as a citation

### Requirement: No hallucination guard
The system SHALL instruct the LLM to state "文档中未找到相关信息" when the retrieved context does not contain the answer.

#### Scenario: Insufficient context
- **WHEN** the retrieved chunks do not contain information relevant to the question
- **THEN** the LLM SHALL respond that the information was not found in the documents, rather than fabricating an answer
