## ADDED Requirements

### Requirement: Unified LLM interface
The system SHALL provide a single `LLM` class that abstracts over DeepSeek API and Alibaba Cloud Bailian API, both via OpenAI-compatible interface.

#### Scenario: DeepSeek query
- **WHEN** provider is `deepseek` and a prompt is submitted
- **THEN** the system SHALL call `https://api.deepseek.com` with the configured API key and model name, and return the text response

#### Scenario: Bailian query
- **WHEN** provider is `bailian` and a prompt is submitted
- **THEN** the system SHALL call `https://dashscope.aliyuncs.com/compatible-mode/v1` with the configured API key and model name, and return the text response

### Requirement: Configurable provider and model
The LLM provider, model name, API key, temperature, and other parameters SHALL be configurable via `config.py` and overridable per call.

#### Scenario: Override model per call
- **WHEN** a query call specifies `model="deepseek-chat"` and `temperature=0.7`
- **THEN** those values SHALL override the defaults for that call only

### Requirement: Auto-routing by task complexity
The system SHALL support automatic routing: simple queries (short question, small context) go to DeepSeek, complex queries go to Bailian.

#### Scenario: Auto-route simple query
- **WHEN** the question is under 20 characters and context under 2000 tokens
- **THEN** the system SHALL route to DeepSeek by default

#### Scenario: Auto-route complex query
- **WHEN** the question is over 20 characters or context over 2000 tokens
- **THEN** the system SHALL route to Bailian by default

### Requirement: Error handling
The system SHALL handle API errors (timeout, rate limit, authentication failure) gracefully, logging the error and returning a descriptive message.

#### Scenario: API authentication failure
- **WHEN** the API key is invalid or missing
- **THEN** the system SHALL log an error and return a message indicating authentication failure without crashing
