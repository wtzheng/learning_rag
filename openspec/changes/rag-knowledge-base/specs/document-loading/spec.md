## ADDED Requirements

### Requirement: Load documents from local directory
The system SHALL scan a configurable local directory (`data/raw/`) and load all supported document files.

#### Scenario: Load PDF files
- **WHEN** a `.pdf` file exists in the data directory
- **THEN** the system extracts its text content using PyMuPDF and attaches metadata (filename, page count, file path)

#### Scenario: Load TXT files
- **WHEN** a `.txt` file exists in the data directory
- **THEN** the system reads its full text content and attaches metadata (filename, file path)

#### Scenario: Skip unsupported file types
- **WHEN** a file with an unsupported extension exists in the data directory
- **THEN** the system SHALL skip it with a warning log and continue loading other files

### Requirement: Attach source metadata
Every loaded document SHALL carry metadata including source filename, file path, page number (for PDFs), and document index.

#### Scenario: Metadata structure
- **WHEN** a document is loaded
- **THEN** its metadata SHALL contain `source` (filename), `file_path` (absolute path), and for PDFs also `page_label`

### Requirement: Handle non-textual PDFs gracefully
For scanned/image-based PDFs where text extraction fails, the system SHALL detect the failure and report which pages could not be extracted. (Full OCR is out of scope for initial version.)

#### Scenario: Text extraction failure
- **WHEN** PyMuPDF extracts fewer than 50 characters from a PDF page
- **THEN** the system SHALL log a warning with the filename and page number, and include the page as empty text with a flag in metadata
