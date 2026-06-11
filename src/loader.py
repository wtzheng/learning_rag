"""
Document loader: reads PDF (via PyMuPDF) and TXT files,
returning a unified Document structure with text + metadata.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

MIN_TEXT_THRESHOLD = 50  # characters; below this, a PDF page is flagged as low-text


@dataclass
class Document:
    """A single loaded document (one file = one Document)."""
    text: str
    metadata: dict = field(default_factory=dict)


def load_pdf(path: Path) -> Document:
    """Load a PDF file using PyMuPDF, extract text page by page."""
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    pages_text: list[str] = []
    low_text_pages: list[int] = []
    total_chars = 0

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        chars = len(text.strip())
        total_chars += chars
        if chars < MIN_TEXT_THRESHOLD:
            low_text_pages.append(page_num)
        pages_text.append(text)

    full_text = "\n\n".join(pages_text)
    doc.close()

    metadata = {
        "source": path.name,
        "file_path": str(path.absolute()),
        "type": "pdf",
        "pages": len(pages_text),
        "total_chars": total_chars,
    }
    if low_text_pages:
        metadata["low_text_pages"] = low_text_pages
        logger.warning(
            "PDF '%s' has low-text pages: %s. These may be scanned images.",
            path.name, low_text_pages,
        )

    return Document(text=full_text, metadata=metadata)


def load_txt(path: Path) -> Document:
    """Load a plain text file."""
    text = path.read_text(encoding="utf-8")
    metadata = {
        "source": path.name,
        "file_path": str(path.absolute()),
        "type": "txt",
        "total_chars": len(text),
    }
    return Document(text=text, metadata=metadata)


def load_document(path: Path) -> Optional[Document]:
    """Load a single document by path, dispatching on file extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".txt":
        return load_txt(path)
    else:
        logger.warning("Unsupported file type: %s (skipping)", path.name)
        return None


def load_directory(directory: str = "data/raw") -> list[Document]:
    """
    Scan a directory for supported documents and load them all.

    Args:
        directory: Path to the data directory.

    Returns:
        List of loaded Document objects.
    """
    data_dir = Path(directory)
    if not data_dir.exists():
        logger.warning("Data directory '%s' does not exist. Creating it.", directory)
        data_dir.mkdir(parents=True, exist_ok=True)
        return []

    documents: list[Document] = []
    # Sort for deterministic ordering
    file_paths = sorted(data_dir.iterdir())

    for path in file_paths:
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.info("Skipping unsupported file: %s", path.name)
            continue

        doc = load_document(path)
        if doc is not None:
            documents.append(doc)
            logger.info("Loaded: %s (%d chars)", path.name, doc.metadata["total_chars"])

    logger.info("Loaded %d document(s) from '%s'", len(documents), directory)
    return documents
