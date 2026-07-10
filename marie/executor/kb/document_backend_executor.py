"""Document Backend Executor for RAG workflows.

Format-aware document text extraction executor. Routes to the appropriate
backend based on document format and handles both parsed and frames modes.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

from marie import requests
from marie.executor.marie_executor import MarieExecutor
from marie.logging_core.logger import MarieLogger

logger = MarieLogger("marie.executor.kb.document_backend_executor").logger


class DocumentBackendExecutor(MarieExecutor):
    """
    Format-aware document text extraction executor.

    Routes to appropriate backend based on document format:
    - Parsed mode: Returns text directly (DOCX, XLSX, HTML, etc.)
    - Frames mode: Routes through OCR pipeline (PDF, images, etc.)

    Endpoints:
        /extract: Extract text from document

    Example workflow configuration:
        ```yaml
        - name: document_backend
          uses: DocumentBackendExecutor
          with:
            ocr_fallback: true
            ocr_executor: "extract_executor://document/extract"
        ```
    """

    def __init__(
        self,
        ocr_fallback: bool = True,
        ocr_executor: str = "extract_executor://document/extract",
        **kwargs,
    ):
        """
        Initialize DocumentBackendExecutor.

        Args:
            ocr_fallback: Whether to use OCR for frames mode documents.
            ocr_executor: Executor endpoint for OCR processing.
            **kwargs: Additional MarieExecutor arguments.
        """
        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info("Initializing DocumentBackendExecutor")

        self._ocr_fallback = ocr_fallback
        self._ocr_executor = ocr_executor
        self._storage_connected = False

    def _ensure_storage(self) -> None:
        """Ensure storage manager is connected."""
        if self._storage_connected:
            return

        from marie.storage import StorageManager

        StorageManager.ensure_connection(silence_exceptions=False)
        self._storage_connected = True

    @requests(on="/extract")
    async def extract(
        self, parameters: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Extract text from a document.

        Args:
            parameters: Dictionary containing:
                - uri: Document URI (s3://, file://)
                - ref_id: Reference document ID
                - ref_type: Document type classification
                - ocr_fallback: Override default OCR fallback setting

        Returns:
            Dictionary with extracted text chunks:
                - chunks: List of text chunks with metadata
                - mode: "parsed" or "ocr"
                - pages: Number of pages processed
        """
        if parameters is None:
            parameters = {}

        uri = parameters.get("uri")
        if not uri:
            raise ValueError("uri parameter is required")

        ref_id = parameters.get("ref_id", "unknown")
        ref_type = parameters.get("ref_type", "document")
        use_ocr = parameters.get("ocr_fallback", self._ocr_fallback)

        self.logger.info(f"Extracting text from {uri} (ref_id={ref_id})")

        self._ensure_storage()

        # Download to temp file
        from marie.storage import StorageManager

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=self._get_extension(uri)
        ) as tmp:
            temp_path = tmp.name

        try:
            StorageManager.read_to_file(uri, temp_path, overwrite=True)

            # Detect format and get backend
            from marie.backend import get_backend
            from marie.utils.docs import get_document_type

            doc_type = get_document_type(temp_path)
            self.logger.info(f"Detected document type: {doc_type}")

            backend = get_backend(doc_type)
            result = backend.convert(temp_path)

            if result["mode"] == "parsed":
                # Direct text extraction
                return self._parsed_to_chunks(result, ref_id, ref_type)
            else:
                # Frames mode - needs OCR
                if use_ocr:
                    return await self._frames_to_chunks(
                        result["frames"], ref_id, ref_type, parameters
                    )
                else:
                    raise ValueError(
                        f"OCR required for {doc_type} but ocr_fallback=False"
                    )

        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _get_extension(self, uri: str) -> str:
        """Extract file extension from URI."""
        path = uri.split("?")[0]  # Remove query params
        _, ext = os.path.splitext(path)
        return ext if ext else ".tmp"

    def _parsed_to_chunks(
        self, result: Dict[str, Any], ref_id: str, ref_type: str
    ) -> Dict[str, Any]:
        """Convert parsed backend output to text chunks with metadata."""
        chunks = []

        for page_idx, page in enumerate(result.get("results", [])):
            for line in page.get("lines", []):
                text = line.get("text", "")
                if not text.strip():
                    continue

                chunks.append(
                    {
                        "text": text,
                        "bbox": line.get("bbox"),
                        "page": page_idx,
                        "confidence": 1.0,  # Parsed = perfect confidence
                        "ref_id": ref_id,
                        "ref_type": ref_type,
                    }
                )

        return {
            "chunks": chunks,
            "mode": "parsed",
            "pages": result.get("pages", len(result.get("results", []))),
            "ref_id": ref_id,
            "ref_type": ref_type,
        }

    async def _frames_to_chunks(
        self,
        frames: List,
        ref_id: str,
        ref_type: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route frames through OCR pipeline."""
        self.logger.info(f"Processing {len(frames)} frames through OCR")

        # Use the existing extract pipeline/executor
        # This delegates to the OCR infrastructure
        from marie.pipe.extract_pipeline import ExtractPipeline

        pipeline = ExtractPipeline()

        # Process frames through OCR
        ocr_results = await pipeline.execute(
            frames,
            parameters={
                "ref_id": ref_id,
                "ref_type": ref_type,
                **parameters,
            },
        )

        # Convert OCR results to chunks
        chunks = []
        for page_idx, page_result in enumerate(ocr_results.get("pages", [])):
            for word in page_result.get("words", []):
                text = word.get("text", "")
                if not text.strip():
                    continue

                chunks.append(
                    {
                        "text": text,
                        "bbox": word.get("bbox"),
                        "page": page_idx,
                        "confidence": word.get("confidence", 0.0),
                        "ref_id": ref_id,
                        "ref_type": ref_type,
                    }
                )

        return {
            "chunks": chunks,
            "mode": "ocr",
            "pages": len(frames),
            "ref_id": ref_id,
            "ref_type": ref_type,
        }

    @requests(on="/health")
    async def health(self, **kwargs) -> Dict[str, Any]:
        """Health check endpoint."""
        return {"status": "healthy", "executor": "DocumentBackendExecutor"}
