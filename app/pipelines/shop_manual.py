"""Shop manual ingestion pipeline with visual status updates."""

from __future__ import annotations

import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

import pandas as pd

try:  # Optional heavy imports guarded for UI-only scenarios
    import fitz  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore

try:  # Azure Document Intelligence optional dependency
    from azure.ai.documentintelligence.models import DocumentAnalysisFeature  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    DocumentAnalysisFeature = None  # type: ignore

try:  # LangChain integrations
    from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader  # type: ignore
    from langchain_community.vectorstores import AzureSearch  # type: ignore
    from langchain_openai import AzureOpenAIEmbeddings  # type: ignore
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    AzureAIDocumentIntelligenceLoader = None  # type: ignore
    AzureSearch = None  # type: ignore
    AzureOpenAIEmbeddings = None  # type: ignore
    RecursiveCharacterTextSplitter = None  # type: ignore


StepCallback = Callable[[str, str, str, float | None], None]
"""Callback signature: (step_key, state, message, progress_fraction)."""


@dataclass
class ShopManualConfig:
    """Runtime configuration for the shop manual pipeline."""

    search_endpoint: str
    search_api_key: str
    index_name: str
    openai_endpoint: str
    openai_deployment: str
    openai_api_key: str
    openai_api_version: str | None = None
    document_intelligence_endpoint: str | None = None
    document_intelligence_key: str | None = None
    document_intelligence_model: str = "prebuilt-layout"
    temp_dir: Path | None = None
    recreate_index: bool = False
    chunk_size: int = 500
    chunk_overlap: int = 40
    max_workers: int = 4
    zoom_x: float = 2.0
    zoom_y: float = 2.0
    retry_attempts: int = 3


@dataclass
class ProcessingResult:
    """Represents the outcome of a processed PDF file."""

    pdf_path: Path
    image_files: list[Path]
    total_pages: int
    processed_pages: int
    indexed_chunks: int


class ShopManualPipeline:
    """Pipeline that converts PDF shop manuals into Azure AI Search documents."""

    def __init__(self, config: ShopManualConfig) -> None:
        self.config = config
        self._validate_dependencies()
        self._temp_dir = config.temp_dir or Path(tempfile.mkdtemp(prefix="shop_manual_"))
        self._vector_store = self._create_vector_store()
        self._splitter = self._create_splitter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        pdf_paths: Sequence[Path],
        mapping_table: pd.DataFrame | None,
        status_callback: StepCallback,
    ) -> list[ProcessingResult]:
        """Execute the pipeline for ``pdf_paths`` using ``mapping_table`` metadata."""

        results: list[ProcessingResult] = []
        for pdf_index, pdf_path in enumerate(pdf_paths, start=1):
            status_callback("prepare", "active", f"Staging {pdf_path.name}", (pdf_index - 1) / max(len(pdf_paths), 1))
            pdf_temp_dir = self._prepare_output_dir(pdf_path)
            image_files = self._convert_pdf_to_images(pdf_path, pdf_temp_dir, status_callback)
            total_pages = len(image_files)

            if total_pages == 0:
                status_callback("prepare", "error", f"No pages detected in {pdf_path.name}", None)
                continue

            status_callback("prepare", "done", f"{pdf_path.name}: {total_pages} pages ready", pdf_index / max(len(pdf_paths), 1))
            status_callback("convert", "active", "Running document intelligence OCR", 0.1)

            publication_number = self._resolve_publication_number(pdf_path, mapping_table)

            indexed_chunks = self._process_images(
                image_files,
                pdf_path,
                publication_number,
                status_callback,
            )

            status_callback(
                "index",
                "done",
                f"Indexed {indexed_chunks} chunks for {pdf_path.name}",
                1.0,
            )
            status_callback(
                "complete",
                "done",
                f"{pdf_path.name} served hot!",
                1.0,
            )

            results.append(
                ProcessingResult(
                    pdf_path=pdf_path,
                    image_files=image_files,
                    total_pages=total_pages,
                    processed_pages=total_pages,
                    indexed_chunks=indexed_chunks,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_dependencies(self) -> None:
        missing: list[str] = []
        if fitz is None:
            missing.append("PyMuPDF (install 'pymupdf')")
        if AzureAIDocumentIntelligenceLoader is None:
            missing.append("langchain-community + azure-ai-documentintelligence")
        if AzureSearch is None or AzureOpenAIEmbeddings is None:
            missing.append("langchain-openai + langchain-community")
        if missing:
            raise RuntimeError(
                "Missing optional dependencies required for the shop manual pipeline: "
                + ", ".join(missing)
            )
        if not self.config.search_api_key:
            raise ValueError("Search API key is required for the shop manual pipeline")
        if not self.config.openai_api_key:
            raise ValueError("Azure OpenAI API key is required for the shop manual pipeline")
        if not self.config.document_intelligence_endpoint or not self.config.document_intelligence_key:
            raise ValueError(
                "Azure Document Intelligence endpoint and key are required for the shop manual pipeline"
            )

    def _create_vector_store(self):  # type: ignore[no-untyped-def]
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=self.config.openai_deployment,
            azure_endpoint=self.config.openai_endpoint,
            api_key=self.config.openai_api_key,
            api_version=self.config.openai_api_version or "2024-02-15-preview",
        )
        return AzureSearch(
            azure_search_endpoint=self.config.search_endpoint,
            azure_search_key=self.config.search_api_key,
            index_name=self.config.index_name,
            embedding_function=embeddings.embed_query,
            recreate_index=self.config.recreate_index,
        )

    def _create_splitter(self):  # type: ignore[no-untyped-def]
        return RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    @contextmanager
    def _managed_doc(self, pdf_path: Path):  # type: ignore[no-untyped-def]
        doc = fitz.open(str(pdf_path))  # type: ignore[union-attr]
        try:
            yield doc
        finally:
            doc.close()

    def _prepare_output_dir(self, pdf_path: Path) -> Path:
        folder = self._temp_dir / pdf_path.stem
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _convert_pdf_to_images(
        self,
        pdf_path: Path,
        output_folder: Path,
        status_callback: StepCallback,
    ) -> list[Path]:
        image_files: list[Path] = []
        with self._managed_doc(pdf_path) as doc:
            matrix = fitz.Matrix(self.config.zoom_x, self.config.zoom_y)  # type: ignore[union-attr]
            total_pages = doc.page_count
            status_callback("convert", "active", f"Baking {total_pages} pages", 0.2)

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._render_page, doc, page_index, output_folder, matrix): page_index
                    for page_index in range(total_pages)
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        image_files.append(result)

        image_files.sort()
        status_callback("convert", "done", f"Prepared {len(image_files)} page images", 0.45)
        return image_files

    def _render_page(self, doc, page_index: int, output_folder: Path, matrix):  # type: ignore[no-untyped-def]
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=matrix)
                output_file = output_folder / f"page-{page_index + 1}.png"
                pix.save(str(output_file))
                if output_file.stat().st_size == 0:
                    raise IOError("Rendered image is empty")
                return output_file
            except Exception:  # pragma: no cover - defensive
                if attempt == self.config.retry_attempts:
                    return None
                time.sleep(0.2 * attempt)
        return None

    def _resolve_publication_number(
        self,
        pdf_path: Path,
        mapping_table: pd.DataFrame | None,
    ) -> str | None:
        if mapping_table is None or "publication_number" not in mapping_table.columns:
            return None
        file_stem = pdf_path.stem
        for value in mapping_table["publication_number"].dropna().astype(str):
            if value and value in file_stem:
                return value
        return None

    def _process_images(
        self,
        image_files: Sequence[Path],
        pdf_path: Path,
        publication_number: str | None,
        status_callback: StepCallback,
    ) -> int:
        total = len(image_files)
        indexed_chunks = 0
        status_callback("ocr", "active", "Quality check in progress", 0.6)

        for page_index, image_file in enumerate(image_files, start=1):
            docs = self._process_single_page(
                image_file=image_file,
                file_stem=pdf_path.stem,
                publication_number=publication_number,
                page_number=page_index,
            )
            if docs:
                self._vector_store.add_documents(documents=docs)  # type: ignore[no-untyped-call]
                indexed_chunks += len(docs)
            progress = 0.6 + (page_index / max(total, 1)) * 0.3
            status_callback(
                "ocr",
                "active" if page_index < total else "done",
                f"Quality check page {page_index}/{total}",
                progress,
            )

        if indexed_chunks == 0:
            status_callback("index", "error", f"No chunks were indexed for {pdf_path.name}", None)
        else:
            status_callback("index", "active", f"Out for delivery: {indexed_chunks} chunks", 0.95)
        return indexed_chunks

    def _process_single_page(
        self,
        image_file: Path,
        file_stem: str,
        publication_number: str | None,
        page_number: int,
    ) -> List[object]:
        if AzureAIDocumentIntelligenceLoader is None:
            return []
        loader = AzureAIDocumentIntelligenceLoader(
            file_path=str(image_file),
            api_key=self.config.document_intelligence_key,
            api_endpoint=self.config.document_intelligence_endpoint,
            api_model=self.config.document_intelligence_model,
            mode="page",
            analysis_features=[DocumentAnalysisFeature.OCR_HIGH_RESOLUTION]
            if DocumentAnalysisFeature
            else None,
        )
        docs = loader.load_and_split(self._splitter)
        for doc in docs:
            metadata = getattr(doc, "metadata", {})
            metadata.update(
                {
                    "page_no": page_number,
                    "pdf_name": file_stem,
                    "publication_number": publication_number,
                }
            )
            doc.metadata = metadata
        return docs

    def cleanup(self) -> None:
        """Remove temporary working directory."""

        if self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        return False

