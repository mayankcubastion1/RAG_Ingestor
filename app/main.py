"""Streamlit front-end for enterprise RAG ingestion."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from dotenv import load_dotenv

if __package__ in (None, ""):
    # Allow running ``streamlit run app/main.py`` without installing the package.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

try:  # Optional visualization dependency
    from sklearn.decomposition import PCA
except ImportError:  # pragma: no cover - optional dependency
    PCA = None
from azure.core.exceptions import AzureError  # type: ignore
from azure.search.documents import SearchClient  # type: ignore
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient  # type: ignore
from azure.storage.blob import BlobServiceClient  # type: ignore

from app.azure_search.aliases import AliasOptions, swap_alias
from app.azure_search.index_schema import IndexSchemaOptions, VectorConfig, build_index_schema
from app.azure_search.indexers import DataSourceOptions, IndexerOptions, ensure_data_source, ensure_indexer, run_indexer
from app.azure_search.push_pipeline import PushIngestionPipeline, PushPipelineConfig, collect_chunks_from_text
from app.azure_search.rbac import get_default_credential, resolve_search_credential
from app.azure_search.skillsets import SkillsetOptions, build_skillset, build_skillset_payload
from app.chunking.textsplit_chunker import TextSplitOptions
from app.embeddings.azure_openai import EmbeddingConfig
from app.pipelines import ShopManualConfig, ShopManualPipeline
from app.utils import hashing
from app.utils.logging import configure_logging, get_logger
from app.utils.validators import require_non_empty

configure_logging()
_LOGGER = get_logger(__name__)

load_dotenv()

st.set_page_config(page_title="RAG Ingestor", layout="wide")
st.title("Enterprise RAG Ingestor for Azure AI Search")

st.markdown(
    """
    <style>
    .status-tracker {display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 1rem;}
    .status-step {background: #f6f6f9; border-radius: 14px; padding: 0.8rem 1rem; min-width: 180px;
                  box-shadow: 0 4px 10px rgba(17, 17, 26, 0.08); border: 1px solid #ececf1; position: relative;}
    .status-step::before {content: ""; position: absolute; top: 50%; left: -0.5rem; transform: translateY(-50%);
                         width: 0.8rem; height: 0.8rem; border-radius: 50%; background: #c5c5d0;}
    .status-step:first-child::before {display: none;}
    .status-step.active {background: linear-gradient(135deg, #fff3e0, #ffe0b2); border-color: #ffb74d;}
    .status-step.done {background: linear-gradient(135deg, #e8f5e9, #c8e6c9); border-color: #66bb6a;}
    .status-step.error {background: linear-gradient(135deg, #ffebee, #ffcdd2); border-color: #e57373;}
    .status-title {font-weight: 600; margin-bottom: 0.35rem; color: #1f1f24;}
    .status-detail {font-size: 0.85rem; color: #4a4a60;}
    </style>
    """,
    unsafe_allow_html=True,
)

_SESSION_DEFAULTS = {
    "pipeline_choice": "Indexer + Skillset",
    "uploaded_files": [],
    "uploaded_temp_dir": "",
    "shop_manual_files": [],
    "shop_manual_temp_dir": "",
    "shop_manual_mapping": None,
    "shop_manual_status": [],
    "shop_chunk_size": 500,
    "shop_chunk_overlap": 40,
    "shop_recreate_index": False,
    "shop_max_workers": 4,
    "shop_manual_progress": 0.0,
    "shop_manual_results": None,
}

for _key, _value in _SESSION_DEFAULTS.items():
    st.session_state.setdefault(_key, _value)


def _list_blob_containers(account_url: str, credential: object | None) -> list[str]:
    try:
        service_client = BlobServiceClient(account_url=account_url, credential=credential)
        return [container.name for container in service_client.list_containers()]
    except Exception as exc:  # pragma: no cover - network dependency
        st.warning(f"Unable to list containers: {exc}")
        return []


_PIZZA_STEPS = [
    ("prepare", "Dough prepped"),
    ("convert", "In the oven"),
    ("ocr", "Quality check"),
    ("index", "Out for delivery"),
    ("complete", "Served hot"),
]


def _initial_status() -> list[dict[str, str]]:
    return [
        {"key": key, "label": label, "state": "waiting", "message": "Waiting to start"}
        for key, label in _PIZZA_STEPS
    ]


def _render_status_tracker(statuses: list[dict[str, str]]) -> str:
    segments: list[str] = []
    for step in statuses:
        state = step.get("state", "waiting")
        label = step.get("label", step.get("key", ""))
        detail = step.get("message", "")
        state_class = {
            "waiting": "",
            "active": "active",
            "done": "done",
            "error": "error",
        }.get(state, "")
        icon = {
            "waiting": "⏳",
            "active": "🔥",
            "done": "✅",
            "error": "❌",
        }.get(state, "⏳")
        segments.append(
            "<div class='status-step {cls}'>"
            "<div class='status-title'>{icon} {label}</div>"
            "<div class='status-detail'>{detail}</div>"
            "</div>".format(cls=state_class, icon=icon, label=label, detail=detail)
        )
    return f"<div class='status-tracker'>{''.join(segments)}</div>"


with st.sidebar:
    st.header("Connections")
    search_endpoint = st.text_input("Azure AI Search endpoint", os.getenv("AZURE_SEARCH_ENDPOINT", ""))
    search_index = st.text_input("Index name", value=os.getenv("AZURE_SEARCH_INDEX", ""))
    auth_mode = st.radio("Authentication", ["Managed Identity / AAD", "API Key"], horizontal=False)
    api_key = st.text_input("Search query API key", type="password") if auth_mode == "API Key" else None
    search_admin_key = st.text_input(
        "Search admin/API key",
        value=os.getenv("AZURE_SEARCH_ADMIN_KEY", ""),
        type="password",
        help="Required for push and shop manual pipelines.",
    )

    openai_endpoint = st.text_input("Azure OpenAI endpoint", os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    openai_deployment = st.text_input("Embeddings deployment name", os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", ""))
    openai_api_key = st.text_input(
        "Azure OpenAI API key",
        value=os.getenv("AZURE_OPENAI_API_KEY", ""),
        type="password",
    )
    openai_api_version = st.text_input(
        "Azure OpenAI API version",
        value=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    )

    blob_account_url = st.text_input("Blob account URL", os.getenv("AZURE_STORAGE_ACCOUNT_URL", ""))
    blob_container = st.selectbox(
        "Blob container",
        options=[""] + _list_blob_containers(blob_account_url, get_default_credential()) if blob_account_url else [""],
    )
    local_upload_enabled = st.checkbox("Enable local uploads", True)

    st.divider()
    st.subheader("Document Intelligence")
    doc_int_endpoint = st.text_input("Service endpoint", os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", ""))
    doc_int_key = st.text_input(
        "Service key",
        value=os.getenv("AZURE_DOC_INTELLIGENCE_KEY", ""),
        type="password",
    )
    doc_int_model = st.text_input(
        "Model ID",
        value=os.getenv("AZURE_DOC_INTELLIGENCE_MODEL", "prebuilt-layout"),
    )

    st.divider()
    st.subheader("Environment checklist")
    env_values = {
        "AZURE_SEARCH_ENDPOINT": search_endpoint,
        "AZURE_SEARCH_INDEX": search_index,
        "AZURE_SEARCH_ADMIN_KEY": search_admin_key,
        "AZURE_OPENAI_ENDPOINT": openai_endpoint,
        "AZURE_OPENAI_EMBED_DEPLOYMENT": openai_deployment,
        "AZURE_OPENAI_API_KEY": openai_api_key,
        "AZURE_OPENAI_API_VERSION": openai_api_version,
        "AZURE_STORAGE_ACCOUNT_URL": blob_account_url,
        "AZURE_DOC_INTELLIGENCE_ENDPOINT": doc_int_endpoint,
        "AZURE_DOC_INTELLIGENCE_KEY": doc_int_key,
    }
    for key, value in env_values.items():
        icon = "✅" if value else "⚠️"
        st.markdown(f"{icon} `{key}`")
    st.caption("⚠️ indicates a missing or blank value. Use the export tab to download a .env template.")

credential = resolve_search_credential(api_key if auth_mode == "API Key" else None)

tabs = ["Pipeline", "Index & Schema", "Run", "Verify", "Visualize", "Export"]
(
    pipeline_tab,
    index_tab,
    run_tab,
    verify_tab,
    visualize_tab,
    export_tab,
) = st.tabs(tabs)

with pipeline_tab:
    st.subheader("Pipeline selection")

    def _stage_uploads(files, session_key: str, temp_key: str) -> list[Path]:
        if not files:
            return st.session_state.get(session_key, [])
        previous_dir = st.session_state.get(temp_key)
        if previous_dir:
            shutil.rmtree(previous_dir, ignore_errors=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{session_key}_"))
        staged: list[Path] = []
        for upload in files:
            file_path = temp_dir / upload.name
            file_path.write_bytes(upload.read())
            staged.append(file_path)
        st.session_state[session_key] = staged
        st.session_state[temp_key] = str(temp_dir)
        return staged

    pipeline_choice = st.radio(
        "Pipeline",
        ["Indexer + Skillset", "Push API", "Shop Manual"],
        horizontal=True,
        key="pipeline_choice",
    )

    skillset_preview: Optional[dict[str, Any]] = None
    skillset_options: Optional[SkillsetOptions] = None

    if pipeline_choice in {"Indexer + Skillset", "Push API"}:
        st.subheader("Source selection")
        source_mode = st.radio("Source", ["Blob container", "Upload files"], horizontal=True, key="source_mode")
        if source_mode == "Upload files" and local_upload_enabled:
            uploads = st.file_uploader(
                "Upload documents",
                accept_multiple_files=True,
                key="generic_uploads",
            )
            if uploads:
                staged_files = _stage_uploads(uploads, "uploaded_files", "uploaded_temp_dir")
                st.success(
                    f"Staged {len(staged_files)} files to {st.session_state.get('uploaded_temp_dir')}"
                )
        else:
            st.info("Blob mode will read directly from the configured container when the pipeline runs.")

        if pipeline_choice == "Indexer + Skillset":
            st.info("Configure data source, skillset, and indexer settings for pull-based ingestion.")
            ds_name = st.text_input("Data source name", value="blob-datasource")
            skillset_name = st.text_input("Skillset name", value="rag-skillset")
            indexer_name = st.text_input("Indexer name", value="rag-indexer")
            schedule_interval = st.text_input("Schedule interval (ISO8601)", value="PT2H")
            batch_size = st.number_input("Batch size", min_value=1, value=50)
            image_action = st.selectbox("Image action", ["none", "generateNormalizedImages"], index=1)

            st.subheader("Skillset configuration")
            ocr_enabled = st.checkbox("Enable Document Layout skill", value=True)
            captioning_enabled = st.checkbox("Generate image captions", value=False)
            embedding_skill = st.checkbox("Use Azure OpenAI embedding skill", value=True)
            embedding_dimensions = st.number_input("Embedding dimensions", value=1536)

            skillset_options = SkillsetOptions(
                name=skillset_name,
                ocr_enabled=ocr_enabled,
                captioning_enabled=captioning_enabled,
                embedding_skill=embedding_skill,
                embedding_deployment=openai_deployment or None,
                embedding_endpoint=openai_endpoint or None,
                embedding_dimensions=int(embedding_dimensions) if embedding_dimensions else None,
            )
            skillset_preview = build_skillset_payload(skillset_options)
            st.code(json.dumps(skillset_preview, indent=2), language="json")
        elif pipeline_choice == "Push API":
            st.info("Client-side chunking and embedding before pushing into the index.")
            chunk_strategy = st.selectbox("Chunking strategy", ["Paragraph (Layout)", "Sentences/Chars", "By Page"])
            chunk_size = st.slider("Chunk size", min_value=200, max_value=2000, value=800, step=50)
            chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=400, value=100, step=10)
            keep_ordinals = st.checkbox("Keep ordinal positions", value=True)
            table_extraction = st.multiselect("Table extraction", ["Markdown", "HTML"], default=["Markdown"])
            ocr_toggle = st.checkbox("Enable OCR for images", value=True)
            caption_toggle = st.checkbox("Generate captions (client-side)", value=False)
            embedding_mode = st.selectbox("Embeddings", ["Client-side", "Indexer skill", "Vectorizer"])
            embedding_dims = st.number_input("Embedding dimensions (client-side)", value=1536)
            precision = st.selectbox("Precision", ["float32", "float16"])
            quantize = st.checkbox("Quantize vectors", value=False)
    else:
        st.subheader("Shop manual inputs")
        st.info(
            "A ready-to-serve workflow that slices manuals, extracts content with Document Intelligence, "
            "and pushes vectors into your index with Uber Eats-style updates."
        )
        manuals = st.file_uploader(
            "Upload shop manuals (PDF)",
            accept_multiple_files=True,
            type=["pdf"],
            key="shop_manual_uploads",
        )
        if manuals:
            staged_manuals = _stage_uploads(manuals, "shop_manual_files", "shop_manual_temp_dir")
            st.success(
                f"Ready to bake {len(staged_manuals)} manuals from {st.session_state.get('shop_manual_temp_dir')}"
            )
        elif st.session_state.get("shop_manual_files"):
            st.info(
                f"{len(st.session_state['shop_manual_files'])} manuals staged from a previous upload session."
            )

        mapping_upload = st.file_uploader(
            "Mapping CSV (optional)",
            type=["csv"],
            key="shop_manual_mapping_upload",
            help="Used to enrich documents with publication numbers.",
        )
        if mapping_upload is not None:
            mapping_df = pd.read_csv(mapping_upload)
            st.session_state["shop_manual_mapping"] = mapping_df
            st.success(f"Loaded mapping with {len(mapping_df)} rows")

        mapping_preview = st.session_state.get("shop_manual_mapping")
        if mapping_preview is not None and not mapping_preview.empty:
            st.dataframe(mapping_preview.head(25))

        st.subheader("Pipeline tuning")
        st.slider(
            "Chunk size",
            min_value=200,
            max_value=1500,
            step=50,
            key="shop_chunk_size",
        )
        st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=400,
            step=10,
            key="shop_chunk_overlap",
        )
        st.number_input(
            "Parallel workers",
            min_value=1,
            max_value=8,
            key="shop_max_workers",
        )
        st.checkbox(
            "Recreate index on run",
            key="shop_recreate_index",
            help="Drops and recreates the Azure AI Search index before ingestion.",
        )

uploaded_files = st.session_state.get("uploaded_files", [])
shop_manual_files = st.session_state.get("shop_manual_files", [])
shop_manual_mapping = st.session_state.get("shop_manual_mapping")

with index_tab:
    st.subheader("Index schema")
    vector_dim = st.number_input("Vector dimension", value=1536)
    vector_field = st.text_input("Vector field name", value="content_vector")
    use_float16 = st.checkbox("Use float16 vectors", value=False)
    semantic_config = st.text_input("Semantic configuration name", value="default")

    index_schema = build_index_schema(
        IndexSchemaOptions(
            index_name=search_index or "rag-index",
            vector_config=VectorConfig(name=vector_field, dimensions=int(vector_dim), use_float16=use_float16),
            semantic_configuration_name=semantic_config,
        )
    )
    st.code(json.dumps(index_schema, indent=2), language="json")

    if st.button("Create / Update index"):
        validation = require_non_empty(search_endpoint, "Search endpoint")
        if not validation.valid:
            st.error(validation.message)
        else:
            try:
                index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
                index_client.create_or_update_index(index_schema)  # type: ignore[arg-type]
                st.success(f"Index {index_schema['name']} created/updated")
            except AzureError as exc:  # pragma: no cover - network call
                st.error(f"Failed to create index: {exc}")

    st.subheader("Aliases")
    alias_name = st.text_input("Alias name", value=f"{search_index or 'rag-index'}-alias")
    target_index_name = st.text_input("New index version", value=f"{search_index or 'rag-index'}_v{int(time.time())}")

    if st.button("Swap alias"):
        try:
            index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
            alias = swap_alias(index_client, AliasOptions(alias_name=alias_name, index_name=target_index_name))
            st.success(f"Alias swapped: {alias.name} -> {target_index_name}")
        except AzureError as exc:  # pragma: no cover - network call
            st.error(f"Failed to swap alias: {exc}")

with run_tab:
    st.subheader("Execute pipeline")
    if pipeline_choice == "Indexer + Skillset":
        if st.button("Create resources and run indexer"):
            validation = require_non_empty(blob_container or "", "Blob container")
            if not validation.valid:
                st.error(validation.message)
            else:
                try:
                    indexer_client = SearchIndexerClient(endpoint=search_endpoint, credential=credential)
                    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
                    data_source = ensure_data_source(
                        indexer_client,
                        DataSourceOptions(
                            name=ds_name,
                            storage_connection_string=connection_string,
                            container_name=blob_container,
                        ),
                    )
                    indexer = ensure_indexer(
                        indexer_client,
                        IndexerOptions(
                            name=indexer_name,
                            data_source_name=data_source.name,
                            skillset_name=skillset_name,
                            target_index_name=search_index,
                            schedule_interval=schedule_interval,
                            batch_size=int(batch_size),
                            image_action=None if image_action == "none" else image_action,
                        ),
                    )
                    indexer_client.create_or_update_skillset(build_skillset(skillset_options or SkillsetOptions(name=skillset_name)))  # type: ignore[arg-type]
                    run_indexer(indexer_client, indexer.name)
                    st.success("Indexer execution triggered")
                except AzureError as exc:  # pragma: no cover - network call
                    st.error(f"Indexer run failed: {exc}")
    elif pipeline_choice == "Push API":
        st.write("Run push pipeline for uploaded files.")
        if st.button("Run push ingestion"):
            if not uploaded_files:
                st.warning("No files uploaded")
            else:
                progress = st.progress(0)
                status_placeholder = st.empty()
                try:
                    pipeline = PushIngestionPipeline(
                        PushPipelineConfig(
                            search_endpoint=search_endpoint,
                            index_name=search_index,
                            credential=credential,
                            embedding_config=EmbeddingConfig(
                                deployment=openai_deployment,
                                endpoint=openai_endpoint or None,
                                api_key=openai_api_key or os.getenv("AZURE_OPENAI_API_KEY", ""),
                                api_version=openai_api_version or os.getenv("AZURE_OPENAI_API_VERSION", ""),
                                dimensions=int(embedding_dims),
                                use_float16=precision == "float16",
                                quantize=quantize,
                            ),
                            chunk_options=TextSplitOptions(
                                chunk_size=int(chunk_size),
                                chunk_overlap=int(chunk_overlap),
                                keep_ordinals=keep_ordinals,
                            ),
                        )
                    )
                    for idx, file_path in enumerate(uploaded_files, start=1):
                        text = file_path.read_text(encoding="utf-8", errors="ignore")
                        chunks = collect_chunks_from_text(
                            text,
                            TextSplitOptions(
                                chunk_size=int(chunk_size),
                                chunk_overlap=int(chunk_overlap),
                                keep_ordinals=keep_ordinals,
                            ),
                        )
                        metadata = {
                            "doc_id": hashing.sha256_path(file_path),
                            "file_name": file_path.name,
                            "source_type": "upload",
                        }
                        pipeline.ingest_chunks(chunks, metadata)
                        progress.progress(idx / len(uploaded_files))
                        status_placeholder.info(f"Processed {file_path.name} with {len(chunks)} chunks")
                    st.success("Push ingestion complete")
                except AzureError as exc:  # pragma: no cover - network call
                    st.error(f"Push ingestion failed: {exc}")
    else:
        st.write("Launch the automated shop manual pipeline.")
        status_container = st.container()
        progress_bar = st.progress(max(0.0, min(st.session_state.get("shop_manual_progress", 0.0), 1.0)))
        if st.session_state.get("shop_manual_status"):
            status_container.markdown(
                _render_status_tracker(st.session_state["shop_manual_status"]),
                unsafe_allow_html=True,
            )

        if st.button("Bake manuals", key="run_shop_manual"):
            if not shop_manual_files:
                st.warning("No manuals uploaded")
            elif not search_endpoint or not search_index:
                st.error("Search endpoint and index name are required.")
            elif not search_admin_key:
                st.error("Provide a search admin/API key in the sidebar.")
            elif not openai_endpoint or not openai_deployment or not openai_api_key:
                st.error("Azure OpenAI endpoint, deployment, and API key are required.")
            elif not doc_int_endpoint or not doc_int_key:
                st.error("Document Intelligence endpoint and key are required.")
            else:
                st.session_state["shop_manual_status"] = _initial_status()
                st.session_state["shop_manual_progress"] = 0.0
                status_container.markdown(
                    _render_status_tracker(st.session_state["shop_manual_status"]),
                    unsafe_allow_html=True,
                )
                progress_bar.progress(0.05)

                config = ShopManualConfig(
                    search_endpoint=search_endpoint,
                    search_api_key=search_admin_key,
                    index_name=search_index,
                    openai_endpoint=openai_endpoint,
                    openai_deployment=openai_deployment,
                    openai_api_key=openai_api_key,
                    openai_api_version=openai_api_version,
                    document_intelligence_endpoint=doc_int_endpoint,
                    document_intelligence_key=doc_int_key,
                    document_intelligence_model=doc_int_model,
                    chunk_size=int(st.session_state["shop_chunk_size"]),
                    chunk_overlap=int(st.session_state["shop_chunk_overlap"]),
                    max_workers=int(st.session_state["shop_max_workers"]),
                    recreate_index=bool(st.session_state["shop_recreate_index"]),
                )

                def _update_status(step_key: str, state: str, message: str, progress: float | None) -> None:
                    statuses = st.session_state.get("shop_manual_status", [])
                    for entry in statuses:
                        if entry.get("key") == step_key:
                            entry.update({"state": state, "message": message})
                            break
                    else:
                        statuses.append(
                            {"key": step_key, "label": step_key.title(), "state": state, "message": message}
                        )
                    st.session_state["shop_manual_status"] = statuses
                    status_container.markdown(_render_status_tracker(statuses), unsafe_allow_html=True)
                    if progress is not None:
                        st.session_state["shop_manual_progress"] = progress
                        progress_bar.progress(max(0.0, min(progress, 1.0)))

                try:
                    with ShopManualPipeline(config) as pipeline:
                        results = pipeline.run(shop_manual_files, shop_manual_mapping, _update_status)
                    progress_bar.progress(1.0)
                    summary = pd.DataFrame(
                        [
                            {
                                "manual": result.pdf_path.name,
                                "pages": result.total_pages,
                                "chunks_indexed": result.indexed_chunks,
                            }
                            for result in results
                        ]
                    )
                    st.session_state["shop_manual_results"] = summary
                    if not summary.empty:
                        st.dataframe(summary)
                    st.success("Manual ingestion complete")
                except Exception as exc:  # pragma: no cover - runtime feedback
                    _update_status("complete", "error", f"Failed with error: {exc}", None)
                    st.error(f"Shop manual pipeline failed: {exc}")

        if st.session_state.get("shop_manual_results") is not None:
            summary_df = st.session_state["shop_manual_results"]
            if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
                st.dataframe(summary_df)

with verify_tab:
    st.subheader("Hybrid search test")
    query = st.text_input("Query text")
    top_k_text = st.slider("topK text", min_value=1, max_value=50, value=5)
    top_k_vector = st.slider("topK vector", min_value=0, max_value=50, value=3)
    semantic_rerank = st.checkbox("Semantic rerank", value=True)

    if st.button("Search") and query:
        try:
            search_client = SearchClient(endpoint=search_endpoint, index_name=search_index, credential=credential)
            results = search_client.search(
                query,
                top=top_k_text,
                vector_queries=[
                    {
                        "kind": "vector",
                        "vector": [0.0] * int(vector_dim),
                        "k": top_k_vector,
                        "fields": vector_field,
                    }
                ]
                if top_k_vector
                else None,
                semantic_configuration=semantic_config if semantic_rerank else None,
            )
            rows = []
            for result in results:
                rows.append(
                    {
                        "score": result["@search.score"],
                        "chunk_text": result.get("chunk_text", ""),
                        "page": result.get("page"),
                        "bbox": result.get("bbox"),
                    }
                )
            st.dataframe(pd.DataFrame(rows))
        except AzureError as exc:  # pragma: no cover - network call
            st.error(f"Search failed: {exc}")

with visualize_tab:
    st.subheader("Data visualizer")

    st.markdown("### Vector embedding explorer")
    sample_size = st.slider("Documents to sample", min_value=10, max_value=200, value=50, step=10)
    vector_field = st.text_input("Vector field", value="content_vector", key="visual_vector_field")
    fetch_embeddings = st.button("Fetch embeddings", key="fetch_embeddings")

    if fetch_embeddings:
        if not search_endpoint or not search_index:
            st.error("Configure search endpoint and index first.")
        else:
            try:
                search_client = SearchClient(endpoint=search_endpoint, index_name=search_index, credential=credential)
                select_fields = ",".join(
                    ["chunk_text", "pdf_name", "page_no", "publication_number", vector_field]
                )
                results = list(
                    search_client.search(
                        "*",
                        top=sample_size,
                        vector_queries=None,
                        select=select_fields,
                    )
                )
                embeddings: list[list[float]] = []
                labels: list[str] = []
                metadata_rows: list[dict[str, Any]] = []
                for item in results:
                    vector = item.get(vector_field)
                    if vector is None:
                        continue
                    try:
                        vector_list = list(vector)
                    except TypeError:
                        continue
                    embeddings.append(vector_list)
                    labels.append(item.get("chunk_text", "")[:120])
                    metadata_rows.append(
                        {
                            "chunk_text": item.get("chunk_text", ""),
                            "pdf_name": item.get("pdf_name"),
                            "page_no": item.get("page_no"),
                            "publication_number": item.get("publication_number"),
                        }
                    )

                if not embeddings:
                    st.warning("No embeddings returned. Ensure the field is retrievable and contains vector data.")
                else:
                    embedding_array = np.array(embeddings)
                    if embedding_array.shape[1] > 2 and PCA is not None:
                        reducer = PCA(n_components=2)
                        reduced = reducer.fit_transform(embedding_array)
                    else:
                        reduced = embedding_array[:, :2]
                    viz_df = pd.DataFrame(
                        {
                            "x": reduced[:, 0],
                            "y": reduced[:, 1],
                            "label": labels,
                        }
                    )
                    meta_df = pd.DataFrame(metadata_rows)
                    combined = pd.concat([viz_df, meta_df], axis=1)
                    st.scatter_chart(combined, x="x", y="y")
                    st.dataframe(combined)
            except Exception as exc:  # pragma: no cover - visualization
                st.error(f"Failed to fetch embeddings: {exc}")

    st.markdown("### Mapping table preview")
    if shop_manual_mapping is not None and not shop_manual_mapping.empty:
        selected_publication = st.selectbox(
            "Filter by publication number",
            options=["All"] + sorted(shop_manual_mapping["publication_number"].dropna().astype(str).unique().tolist()),
            key="mapping_filter",
        )
        if selected_publication != "All":
            filtered_df = shop_manual_mapping[shop_manual_mapping["publication_number"].astype(str) == selected_publication]
        else:
            filtered_df = shop_manual_mapping
        st.dataframe(filtered_df)
    else:
        st.info("Upload a mapping CSV in the pipeline tab to visualize it here.")

    st.markdown("### Blob storage snapshot")
    blob_limit = st.slider("Blobs to show", min_value=5, max_value=100, value=20, step=5, key="blob_limit")
    if st.button("List blobs", key="list_blobs"):
        if not blob_account_url or not blob_container:
            st.error("Provide a blob account URL and container name.")
        else:
            try:
                service_client = BlobServiceClient(account_url=blob_account_url, credential=get_default_credential())
                container_client = service_client.get_container_client(blob_container)
                rows = []
                for idx, blob in enumerate(container_client.list_blobs()):
                    if idx >= blob_limit:
                        break
                    rows.append(
                        {
                            "name": blob.name,
                            "size (KB)": round((blob.size or 0) / 1024, 2),
                            "last_modified": getattr(blob, "last_modified", None),
                        }
                    )
                if rows:
                    st.dataframe(pd.DataFrame(rows))
                else:
                    st.info("No blobs found or container is empty.")
            except Exception as exc:  # pragma: no cover - network interaction
                st.error(f"Unable to list blobs: {exc}")

with export_tab:
    st.subheader("Export definitions")
    export_payload = {
        "index": index_schema,
        "skillset": skillset_preview if pipeline_choice == "Indexer + Skillset" else None,
        "env_template": {
            "AZURE_SEARCH_ENDPOINT": search_endpoint,
            "AZURE_SEARCH_INDEX": search_index,
            "AZURE_SEARCH_ADMIN_KEY": search_admin_key,
            "AZURE_OPENAI_ENDPOINT": openai_endpoint,
            "AZURE_OPENAI_EMBED_DEPLOYMENT": openai_deployment,
            "AZURE_OPENAI_API_KEY": openai_api_key,
            "AZURE_OPENAI_API_VERSION": openai_api_version,
            "AZURE_STORAGE_ACCOUNT_URL": blob_account_url,
            "AZURE_DOC_INTELLIGENCE_ENDPOINT": doc_int_endpoint,
            "AZURE_DOC_INTELLIGENCE_KEY": doc_int_key,
            "AZURE_DOC_INTELLIGENCE_MODEL": doc_int_model,
        },
    }
    st.download_button(
        "Download JSON",
        data=json.dumps(export_payload, indent=2).encode("utf-8"),
        file_name="rag_ingestor_export.json",
        mime="application/json",
    )

    st.download_button(
        "Download .env template",
        data="\n".join(f"{key}={value}" for key, value in export_payload["env_template"].items()).encode("utf-8"),
        file_name=".env.sample",
        mime="text/plain",
    )
