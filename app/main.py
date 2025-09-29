"""Streamlit front-end for enterprise RAG ingestion."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):
    # Allow running ``streamlit run app/main.py`` without installing the package.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from azure.core.exceptions import AzureError  # type: ignore
from azure.search.documents import SearchClient  # type: ignore
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient  # type: ignore
from azure.storage.blob import BlobServiceClient  # type: ignore

from app.azure_search.aliases import AliasFeatureUnavailableError, AliasOptions, alias_feature_status, swap_alias
from app.azure_search.index_schema import IndexSchemaOptions, VectorConfig, build_index_schema
from app.azure_search.indexers import DataSourceOptions, IndexerOptions, ensure_data_source, ensure_indexer, run_indexer
from app.azure_search.push_pipeline import PushIngestionPipeline, PushPipelineConfig, collect_chunks_from_text
from app.azure_search.rbac import get_default_credential, resolve_search_credential
from app.azure_search.skillsets import SkillsetOptions, build_skillset, build_skillset_payload
from app.chunking.textsplit_chunker import TextSplitOptions
from app.embeddings.azure_openai import EmbeddingConfig
from app.utils import hashing
from app.utils.logging import configure_logging, get_logger
from app.utils.validators import require_non_empty

configure_logging()
_LOGGER = get_logger(__name__)

st.set_page_config(page_title="RAG Ingestor", layout="wide")
st.title("Enterprise RAG Ingestor for Azure AI Search")


def _list_blob_containers(account_url: str, credential: object | None) -> list[str]:
    try:
        service_client = BlobServiceClient(account_url=account_url, credential=credential)
        return [container.name for container in service_client.list_containers()]
    except Exception as exc:  # pragma: no cover - network dependency
        st.warning(f"Unable to list containers: {exc}")
        return []


with st.sidebar:
    st.header("Connections")
    search_endpoint = st.text_input("Azure AI Search endpoint", os.getenv("AZURE_SEARCH_ENDPOINT", ""))
    search_index = st.text_input("Index name", value=os.getenv("AZURE_SEARCH_INDEX", ""))
    auth_mode = st.radio("Authentication", ["Managed Identity / AAD", "API Key"], horizontal=False)
    api_key = st.text_input("Search API Key", type="password") if auth_mode == "API Key" else None

    openai_endpoint = st.text_input("Azure OpenAI endpoint", os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    openai_deployment = st.text_input("Embeddings deployment name", os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", ""))

    blob_account_url = st.text_input("Blob account URL", os.getenv("AZURE_STORAGE_ACCOUNT_URL", ""))
    blob_container = st.selectbox(
        "Blob container",
        options=[""] + _list_blob_containers(blob_account_url, get_default_credential()) if blob_account_url else [""],
    )
    local_upload_enabled = st.checkbox("Enable local uploads", True)

credential = resolve_search_credential(api_key if auth_mode == "API Key" else None)

pipeline_tab, index_tab, run_tab, verify_tab, export_tab = st.tabs(
    ["Pipeline", "Index & Schema", "Run", "Verify", "Export"]
)

with pipeline_tab:
    st.subheader("Source selection")
    source_mode = st.radio("Source", ["Blob container", "Upload files"], horizontal=True)

    uploaded_files: list[Path] = []
    if source_mode == "Upload files" and local_upload_enabled:
        uploads = st.file_uploader("Upload documents", accept_multiple_files=True)
        if uploads:
            temp_dir = Path(tempfile.mkdtemp(prefix="rag_uploads_"))
            for upload in uploads:
                file_path = temp_dir / upload.name
                file_path.write_bytes(upload.read())
                uploaded_files.append(file_path)
            st.success(f"Staged {len(uploaded_files)} files to {temp_dir}")

    st.subheader("Pipeline selection")
    pipeline_choice = st.radio("Pipeline", ["Indexer + Skillset", "Push API"], horizontal=True)

    skillset_preview: Optional[dict[str, Any]] = None

    skillset_options: Optional[SkillsetOptions] = None

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
    else:
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
    alias_supported, alias_message = alias_feature_status()
    if alias_message:
        st.info(alias_message)

    alias_name = st.text_input("Alias name", value=f"{search_index or 'rag-index'}-alias")
    target_index_name = st.text_input("New index version", value=f"{search_index or 'rag-index'}_v{int(time.time())}")

    if st.button("Swap alias", disabled=not alias_supported):
        try:
            index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
            alias = swap_alias(index_client, AliasOptions(alias_name=alias_name, index_name=target_index_name))
            st.success(f"Alias swapped: {alias.name} -> {target_index_name}")
        except AliasFeatureUnavailableError as exc:
            st.error(str(exc))
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
    else:
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

with export_tab:
    st.subheader("Export definitions")
    export_payload = {
        "index": index_schema,
        "skillset": skillset_preview if pipeline_choice == "Indexer + Skillset" else None,
        "env_template": {
            "AZURE_SEARCH_ENDPOINT": search_endpoint,
            "AZURE_SEARCH_INDEX": search_index,
            "AZURE_OPENAI_ENDPOINT": openai_endpoint,
            "AZURE_OPENAI_EMBED_DEPLOYMENT": openai_deployment,
            "AZURE_STORAGE_ACCOUNT_URL": blob_account_url,
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
