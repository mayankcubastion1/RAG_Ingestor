"""Azure AI Search index schema builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    from azure.search.documents.indexes.models import (  # type: ignore
        HnswParameters,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SemanticConfiguration,
        SemanticField,
        SemanticSearch,
        VectorSearch,
        VectorSearchAlgorithmConfiguration,
        VectorSearchProfile,
    )
except ImportError:  # pragma: no cover - fallback types
    HnswParameters = SearchField = SearchFieldDataType = SearchIndex = SemanticConfiguration = SemanticField = (
        SemanticSearch
    ) = VectorSearch = VectorSearchAlgorithmConfiguration = VectorSearchProfile = object  # type: ignore


@dataclass
class FieldToggle:
    name: str
    searchable: bool = True
    filterable: bool = False
    sortable: bool = False
    facetable: bool = False
    retrievable: bool = True


@dataclass
class VectorConfig:
    name: str = "content_vector"
    dimensions: int = 1536
    distance_metric: str = "cosine"
    use_float16: bool = False


@dataclass
class IndexSchemaOptions:
    index_name: str
    fields: List[FieldToggle] = field(default_factory=list)
    vector_config: VectorConfig = field(default_factory=VectorConfig)
    semantic_configuration_name: str = "default"


DEFAULT_FIELDS: List[FieldToggle] = [
    FieldToggle("id", searchable=False, filterable=True),
    FieldToggle("doc_id", searchable=False, filterable=True),
    FieldToggle("source_type", searchable=False, filterable=True),
    FieldToggle("container", searchable=False, filterable=True),
    FieldToggle("blob_path", searchable=False, filterable=True),
    FieldToggle("file_name", filterable=True),
    FieldToggle("file_ext", filterable=True),
    FieldToggle("content_type", filterable=True),
    FieldToggle("title", searchable=True, filterable=False),
    FieldToggle("section_path", searchable=True, filterable=True),
    FieldToggle("language", searchable=False, filterable=True),
    FieldToggle("page", searchable=False, filterable=True, sortable=True),
    FieldToggle("bbox", searchable=False, retrievable=True),
    FieldToggle("chunk_index", searchable=False),
    FieldToggle("char_start", searchable=False),
    FieldToggle("char_end", searchable=False),
    FieldToggle("chunk_text", searchable=True, retrievable=True),
    FieldToggle("is_table", searchable=False, filterable=True),
    FieldToggle("table_markdown", retrievable=True),
    FieldToggle("image_caption", searchable=True),
    FieldToggle("image_ref", retrievable=True),
    FieldToggle("content_vector", searchable=True, retrievable=True),
    FieldToggle("embedding_model", filterable=True),
    FieldToggle("embedding_dim", filterable=True),
    FieldToggle("embedding_ts", filterable=True, sortable=True),
    FieldToggle("tags", filterable=True, facetable=True),
    FieldToggle("created_at", filterable=True, sortable=True),
    FieldToggle("last_modified", filterable=True, sortable=True),
    FieldToggle("sha256", filterable=True),
    FieldToggle("pipeline_version", filterable=True),
    FieldToggle("skillset_name", filterable=True),
]


def build_index_schema(options: IndexSchemaOptions) -> Dict[str, Any]:
    """Build a JSON serializable index definition."""

    fields = [
        {
            "name": toggle.name,
            "type": _infer_field_type(toggle.name, options.vector_config),
            "searchable": toggle.searchable,
            "filterable": toggle.filterable,
            "facetable": toggle.facetable,
            "sortable": toggle.sortable,
            "retrievable": toggle.retrievable,
        }
        for toggle in (options.fields or DEFAULT_FIELDS)
    ]

    vector_search = {
        "algorithms": [
            {
                "name": "hnsw",
                "kind": "hnsw",
                "parameters": {
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 100,
                    "metric": options.vector_config.distance_metric,
                },
            }
        ],
        "profiles": [
            {
                "name": options.vector_config.name,
                "algorithmConfigurationName": "hnsw",
            }
        ],
    }

    semantic_search = {
        "configurations": [
            {
                "name": options.semantic_configuration_name,
                "prioritizedFields": {
                    "titleField": {"fieldName": "title"},
                    "prioritizedContentFields": [{"fieldName": "chunk_text"}],
                    "prioritizedKeywordsFields": [{"fieldName": "tags"}],
                },
            }
        ]
    }

    index_definition = {
        "name": options.index_name,
        "fields": fields,
        "vectorSearch": vector_search,
        "semanticSearch": semantic_search,
    }
    return index_definition


def _infer_field_type(field_name: str, vector_config: VectorConfig) -> str:
    if field_name == vector_config.name:
        return "Collection(Edm.Half)" if vector_config.use_float16 else "Collection(Edm.Single)"
    mapping = {
        "id": "Edm.String",
        "doc_id": "Edm.String",
        "source_type": "Edm.String",
        "container": "Edm.String",
        "blob_path": "Edm.String",
        "file_name": "Edm.String",
        "file_ext": "Edm.String",
        "content_type": "Edm.String",
        "title": "Edm.String",
        "section_path": "Collection(Edm.String)",
        "language": "Edm.String",
        "page": "Edm.Int32",
        "bbox": "Edm.String",
        "chunk_index": "Edm.Int32",
        "char_start": "Edm.Int32",
        "char_end": "Edm.Int32",
        "chunk_text": "Edm.String",
        "is_table": "Edm.Boolean",
        "table_markdown": "Edm.String",
        "image_caption": "Edm.String",
        "image_ref": "Edm.String",
        "embedding_model": "Edm.String",
        "embedding_dim": "Edm.Int32",
        "embedding_ts": "Edm.DateTimeOffset",
        "tags": "Collection(Edm.String)",
        "created_at": "Edm.DateTimeOffset",
        "last_modified": "Edm.DateTimeOffset",
        "sha256": "Edm.String",
        "pipeline_version": "Edm.String",
        "skillset_name": "Edm.String",
    }
    return mapping.get(field_name, "Edm.String")
