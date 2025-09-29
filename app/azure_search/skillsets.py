"""Skillset builders for Azure AI Search indexer pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

try:  # pragma: no cover - optional dependency
    from azure.search.documents.indexes.models import SearchIndexerSkillset  # type: ignore
except ImportError:  # pragma: no cover - fallback
    SearchIndexerSkillset = None  # type: ignore


@dataclass
class SkillsetOptions:
    name: str
    description: str = "RAG ingestion skillset"
    ocr_enabled: bool = True
    captioning_enabled: bool = False
    embedding_skill: bool = True
    embedding_deployment: str | None = None
    embedding_endpoint: str | None = None
    embedding_dimensions: int | None = None
    image_caption_prompt: str = "Generate a concise caption for this image."


def build_skillset_payload(options: SkillsetOptions) -> Dict[str, Any]:
    """Return a JSON serializable skillset definition."""

    skills: List[Dict[str, Any]] = []

    if options.ocr_enabled:
        skills.append(
            {
                "@odata.type": "#Microsoft.Skills.Vision.DocumentLayoutSkill",
                "name": "layoutSkill",
                "description": "Extract text and layout",
                "context": "/document",
                "inputs": [{"name": "document", "source": "/document/content"}],
                "outputs": [
                    {"name": "layoutText", "targetName": "layoutText"},
                    {"name": "pages", "targetName": "pages"},
                ],
            }
        )

    skills.append(
        {
            "@odata.type": "#Microsoft.Skills.Text.SplitSkill",
            "name": "textSplitSkill",
            "description": "Split text into chunks",
            "context": "/document/layoutText",
            "textSplitMode": "pages",
            "maximumPageLength": 1000,
            "inputs": [{"name": "text", "source": "/document/layoutText"}],
            "outputs": [{"name": "textItems", "targetName": "chunks"}],
        }
    )

    if options.embedding_skill:
        skills.append(
            {
                "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
                "name": "embeddingSkill",
                "context": "/document/chunks",
                "inputs": [{"name": "text", "source": "/document/chunks"}],
                "outputs": [
                    {"name": "embedding", "targetName": "contentVector"},
                    {"name": "modelId", "targetName": "embeddingModel"},
                ],
                "azureOpenAIParameters": {
                    "endpoint": options.embedding_endpoint,
                    "deploymentId": options.embedding_deployment,
                    "modelDimensions": options.embedding_dimensions,
                },
            }
        )

    if options.captioning_enabled:
        skills.append(
            {
                "@odata.type": "#Microsoft.Skills.Vision.AzureOpenAICaptioningSkill",
                "name": "captionSkill",
                "context": "/document/normalized_images/*",
                "inputs": [
                    {"name": "image", "source": "/document/normalized_images/*"},
                    {"name": "prompt", "source": options.image_caption_prompt},
                ],
                "outputs": [{"name": "captions", "targetName": "imageCaptions"}],
            }
        )

    return {
        "name": options.name,
        "description": options.description,
        "skills": skills,
        "cognitiveServices": {"@odata.type": "#Microsoft.Azure.Search.CognitiveServicesByKey"},
    }


def build_skillset(options: SkillsetOptions) -> Dict[str, Any] | SearchIndexerSkillset:
    payload = build_skillset_payload(options)
    if SearchIndexerSkillset:
        return SearchIndexerSkillset.deserialize(payload)  # type: ignore[return-value]
    return payload
