# RAG Ingestor Streamlit App

This repository contains a production-ready Streamlit application for ingesting heterogeneous content into Azure AI Search. The UI allows operations teams to configure ingestion pipelines, manage index lifecycle, and validate search quality without leaving the browser.

## Features

- Multiple ingestion sources: Azure Blob Storage and drag-and-drop uploads
- Dual pipelines:
  - **Indexer + Skillset** for pull-based ingestion from Blob Storage
  - **Push API** for client-side chunking, embeddings, and bulk upload
- Rich configuration for chunking, table extraction, image captioning, and embeddings
- Index schema builder with vector search configuration and semantic options
- Alias management for versioned indexes *(requires the Azure Search preview SDK)*
- Hybrid search test console with semantic reranking
- Export utilities for Infrastructure-as-Code definitions and environment templates

## Project structure

```
app/
  main.py                       # Streamlit UI
  azure_search/
    index_schema.py             # Index schema builders
    skillsets.py                # Skillset payload builders
    indexers.py                 # Data source and indexer helpers
    push_pipeline.py            # Client-side ingestion pipeline
    aliases.py                  # Index alias helpers
    rbac.py                     # Credential resolution helpers
  chunking/
    layout_chunker.py           # Layout-aware chunking (PDFs)
    textsplit_chunker.py        # Text splitting utilities
    tables.py                   # Table extraction helpers
  embeddings/
    azure_openai.py             # Azure OpenAI embedding helpers
    quantize.py                 # Optional vector quantization
  utils/
    blob.py                     # Blob Storage helpers
    hashing.py                  # SHA256 hashing utilities
    validators.py               # Simple validation helpers
    logging.py                  # Logging configuration
```

## Prerequisites

1. **Python** 3.11 or higher
2. **Azure resources**
   - Azure AI Search (Basic tier or higher) with Semantic Ranker enabled
   - Azure OpenAI resource with an embedding deployment
   - Azure Storage account with a container for ingestion
   - Optional: Cosmos DB for MongoDB vCore or Azure Database for PostgreSQL with pgvector
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** Azure Search index alias APIs are only available in preview builds of
   > `azure-search-documents`. Install `azure-search-documents==11.6.0b12` (or a newer
   > beta) if you plan to manage aliases from the UI.

## Configuration

Create an `.env` file (see [.env.sample](./.env.sample)) or export the following environment variables:

- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_INDEX`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_EMBED_DEPLOYMENT`
- `AZURE_STORAGE_ACCOUNT_URL`
- `AZURE_STORAGE_CONNECTION_STRING` (for indexer data source creation)
- Optional: `AZURE_SEARCH_API_KEY` for API key authentication

When running in Azure, the app prefers Managed Identity / AAD credentials via `DefaultAzureCredential`.

## Running the app

1. Install dependencies as shown above.
2. Launch Streamlit from the repository root:

   ```bash
   streamlit run app/main.py
   ```

3. Set the required environment variables before launching (or provide them in the sidebar when prompted).

## Usage overview

1. **Configure connections** on the sidebar. Choose authentication mode and supply Azure endpoints.
2. **Select ingestion source and pipeline** in the Pipeline tab. For uploads, stage files locally or push to Blob Storage when using the indexer path.
3. **Adjust chunking and embedding settings**. Client-side pipeline offers paragraph/sentence/page chunking, optional quantization, and local embedding generation.
4. **Design the index** in the Index tab, then create/update it directly from the UI. Manage aliases to swap traffic between versions.
5. **Run ingestion** via the Run tab. For indexer mode, the app provisions the data source, skillset, and indexer before triggering execution. For push mode, it shows per-file progress, chunk counts, and embedding activity.
6. **Validate search quality** using the Verify tab. Perform hybrid text + vector queries and inspect merged results.
7. **Export definitions** for automation. Download the index and skillset JSON definitions alongside an environment template.

## Extensibility

- Extend `app/chunking/` with custom chunking strategies.
- Integrate other vector databases (Cosmos DB Mongo vCore, PostgreSQL with pgvector) by adding writers under a new module.
- Customize skillsets by editing `app/azure_search/skillsets.py` to include additional Cognitive Skills or Web API skills.

## Security considerations

- Authentication defaults to `DefaultAzureCredential`, enabling Managed Identity usage in Azure.
- API keys are only used when explicitly provided.
- The app avoids persisting secrets in source control; use environment variables or Azure App Service settings.

## TODOs

- Implement staging of uploaded files to Blob Storage when using the indexer pipeline.
- Add Cosmos DB Mongo vCore and PostgreSQL pgvector integrations.
- Enrich table extraction with document layout skills when available.
