# app/core/config.py

import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Determine the project root directory dynamically
# This assumes config.py is in app/core/
# So, project_root will be the parent of 'app'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env file from the project root for local development
# In production/Kubernetes, environment variables will be set directly.
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    """
    Application settings.
    Values are loaded from environment variables or a .env file.
    """
    # --- Application Settings ---
    APP_NAME: str = "RAG with Kubernetes API"
    APP_VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development" # E.g., development, staging, production
    DEBUG: bool = True # Set to False in production


    # --- LLM Configuration (Example: OpenAI) ---
    # Replace these with actual settings for your chosen LLM provider
    LLM_PROVIDER: Optional[str] = "openai" # e.g., "openai", "anthropic", "huggingface_hub", "local_ollama"
    OPENAI_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL:str = "http://localhost:11434"
    OLLAMA_MODEL_NAME:str = "llama3" # Or whatever model you pulled, e.g., "phi3:mini"
    OLLAMA_REQUEST_TIMEOUT:int = 25
    OPENAI_MODEL_NAME:str = "gpt5"
    # Add other LLM specific settings as needed
    # ANTHROPIC_API_KEY: Optional[str] = None
    # HUGGINGFACE_HUB_API_TOKEN: Optional[str] = None
    # OLLAMA_BASE_URL: str = "http://localhost:11434" # If using local Ollama

    # --- Vector Store Configuration (Example) ---
    # Replace/add settings for your chosen vector store
    VECTOR_STORE_TYPE: str = "faiss_local" # e.g., "faiss_local", "milvus", "weaviate", "pinecone"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2" # Hugging Face model for embeddings

    # FAISS specific (if using local FAISS)
    FAISS_INDEX_PATH: Path = PROJECT_ROOT / "data" / "vector_store_data" / "faiss_index.bin"
    FAISS_METADATA_PATH: Path = PROJECT_ROOT / "data" / "vector_store_data" / "faiss_metadata.pkl"

    # Milvus specific (example, if using Milvus)
    # MILVUS_HOST: str = "localhost"
    # MILVUS_PORT: str = "19530"
    # MILVUS_COLLECTION_NAME: str = "rag_collection"

    # Weaviate specific (example)
    # WEAVIATE_URL: str = "http://localhost:8080"
    # WEAVIATE_API_KEY: Optional[str] = None # If using Weaviate Cloud Services (WCS)

    # Pinecone specific (example)
    # PINECONE_API_KEY: Optional[str] = None
    # PINECONE_ENVIRONMENT: Optional[str] = None # e.g., "us-west1-gcp"
    # PINECONE_INDEX_NAME: str = "rag-index"


    # --- Database Configuration (Optional, for metadata if not using vector store for it) ---
    # Example for PostgreSQL
    # POSTGRES_SERVER: Optional[str] = "localhost"
    # POSTGRES_PORT: Optional[int] = 5432
    # POSTGRES_USER: Optional[str] = "postgres"
    # POSTGRES_PASSWORD: Optional[str] = "password"
    # POSTGRES_DB: Optional[str] = "rag_app_metadata"
    # DATABASE_URL: Optional[str] = None

    # @validator("DATABASE_URL", pre=True, always=True)
    # def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
    #     if isinstance(v, str):
    #         return v
    #     if values.get("POSTGRES_SERVER") and values.get("POSTGRES_DB"):
    #         return (
    #             f"postgresql+asyncpg://{values['POSTGRES_USER']}:{values['POSTGRES_PASSWORD']}"
    #             f"@{values['POSTGRES_SERVER']}:{values['POSTGRES_PORT']}/{values['POSTGRES_DB']}"
    #         )
    #     return "" # Or raise an error if DB is mandatory

    # --- CORS (Cross-Origin Resource Sharing) ---
    # List of origins that should be permitted to make cross-origin requests.
    # e.g., ["http://localhost:3000", "https://your-frontend-domain.com"]
    # Use ["*"] to allow all origins (less secure, for development only)
    BACKEND_CORS_ORIGINS: list[str] = ["*"] # Adjust for production

    # --- File Upload Configuration ---
    MAX_UPLOAD_FILE_SIZE_MB: int = 25 # Max size for document uploads in MB
    ALLOWED_UPLOAD_FILE_TYPES: list[str] = ["text/plain", "application/pdf", "text/markdown"]

    ## LLM Configuration
    LLM_TEMPERATURE:float = 0.7
    LLM_MAX_TOKENS:int = 1024

    model_config = SettingsConfigDict(
        env_file=".env",          # Specifies the .env file to load (already handled by load_dotenv)
        env_file_encoding="utf-8",
        case_sensitive=False,     # Environment variable names are case-insensitive
        extra="ignore"            # Ignore extra fields from environment variables
    )

# Create a single instance of the settings to be imported by other modules
settings = Settings()

# --- Helper function to ensure directories exist ---
def ensure_directories():
    """
    Ensures that necessary data directories exist.
    Called at application startup.
    """
    if settings.VECTOR_STORE_TYPE == "faiss_local":
        settings.FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        # settings.FAISS_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True) # Same parent as index

# You might call ensure_directories() in your main.py at startup.
# For now, it's defined here.

if __name__ == "__main__":
    # This part is for testing the configuration loading
    print("--- Application Settings ---")
    print(f"App Name: {settings.APP_NAME}")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"OpenAI API Key Loaded: {'Yes' if settings.OPENAI_API_KEY else 'No'}")
    print(f" Model: {settings.OLLAMA_MODEL_NAME}")
    print(f"Vector Store Type: {settings.VECTOR_STORE_TYPE}")
    print(f"Embedding Model: {settings.EMBEDDING_MODEL_NAME}")
    if settings.VECTOR_STORE_TYPE == "faiss_local":
        print(f"FAISS Index Path: {settings.FAISS_INDEX_PATH}")
    # print(f"Database URL: {settings.DATABASE_URL}")
    print(f"CORS Origins: {settings.BACKEND_CORS_ORIGINS}")

    # Example of ensuring FAISS directory exists
    ensure_directories()
    print(f"FAISS directory ensured: {settings.FAISS_INDEX_PATH.parent}")
