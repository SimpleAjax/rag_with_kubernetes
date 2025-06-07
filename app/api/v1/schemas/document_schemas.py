# app/api/v1/schemas/document_schemas.py

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

class DocumentMetadata(BaseModel):
    """
    Schema for document metadata.
    """
    source_name: Optional[str] = Field(None, description="Original name of the uploaded file or source (e.g., 'annual_report_2023.pdf').")
    source_url: Optional[HttpUrl] = Field(None, description="URL if the document was fetched from a web source.")
    custom_tags: Optional[Dict[str, str]] = Field(default_factory=dict, description="User-defined tags for categorization or filtering.")
    # Add any other relevant metadata fields

class DocumentBase(BaseModel):
    """
    Base schema for document properties.
    """
    title: Optional[str] = Field(None, description="Optional title for the document.")
    metadata: Optional[DocumentMetadata] = Field(default_factory=DocumentMetadata, description="Metadata associated with the document.") # type: ignore

class DocumentCreate(DocumentBase):
    """
    Schema for creating a new document (metadata part).
    The actual file content will be handled via UploadFile.
    """
    # Content is handled by UploadFile in the endpoint, this schema is for metadata
    pass

class DocumentResponse(DocumentBase):
    """
    Schema for representing a document in API responses.
    """
    id: str = Field(..., description="Unique identifier for the document in the system.")
    status: str = Field(default="processing", description="Processing status of the document (e.g., 'pending', 'processing', 'completed', 'failed').")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of document creation.")
    updated_at: Optional[datetime] = Field(None, description="Timestamp of last update.")
    chunk_count: Optional[int] = Field(None, description="Number of chunks the document was split into.")
    error_message: Optional[str] = Field(None, description="Error message if processing failed.")

    class Config:
        from_attributes = True # For Pydantic V2, was orm_mode = True in V1
                               # Allows creating this model from ORM objects or other attribute-based objects.

class DocumentListResponse(BaseModel):
    """
    Schema for a list of documents with pagination details.
    """
    items: List[DocumentResponse]
    total: int
    page: int
    size: int
    # pages: int # Optional: total number of pages

class DocumentUpdateRequest(BaseModel):
    """
    Schema for updating document metadata.
    """
    title: Optional[str] = Field(None, description="New title for the document.")
    metadata: Optional[DocumentMetadata] = Field(None, description="New metadata for the document.")
    # Note: Re-processing or re-chunking would likely be a separate, more complex operation.
