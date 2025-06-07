# app/api/v1/endpoints/documents.py

from fastapi import (
    APIRouter,
    Body,
    HTTPException,
    Depends,
    UploadFile,
    File,
    Form,
    Query,
    Path,
    status
)
from typing import List, Optional, Annotated
from datetime import datetime
import uuid
from app.services.document_service import DocumentService

from pydantic import HttpUrl # For generating IDs

# Import schemas
from app.api.v1.schemas.document_schemas import (
    DocumentCreate,
    DocumentResponse,
    DocumentListResponse,
    DocumentUpdateRequest,
    DocumentMetadata as DocumentMetadataSchema # Alias to avoid conflict
)
from app.core.config import settings # For file upload settings

# Placeholder for service layer dependency - to be implemented later
# from app.services.document_service import DocumentService
# from app.core.dependencies import get_document_service

import logging
logger = logging.getLogger(__name__)

router = APIRouter()


# Dependency function (placeholder)
async def get_document_service():
    return DocumentService()


# --- API Endpoints ---

@router.post(
    "/",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED, # Accepted for processing
    summary="Upload a new document",
    description="Uploads a document file (e.g., .txt, .pdf, .md) for ingestion into the RAG system. "
                "Metadata can be provided as form data."
)
async def upload_document_endpoint(
    # Use Annotated for modern FastAPI
    # For file uploads with metadata, FastAPI typically uses Form data alongside File.
    # The Pydantic model for metadata needs to be converted to Form fields.
    # A simpler approach for complex metadata is to have a JSON body for metadata
    # and the file as a separate part, but FastAPI's default for `UploadFile` with
    # other Pydantic models is often to expect them as Form fields.
    # Let's try with Form(...) for metadata fields.

    file: Annotated[UploadFile, File(description="The document file to upload.")],
    # document_service: Annotated[DocumentService, Depends(get_document_service)] # Real dependency
    document_service: Annotated[DocumentService, Depends(get_document_service)], # Mock dependency

    # Option 1: Individual Form fields for metadata (simpler for basic metadata)
    title: Annotated[Optional[str], Form()] = None,
    source_name: Annotated[Optional[str], Form()] = None,
    source_url: Annotated[Optional[HttpUrl], Form()] = None
    # custom_tags: Annotated[Optional[str], Form()] = None, # JSON string for dict
):
    """
    Handles document uploads.
    - `file`: The document file.
    - `title`, `source_name`, `source_url`: Optional metadata fields.
    """
    try:
        logger.info(f"Received document upload request for file: {file.filename}")
        
        # Construct metadata schema from Form fields
        # For custom_tags, if it's a JSON string:
        # parsed_custom_tags = json.loads(custom_tags) if custom_tags else {}
        doc_metadata = DocumentMetadataSchema(
            source_name=source_name or file.filename, # Default to filename if not provided
            source_url=source_url,
            # custom_tags=parsed_custom_tags
        )
        doc_create_schema = DocumentCreate(
            title=title,
            metadata=doc_metadata
        )

        response = await document_service.create_document   (file, doc_create_schema)
        return response
    except HTTPException:
        raise # Re-raise HTTPException directly (e.g. from service validation)
    except Exception as e:
        logger.error(f"Error uploading document '{file.filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during file upload: {str(e)}"
        )

@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
    description="Retrieves a paginated list of all documents in the system."
)
async def list_documents_endpoint(
    document_service: Annotated[DocumentService, Depends(get_document_service)],
    page: Annotated[int, Query(ge=1, description="Page number to retrieve.")] = 1,
    size: Annotated[int, Query(ge=1, le=100, description="Number of documents per page.")] = 10
):
    skip = (page - 1) * size
    return await document_service.list_documents(skip=skip, limit=size)

@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a specific document by ID",
    description="Retrieves details and status of a specific document."
)
async def get_document_endpoint(
    document_id: Annotated[str, Path(description="The ID of the document to retrieve.")],
    document_service: Annotated[DocumentService, Depends(get_document_service)]
):
    doc = await document_service.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return doc

@router.patch(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Update document metadata",
    description="Updates the metadata (e.g., title, custom tags) of an existing document."
)
async def update_document_endpoint(
    document_id: Annotated[str, Path(description="The ID of the document to update.")],
    update_data: Annotated[DocumentUpdateRequest, Body(...)],
    document_service: Annotated[DocumentService, Depends(get_document_service)]
):
    updated_doc = await document_service.update_document_metadata(document_id, update_data)
    if not updated_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found for update.")
    return updated_doc


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
    description="Deletes a document and its associated data from the system."
)
async def delete_document_endpoint(
    document_id: Annotated[str, Path(description="The ID of the document to delete.")],
    document_service: Annotated[DocumentService, Depends(get_document_service)]
):
    success = await document_service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found for deletion.")
    return None # FastAPI handles 204 No Content response automatically

