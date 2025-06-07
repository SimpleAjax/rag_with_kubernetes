# app/services/document_service.py

from pathlib import Path
from typing import Optional
from fastapi import UploadFile, HTTPException, status
from app.api.v1.schemas.document_schemas import (
    DocumentCreate,
    DocumentResponse,
    DocumentListResponse,
    DocumentUpdateRequest,
    DocumentMetadata as DocumentMetadataSchema
)
from app.core.config import settings, PROJECT_ROOT
# from app.core.rag_pipeline import RAGPipeline # For document processing (chunking, embedding)
# from app.core.vector_store_client import VectorStoreClient # To store embeddings
# from app.models.db_models import Document as DocumentDBModel # If using a DB for metadata

import logging
import uuid
from datetime import datetime
import os # For file operations in mock
import shutil # For file operations in mock
import asyncio # For background processing simulation

logger = logging.getLogger(__name__)

# --- Mock Document Storage ---
# In a real application, this would interact with a database and a file storage system (like S3 or local disk),
# and the vector store.
MOCK_DB_DOCUMENTS_SERVICE: dict[str, DocumentResponse] = {} # Stores DocumentResponse like objects
MOCK_UPLOAD_DIR = PROJECT_ROOT / "data" / "uploaded_files_mock"
MOCK_UPLOAD_DIR.mkdir(parents=True, exist_ok=True) # Ensure mock upload dir exists


class DocumentService:
    def __init__(self):
        """
        Initializes the DocumentService.
        In a real application, you would initialize your RAG pipeline (for processing),
        Vector Store client, and potentially a database client for metadata.
        
        Example:
        self.rag_pipeline = RAGPipeline(...)
        self.vector_store = VectorStoreClient(...)
        # self.db_session_factory = get_db_session_factory() # If using SQLAlchemy
        """
        logger.info("DocumentService initialized (mock implementation).")
        # For now, we are using mock logic.

    async def _process_document_background(self, doc_id: str, file_path: Path, original_filename: str):
        """
        Simulates background processing of the document (chunking, embedding, indexing).
        """
        logger.info(f"Background task: Starting processing for document ID {doc_id} ({original_filename}) from path {file_path}")
        try:
            # Simulate reading content (in a real app, you'd read from file_path)
            # with open(file_path, "rb") as f:
            #     content_bytes = f.read()
            # text_content = content_bytes.decode("utf-8") # Assuming text, handle other types

            # 1. Chunk the document content
            #    chunks = self.rag_pipeline.chunk_document(text_content)
            mock_chunk_count = 5 # Simulate 5 chunks
            await asyncio.sleep(2) # Simulate chunking time

            # 2. Generate embeddings for each chunk
            #    embeddings = self.rag_pipeline.generate_embeddings_for_chunks(chunks)
            await asyncio.sleep(1) # Simulate embedding time

            # 3. Store chunks and their embeddings in the vector store
            #    self.vector_store.add_documents(chunks_with_embeddings, document_id=doc_id)
            await asyncio.sleep(1) # Simulate indexing time

            # 4. Update document status and metadata in the database (or mock DB)
            if doc_id in MOCK_DB_DOCUMENTS_SERVICE:
                MOCK_DB_DOCUMENTS_SERVICE[doc_id].status = "completed"
                MOCK_DB_DOCUMENTS_SERVICE[doc_id].chunk_count = mock_chunk_count
                MOCK_DB_DOCUMENTS_SERVICE[doc_id].updated_at = datetime.utcnow()
                logger.info(f"Background task: Processing COMPLETED for document ID {doc_id}. Status: completed, Chunks: {mock_chunk_count}")
            else:
                logger.warning(f"Background task: Document ID {doc_id} not found in mock DB after processing.")

        except Exception as e:
            logger.error(f"Background task: Error processing document ID {doc_id}: {e}", exc_info=True)
            if doc_id in MOCK_DB_DOCUMENTS_SERVICE:
                MOCK_DB_DOCUMENTS_SERVICE[doc_id].status = "failed"
                MOCK_DB_DOCUMENTS_SERVICE[doc_id].error_message = str(e)
                MOCK_DB_DOCUMENTS_SERVICE[doc_id].updated_at = datetime.utcnow()
        finally:
            # Clean up the temporary file if it's no longer needed
            # try:
            #     if file_path.exists():
            #         os.remove(file_path)
            #         logger.info(f"Background task: Cleaned up temporary file {file_path}")
            # except OSError as e_os:
            #     logger.error(f"Background task: Error cleaning up temporary file {file_path}: {e_os}")
            pass


    async def create_document(self, file: UploadFile, metadata_schema: DocumentCreate) -> DocumentResponse:
        doc_id = str(uuid.uuid4())
        original_filename = file.filename or f"untitled_{doc_id}"
        logger.info(f"Creating document '{original_filename}' with proposed id {doc_id}")

        # Basic file size check
        if file.size is not None and file.size > settings.MAX_UPLOAD_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds limit of {settings.MAX_UPLOAD_FILE_SIZE_MB}MB."
            )
        # Basic file type check
        if file.content_type not in settings.ALLOWED_UPLOAD_FILE_TYPES:
             raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: {file.content_type}. Allowed types: {settings.ALLOWED_UPLOAD_FILE_TYPES}"
            )

        # Save file temporarily for processing (mock)
        # In a real system, you might stream to S3 or a persistent volume.
        temp_file_path = MOCK_UPLOAD_DIR / f"{doc_id}_{original_filename}"
        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"Failed to save uploaded file temporarily: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not save uploaded file.")
        finally:
            await file.close()

        doc_response = DocumentResponse(
            id=doc_id,
            title=metadata_schema.title or original_filename,
            metadata=metadata_schema.metadata or DocumentMetadataSchema(source_name=original_filename, source_url=None),
            status="pending_processing",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            chunk_count=1,
            error_message=None
        )
        MOCK_DB_DOCUMENTS_SERVICE[doc_id] = doc_response
        logger.info(f"Document {doc_id} ('{original_filename}') entry created. Status: pending_processing.")

        # Trigger background processing
        asyncio.create_task(self._process_document_background(doc_id, temp_file_path, original_filename))
        logger.info(f"Scheduled background processing for document {doc_id}.")
        
        return doc_response

    async def get_document_by_id(self, doc_id: str) -> Optional[DocumentResponse]:
        logger.info(f"Retrieving document by id: {doc_id}")
        return MOCK_DB_DOCUMENTS_SERVICE.get(doc_id)

    async def list_documents(self, skip: int = 0, limit: int = 10) -> DocumentListResponse:
        logger.info(f"Listing documents (skip={skip}, limit={limit})")
        all_docs_sorted = sorted(
            MOCK_DB_DOCUMENTS_SERVICE.values(), 
            key=lambda doc: doc.created_at, 
            reverse=True
        )
        paginated_docs = all_docs_sorted[skip : skip + limit]
        return DocumentListResponse(
            items=paginated_docs,
            total=len(all_docs_sorted),
            page=(skip // limit) + 1 if limit > 0 else 1,
            size=len(paginated_docs)
        )

    async def update_document_metadata(self, doc_id: str, update_data: DocumentUpdateRequest) -> Optional[DocumentResponse]:
        logger.info(f"Updating metadata for document id: {doc_id}")
        doc = MOCK_DB_DOCUMENTS_SERVICE.get(doc_id)
        if not doc:
            return None
        
        update_fields = update_data.model_dump(exclude_unset=True)
        if "title" in update_fields:
            doc.title = update_fields["title"]
        if "metadata" in update_fields: # Pydantic model for metadata
            # This assumes metadata is fully replaced if provided.
            # For partial updates to metadata, you'd need more complex logic.
            doc.metadata = DocumentMetadataSchema(**update_fields["metadata"])

        doc.updated_at = datetime.utcnow()
        MOCK_DB_DOCUMENTS_SERVICE[doc_id] = doc # Ensure the mock DB has the updated object
        return doc

    async def delete_document(self, doc_id: str) -> bool:
        logger.info(f"Deleting document with id: {doc_id}")
        if doc_id in MOCK_DB_DOCUMENTS_SERVICE:
            del MOCK_DB_DOCUMENTS_SERVICE[doc_id]
            # Also delete the mock uploaded file
            # In a real system, delete from file storage and vector store
            mock_file_to_delete = next((f for f in MOCK_UPLOAD_DIR.iterdir() if f.name.startswith(doc_id)), None)
            if mock_file_to_delete and mock_file_to_delete.exists():
                try:
                    os.remove(mock_file_to_delete)
                    logger.info(f"Deleted mock file: {mock_file_to_delete}")
                except OSError as e:
                    logger.error(f"Error deleting mock file {mock_file_to_delete}: {e}")
            return True
        return False

