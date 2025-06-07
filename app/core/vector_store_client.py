# app/core/vector_store_client.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING, Union
from pathlib import Path
import numpy as np
import pickle # For saving/loading metadata store with FAISS

from app.core.config import settings
from app.core.embedding_manager import EmbeddingManager, get_embedding_manager # Import EmbeddingManager

import logging
logger = logging.getLogger(__name__)

# This block is only processed by type checkers (like Pylance)
if TYPE_CHECKING:
    import faiss
    # Define the type alias for type checkers
    # You can be general with faiss.Index or specific with Union
    FAISS_INDEX_TYPE = faiss.Index
    # Or, if you want to be extra safe and avoid type checker errors if faiss isn't in your env during dev:
    # from faiss import Index as FaissIndex
    # FAISS_INDEX_TYPE = FaissIndex
else:
    # At runtime, if faiss is not installed, FAISS_INDEX_TYPE will be None
    # This prevents runtime errors related to faiss if you're not using it in this specific execution path
    # But it's important that you don't actually try to INSTANTIATE a FAISS_INDEX_TYPE if faiss is None at runtime
    FAISS_INDEX_TYPE = Union[object, None] # A placeholder type hint that's always valid

# This block handles the actual runtime import
try:
    import faiss
except ImportError:
    faiss = None # Set to None at runtime if not found


# Define a type alias for an embedding
Embedding = List[float]

class DocumentChunk:
    id: str
    document_id: str
    text: str
    embedding: Optional[Embedding] = None
    metadata: Dict[str, Any]

    def __init__(self, id: str, document_id: str, text: str, metadata: Optional[Dict[str, Any]] = None, embedding: Optional[Embedding] = None):
        self.id = id
        self.document_id = document_id
        self.text = text
        self.metadata = metadata or {}
        self.embedding = embedding

class BaseVectorStoreClient(ABC):
    @abstractmethod
    async def initialize(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    async def add_documents(self, documents: List[DocumentChunk]) -> List[str]:
        pass

    @abstractmethod
    async def similarity_search(
        self, query_embedding: Embedding, top_k: int = 5, filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        pass

    @abstractmethod
    async def delete_documents_by_id(self, document_ids: List[str]) -> bool:
        pass

    @abstractmethod
    async def delete_chunks_by_id(self, chunk_ids: List[str]) -> bool:
        pass

    @abstractmethod
    async def get_document_chunk_count(self, document_id: Optional[str] = None) -> int:
        pass


class MockVectorStoreClient(BaseVectorStoreClient):
    def __init__(self):
        self._store: Dict[str, DocumentChunk] = {}
        self._doc_to_chunks: Dict[str, List[str]] = {}
        logger.info("MockVectorStoreClient initialized.")

    async def initialize(self, *args, **kwargs) -> None: logger.info("MockVectorStoreClient: No specific initialization required.")
    async def add_documents(self, documents: List[DocumentChunk]) -> List[str]:
        added_ids = []
        for doc_chunk in documents:
            if not doc_chunk.embedding: continue
            self._store[doc_chunk.id] = doc_chunk
            self._doc_to_chunks.setdefault(doc_chunk.document_id, []).append(doc_chunk.id)
            added_ids.append(doc_chunk.id)
        return added_ids
    async def similarity_search(self, query_embedding: Embedding, top_k: int = 5, filter_criteria: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        if not self._store: return []
        all_chunks = [c for c in self._store.values() if c.embedding]
        results = []
        for i, chunk in enumerate(all_chunks[:top_k]):
            rc = DocumentChunk(id=chunk.id,document_id=chunk.document_id,text=chunk.text,metadata=chunk.metadata.copy(),embedding=chunk.embedding)
            rc.metadata['similarity_score'] = 1.0-(i*0.05); results.append(rc)
        return results
    async def delete_documents_by_id(self, document_ids: List[str]) -> bool:
        deleted = False
        for doc_id in document_ids:
            if doc_id in self._doc_to_chunks:
                for cid in self._doc_to_chunks.pop(doc_id,[]): self._store.pop(cid,None); deleted=True
        return deleted
    async def delete_chunks_by_id(self, chunk_ids: List[str]) -> bool:
        deleted = False
        for cid in chunk_ids:
            if cid in self._store:
                doc_id = self._store.pop(cid).document_id; deleted=True
                if doc_id in self._doc_to_chunks and cid in self._doc_to_chunks[doc_id]:
                    self._doc_to_chunks[doc_id].remove(cid)
                    if not self._doc_to_chunks[doc_id]: self._doc_to_chunks.pop(doc_id,None)
        return deleted
    async def get_document_chunk_count(self, document_id: Optional[str]=None) -> int:
        if document_id: return len(self._doc_to_chunks.get(document_id,[]))
        return len(self._store)


class FaissVectorStoreClient(BaseVectorStoreClient):
    index: Optional[FAISS_INDEX_TYPE] = None # Explicit type hint for the FAISS index

    def __init__(self, index_path: Path, metadata_path: Path, embedding_manager: EmbeddingManager):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_manager = embedding_manager
        self.embedding_dim: Optional[int] = None
        self.metadata_store: Dict[int, DocumentChunk] = {}
        self._is_initialized = False
        logger.info(f"FaissVectorStoreClient created. Index path: {index_path}, Metadata path: {metadata_path}")

    async def initialize(self, *args, **kwargs) -> None:
        if self._is_initialized:
            logger.info("FaissVectorStoreClient already initialized.")
            return

        global faiss # Make faiss available if imported at module level
        if faiss is None: # Check if module-level import failed
            try:
                import faiss as faiss_local_import
                faiss = faiss_local_import # Assign to global faiss if successful
            except ImportError:
                logger.error("FAISS package not installed. Please run 'poetry add faiss-cpu' (or 'faiss-gpu')")
                raise ImportError("FAISS package not found.")

        try:
            self.embedding_manager.load_model()
            self.embedding_dim = self.embedding_manager.get_embedding_dimension()
            logger.info(f"Using embedding dimension: {self.embedding_dim} from EmbeddingManager.")
        except Exception as e:
            logger.error(f"Failed to get embedding dimension from EmbeddingManager: {e}", exc_info=True)
            raise RuntimeError(f"Could not determine embedding dimension for FAISS: {e}") from e

        if self.index_path.exists() and self.metadata_path.exists():
            logger.info("Loading existing FAISS index and metadata...")
            try:
                self.index = faiss.read_index(str(self.index_path))
                if self.index is None: raise Exception("Received None self.index")
                with open(self.metadata_path, "rb") as f: self.metadata_store = pickle.load(f)
                if self.index.d != self.embedding_dim:
                    logger.warning(f"Loaded FAISS index dimension ({self.index.d}) != current embedding model dimension ({self.embedding_dim}).")
                logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors) and metadata ({len(self.metadata_store)} entries).")
            except Exception as e:
                logger.error(f"Error loading FAISS data: {e}. Initializing new store.", exc_info=True)
                self._initialize_new_store(faiss)
        else:
            logger.info("No existing FAISS data found. Initializing new store.")
            self._initialize_new_store(faiss)
        self._is_initialized = True

    def _initialize_new_store(self, faiss_module_ref): # Pass faiss module
        if self.embedding_dim is None: 
            raise RuntimeError("Embedding dimension not set for new FAISS store.")
        self.index = faiss_module_ref.IndexFlatL2(self.embedding_dim)
        self.metadata_store = {}
        logger.info(f"Initialized new FAISS store with dimension {self.embedding_dim}.")

    def _save_store(self):
        if not self._is_initialized or self.index is None:
            logger.error("Attempted to save uninitialized FAISS store."); return
        global faiss
        if faiss is None: logger.error("FAISS module not available for saving."); return
        logger.info(f"Saving FAISS index to {self.index_path} and metadata to {self.metadata_path}")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as f: pickle.dump(self.metadata_store, f)
        logger.info("FAISS index and metadata saved.")

    async def add_documents(self, documents: List[DocumentChunk]) -> List[str]:
        if not self._is_initialized: await self.initialize()
        assert self.index is not None, "FAISS index is not initialized."

        embeddings_for_faiss = []
        original_chunk_ids_added = []
        
        for chunk in documents:
            if chunk.embedding:
                embeddings_for_faiss.append(np.array(chunk.embedding, dtype=np.float32))
                original_chunk_ids_added.append(chunk.id)
            else:
                logger.warning(f"Chunk {chunk.id} (doc: {chunk.document_id}) has no embedding. Skipping.")
        
        if embeddings_for_faiss:
            # Ensure final_embeddings_array is 2D and C-contiguous
            np_array_of_embeddings = np.array(embeddings_for_faiss, dtype=np.float32)
            if np_array_of_embeddings.ndim == 1: # Single embedding
                np_array_of_embeddings = np_array_of_embeddings.reshape(1, -1)
            
            # Ensure C-style contiguity, FAISS C++ backend expects this.
            final_embeddings_array_contiguous = np.ascontiguousarray(np_array_of_embeddings, dtype=np.float32)

            start_index = self.index.ntotal
            self.index.add(final_embeddings_array_contiguous) # type: ignore # Use the contiguous array
            
            for i, original_id in enumerate(original_chunk_ids_added):
                original_chunk_obj = next((doc for doc in documents if doc.id == original_id), None)
                if original_chunk_obj: self.metadata_store[start_index + i] = original_chunk_obj
                else: logger.error(f"Consistency error: Original chunk for ID {original_id} not found.")
            self._save_store()
            logger.info(f"Added {len(embeddings_for_faiss)} chunks to FAISS. Total vectors: {self.index.ntotal}")
        return original_chunk_ids_added

    async def similarity_search(
        self, query_embedding: Embedding, top_k: int = 5, filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        if not self._is_initialized: await self.initialize()
        assert self.index is not None, "FAISS index not initialized for search."
        if self.index.ntotal == 0: logger.info("FAISS index is empty."); return []

        query_np = np.ascontiguousarray(np.array([query_embedding], dtype=np.float32))
        distances, indices = self.index.search(query_np, top_k) # type: ignore
        
        results = []
        for i, faiss_idx in enumerate(indices[0]):
            if faiss_idx == -1: continue
            if faiss_idx in self.metadata_store:
                data = self.metadata_store[faiss_idx]
                rc = DocumentChunk(id=data.id,document_id=data.document_id,text=data.text,metadata=data.metadata.copy(),embedding=None)
                rc.metadata['similarity_score_l2_distance'] = float(distances[0][i]); results.append(rc)
            else: logger.warning(f"Index {faiss_idx} from FAISS search not in metadata_store.")
        return results

    async def delete_documents_by_id(self, document_ids: List[str]) -> bool:
        if not self._is_initialized: await self.initialize()
        assert self.index is not None, "FAISS index not initialized for deletion."
        global faiss; # Ensure faiss module is accessible
        if faiss is None: logger.error("FAISS module not loaded, cannot delete."); return False
        
        logger.warning("FaissVectorStoreClient: Rebuilding index for deletion by document ID.")
        indices_to_remove = {idx for idx, cd in self.metadata_store.items() if cd.document_id in document_ids}
        if not indices_to_remove: logger.info("No chunks for given document IDs to delete."); return False

        new_meta, new_embeds = {}, []
        for old_idx in sorted(self.metadata_store.keys()):
            if old_idx not in indices_to_remove:
                cd = self.metadata_store[old_idx]
                if cd.embedding: new_embeds.append(np.array(cd.embedding,dtype=np.float32)); new_meta[len(new_embeds)-1]=cd
        
        if not new_embeds: self._initialize_new_store(faiss)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(np.ascontiguousarray(np.array(new_embeds, dtype=np.float32))) # type: ignore
            self.metadata_store = new_meta
        self._save_store()
        logger.info(f"Rebuilt FAISS index after deletion. New total: {self.index.ntotal if self.index else 0}")
        return True

    async def delete_chunks_by_id(self, chunk_ids: List[str]) -> bool:
        logger.warning("FaissVectorStoreClient: Deleting by individual chunk IDs is inefficient (requires index rebuild).")
        # Similar logic to delete_documents_by_id but filter by chunk_data.id in chunk_ids
        return False # Placeholder for full implementation

    async def get_document_chunk_count(self, document_id: Optional[str] = None) -> int:
        if not self._is_initialized: await self.initialize()
        if self.index is None: return 0
        if document_id: return sum(1 for cd in self.metadata_store.values() if cd.document_id == document_id)
        return self.index.ntotal


def get_vector_store_client() -> BaseVectorStoreClient:
    store_type = settings.VECTOR_STORE_TYPE.lower()
    emb_manager = get_embedding_manager() 

    if store_type == "faiss_local":
        logger.info(f"Configuring FaissVectorStoreClient. Index: {settings.FAISS_INDEX_PATH}, Meta: {settings.FAISS_METADATA_PATH}")
        client = FaissVectorStoreClient(
            index_path=settings.FAISS_INDEX_PATH,
            metadata_path=settings.FAISS_METADATA_PATH,
            embedding_manager=emb_manager
        )
        return client
    elif store_type == "mock":
        logger.info(f"Using MockVectorStoreClient for VectorStore.")
        return MockVectorStoreClient()
    else:
        logger.error(f"Unsupported VECTOR_STORE_TYPE: {store_type}. Falling back to MockVectorStoreClient.")
        return MockVectorStoreClient()

