# app/core/rag_pipeline.py

from typing import List, Dict, Any, Optional, Tuple

from app.core.config import settings
from app.core.llm_client import BaseLLMClient, get_llm_client
from app.core.vector_store_client import BaseVectorStoreClient, get_vector_store_client, DocumentChunk
from app.core.embedding_manager import EmbeddingManager, get_embedding_manager
from app.api.v1.schemas.chat_schemas import ChatMessage, RetrievedContext # For type hinting

import logging
import re # For basic text cleaning or chunking

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Orchestrates the Retrieval Augmented Generation (RAG) process.
    It handles document chunking, embedding generation, context retrieval,
    and answer generation using an LLM.
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        vector_store_client: Optional[BaseVectorStoreClient] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        default_top_k_retrieval: int = 3, # Number of chunks to retrieve
        chunk_size: int = 500, # Approximate characters per chunk
        chunk_overlap: int = 50 # Characters of overlap between chunks
    ):
        """
        Initializes the RAGPipeline.

        Args:
            llm_client: An instance of a BaseLLMClient. If None, uses get_llm_client().
            vector_store_client: An instance of a BaseVectorStoreClient. If None, uses get_vector_store_client().
            embedding_manager: An instance of EmbeddingManager. If None, uses get_embedding_manager().
            default_top_k_retrieval: Default number of relevant chunks to retrieve.
            chunk_size: Target size for text chunks.
            chunk_overlap: Overlap between consecutive text chunks.
        """
        self.llm_client = llm_client or get_llm_client()
        self.vector_store_client = vector_store_client or get_vector_store_client()
        self.embedding_manager = embedding_manager or get_embedding_manager()
        
        self.default_top_k_retrieval = default_top_k_retrieval
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info(
            f"RAGPipeline initialized with LLM: {self.llm_client.__class__.__name__}, "
            f"VectorStore: {self.vector_store_client.__class__.__name__}, "
            f"EmbeddingManager: {self.embedding_manager.__class__.__name__}"
        )
        # Ensure embedding model is loaded for dimension info if needed by vector store immediately
        # self.embedding_manager.load_model() # Or ensure it's loaded at app startup

    def _simple_text_chunker(self, text: str, document_id: str, doc_metadata: Optional[Dict[str,Any]] = None) -> List[DocumentChunk]:
        """
        A very basic text chunker.
        Splits text by paragraphs or fixed size with overlap.
        A more sophisticated library like LangChain or LlamaIndex would typically be used.

        Args:
            text: The raw text content of the document.
            document_id: The ID of the parent document.
            doc_metadata: Optional metadata from the parent document.

        Returns:
            A list of DocumentChunk objects.
        """
        chunks = []
        if not text:
            return chunks

        # Attempt to split by double newlines (paragraphs) first
        paragraphs = text.split('\n\n')
        current_chunk_text = ""
        chunk_id_counter = 0

        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(current_chunk_text) + len(paragraph) + 1 < self.chunk_size or not current_chunk_text:
                current_chunk_text += (paragraph + "\n") if current_chunk_text else paragraph
            else:
                # Current chunk is full, or adding this paragraph would make it too large
                chunk_id_counter += 1
                chunk_metadata = (doc_metadata or {}).copy()
                chunk_metadata["paragraph_index_start"] = para_idx # Approximate
                chunks.append(DocumentChunk(
                    id=f"{document_id}_chunk_{chunk_id_counter}",
                    document_id=document_id,
                    text=current_chunk_text.strip(),
                    metadata=chunk_metadata
                ))
                # Start new chunk with overlap (naive overlap for this simple chunker)
                overlap_text = current_chunk_text[-self.chunk_overlap:] if self.chunk_overlap > 0 and current_chunk_text else ""
                current_chunk_text = overlap_text + paragraph + "\n"
        
        # Add any remaining text in current_chunk_text
        if current_chunk_text.strip():
            chunk_id_counter += 1
            chunk_metadata = (doc_metadata or {}).copy()
            chunk_metadata["paragraph_index_start"] = len(paragraphs) # Approximate
            chunks.append(DocumentChunk(
                id=f"{document_id}_chunk_{chunk_id_counter}",
                document_id=document_id,
                text=current_chunk_text.strip(),
                metadata=chunk_metadata
            ))
        
        # Fallback/Alternative: Fixed-size chunking if paragraph chunking results in very few/large chunks
        # This is a very basic fixed-size chunker for demonstration.
        if not chunks or any(len(c.text) > self.chunk_size * 1.5 for c in chunks): # If any chunk is too big
            logger.info(f"Using fixed-size chunking for document {document_id} as paragraph chunks are too large or empty.")
            chunks = []
            start = 0
            chunk_id_counter = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_id_counter +=1
                chunk_metadata = (doc_metadata or {}).copy()
                chunk_metadata["char_offset_start"] = start
                chunks.append(DocumentChunk(
                    id=f"{document_id}_chunk_{chunk_id_counter}",
                    document_id=document_id,
                    text=text[start:end],
                    metadata=chunk_metadata
                ))
                start += self.chunk_size - self.chunk_overlap # Move start with overlap
                if start >= end and end < len(text): # Ensure progress if overlap is large
                    start = end


        logger.info(f"Document {document_id} split into {len(chunks)} chunks using simple chunker.")
        return chunks


    async def process_and_embed_document(self, text_content: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[DocumentChunk], int]:
        """
        Processes a document: chunks it, generates embeddings for chunks.
        This method prepares chunks for addition to the vector store but doesn't add them itself.

        Args:
            text_content: The raw text of the document.
            document_id: A unique ID for the document.
            metadata: Optional metadata associated with the document.

        Returns:
            A tuple containing:
                - A list of DocumentChunk objects, each with its text and generated embedding.
                - The number of chunks created.
        """
        logger.info(f"Processing and embedding document_id: {document_id}")
        
        # 1. Chunk the document
        document_chunks = self._simple_text_chunker(text_content, document_id, metadata)
        if not document_chunks:
            logger.warning(f"No chunks created for document_id: {document_id}. Document might be empty or too short.")
            return [], 0

        # 2. Generate embeddings for each chunk
        chunk_texts = [chunk.text for chunk in document_chunks]
        try:
            embeddings = self.embedding_manager.generate_embeddings(chunk_texts)
            for chunk, embedding in zip(document_chunks, embeddings):
                chunk.embedding = embedding # Assign embedding to each chunk object
        except Exception as e:
            logger.error(f"Failed to generate embeddings for document {document_id}: {e}", exc_info=True)
            # Decide how to handle: raise error, or return chunks without embeddings?
            # For now, let's return chunks, they just won't be addable to some vector stores.
            # Or, re-raise to signal failure upstream.
            raise RuntimeError(f"Embedding generation failed for document {document_id}") from e
            
        logger.info(f"Successfully generated embeddings for {len(document_chunks)} chunks of document {document_id}.")
        return document_chunks, len(document_chunks)


    async def retrieve_relevant_context(self, query: str, top_k: Optional[int] = None) -> List[RetrievedContext]:
        """
        Retrieves relevant context chunks from the vector store for a given query.

        Args:
            query: The user's query string.
            top_k: Number of chunks to retrieve. Uses default if None.

        Returns:
            A list of RetrievedContext objects.
        """
        if top_k is None:
            top_k = self.default_top_k_retrieval

        logger.info(f"Retrieving top {top_k} relevant contexts for query: '{query[:100]}...'")
        
        # 1. Generate embedding for the query
        query_embedding = self.embedding_manager.generate_embedding(query)
        
        # 2. Perform similarity search in the vector store
        retrieved_chunks: List[DocumentChunk] = await self.vector_store_client.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k
            # filter_criteria=... # Add if your vector store and use case support it
        )
        
        # 3. Convert DocumentChunk to RetrievedContext schema
        contexts_for_response: List[RetrievedContext] = []
        for chunk in retrieved_chunks:
            contexts_for_response.append(
                RetrievedContext(
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    text=chunk.text,
                    score=chunk.metadata.get('similarity_score'), # Assuming score is in metadata
                    metadata=chunk.metadata
                )
            )
        
        logger.info(f"Retrieved {len(contexts_for_response)} contexts.")
        return contexts_for_response

    def _format_prompt_with_context(self, query: str, context_chunks: List[RetrievedContext], history: Optional[List[ChatMessage]] = None) -> str:
        """
        Formats the LLM prompt by combining the user query, retrieved context, and chat history.
        """
        context_str = "\n\n---\n\n".join([f"Context from document '{c.document_id}' (chunk '{c.chunk_id}', score: {c.score:.2f}):\n{c.text}" for c in context_chunks if c.text])
        
        history_str = ""
        if history:
            history_str = "\nPrevious conversation:\n"
            for msg in history:
                history_str += f"{msg.role.capitalize()}: {msg.content}\n"
            history_str += "\n"

        # Basic prompt template - this can be significantly more sophisticated
        prompt = f"""{history_str}You are a helpful AI assistant. Answer the following question based on the provided context.
If the context doesn't contain the answer, say you don't know or answer based on your general knowledge if appropriate, but indicate that the information was not in the provided context.

Context:
{context_str if context_chunks else "No specific context was retrieved for this query."}

Question:
{query}

Answer:
"""
        logger.debug(f"Formatted LLM Prompt (first 300 chars):\n{prompt[:300]}...")
        return prompt

    async def generate_answer(
        self,
        query: str,
        history: Optional[List[ChatMessage]] = None,
        top_k_retrieval: Optional[int] = None
    ) -> Tuple[str, List[RetrievedContext]]:
        """
        Generates an answer to a query using the RAG process.

        Args:
            query: The user's query.
            history: Optional chat history.
            top_k_retrieval: Number of context chunks to retrieve.

        Returns:
            A tuple containing:
                - The generated answer string.
                - A list of RetrievedContext objects that were used.
        """
        # 1. Retrieve relevant context
        retrieved_contexts = await self.retrieve_relevant_context(query, top_k=top_k_retrieval)
        
        # 2. Format the prompt
        prompt_for_llm = self._format_prompt_with_context(query, retrieved_contexts, history)
        
        # 3. Generate response using LLM
        # You might want to pass specific LLM params from settings or request here
        llm_response_text = await self.llm_client.generate_response(
            prompt=prompt_for_llm,
            # history can be implicitly part of the prompt_for_llm, or passed separately if LLM supports
            temperature=settings.LLM_TEMPERATURE if hasattr(settings, 'LLM_TEMPERATURE') else 0.7,
            max_tokens=settings.LLM_MAX_TOKENS if hasattr(settings, 'LLM_MAX_TOKENS') else 500
        )
        
        logger.info(f"LLM generated answer for query: '{query[:50]}...'")
        return llm_response_text, retrieved_contexts


# Global instance (optional, or manage via DI in services)
# rag_pipeline_instance = RAGPipeline()

# def get_rag_pipeline() -> RAGPipeline:
#     return rag_pipeline_instance
