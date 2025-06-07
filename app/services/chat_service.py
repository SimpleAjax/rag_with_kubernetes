# app/services/chat_service.py

from app.api.v1.schemas.chat_schemas import ChatRequest, ChatResponse, RetrievedContext, ChatMessage
from app.core.config import settings
from app.core.rag_pipeline import RAGPipeline # Import the RAGPipeline
# from app.core.llm_client import get_llm_client # RAGPipeline handles its own clients by default
# from app.core.vector_store_client import get_vector_store_client
# from app.core.embedding_manager import get_embedding_manager

import logging
import uuid

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        """
        Initializes the ChatService.
        It now instantiates or gets an instance of RAGPipeline.
        """
        # Instantiate RAGPipeline. It will use its default internal clients
        # (LLM, VectorStore, EmbeddingManager) based on their factory functions.
        # If you need to pass specific pre-configured clients, you can do so here:
        # self.rag_pipeline = RAGPipeline(
        #     llm_client=get_llm_client(),
        #     vector_store_client=get_vector_store_client(),
        #     embedding_manager=get_embedding_manager()
        # )
        self.rag_pipeline = RAGPipeline()
        logger.info(f"ChatService initialized with RAGPipeline using: "
                    f"LLM: {self.rag_pipeline.llm_client.__class__.__name__}, "
                    f"VectorStore: {self.rag_pipeline.vector_store_client.__class__.__name__}, "
                    f"EmbeddingManager: {self.rag_pipeline.embedding_manager.__class__.__name__}")

    async def process_chat_query(self, request: ChatRequest) -> ChatResponse:
        """
        Processes a chat query using the RAG pipeline.
        
        Args:
            request: The ChatRequest object containing the user's query and context.
            
        Returns:
            A ChatResponse object with the generated answer and retrieved context.
        """
        logger.info(f"ChatService: Processing chat query for session_id: {request.session_id} with query: '{request.query}'")

        try:
            # Use the RAG pipeline to generate an answer
            answer_text, retrieved_contexts = await self.rag_pipeline.generate_answer(
                query=request.query,
                history=request.history, # Pass history to the RAG pipeline
                # top_k_retrieval can be passed if needed, otherwise RAGPipeline default is used
            )

            logger.info(f"ChatService: RAGPipeline generated answer for session_id: {request.session_id}")

            return ChatResponse(
                session_id=request.session_id,
                answer=answer_text,
                retrieved_context=retrieved_contexts # These are already List[RetrievedContext]
            )
        except Exception as e:
            logger.error(f"ChatService: Error processing chat query with RAGPipeline: {e}", exc_info=True)
            # Re-raise the exception to be handled by the API endpoint or a global exception handler
            # Or, you could return a ChatResponse with an error field populated.
            # For now, let's re-raise to make errors visible at the API level.
            raise # This allows HTTPException from deeper layers to propagate

    async def get_session_history(self, session_id: str) -> list[ChatMessage]:
        """
        (Optional) Retrieves chat history for a given session.
        This might involve a cache or a database in a real application.
        For now, this remains a mock.
        """
        logger.info(f"ChatService: Retrieving history for session_id: {session_id} (mock implementation).")
        # Mock implementation - in a real app, this would fetch from a persistent store
        # or a session management system.
        return []

    async def clear_session_history(self, session_id: str) -> bool:
        """
        (Optional) Clears chat history for a given session.
        For now, this remains a mock.
        """
        logger.info(f"ChatService: Clearing history for session_id: {session_id} (mock implementation).")
        # Mock implementation
        return True
