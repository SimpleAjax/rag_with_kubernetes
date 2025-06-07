# app/api/v1/schemas/chat_schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid # For unique session IDs

class ChatMessage(BaseModel):
    """
    Represents a single message in a chat history.
    """
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant', 'system').")
    content: str = Field(..., description="Content of the message.")

class ChatRequest(BaseModel):
    """
    Schema for a chat request.
    """
    query: str = Field(..., min_length=1, description="The user's query or message.")
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique session ID for maintaining conversation context. If not provided, a new one will be generated.")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Optional chat history to provide context.")
    # You might add other parameters like temperature, max_tokens for LLM, etc.
    # stream: bool = Field(default=False, description="Whether to stream the response.") # For later implementation

class RetrievedContext(BaseModel):
    """
    Represents a piece of retrieved context used for generation.
    """
    document_id: Optional[str] = Field(None, description="ID of the source document.")
    chunk_id: Optional[str] = Field(None, description="ID of the specific chunk within the document.")
    text: str = Field(..., description="The actual text content of the retrieved chunk.")
    score: Optional[float] = Field(None, description="Relevance score of the retrieved chunk.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Any other metadata associated with the chunk.")


class ChatResponse(BaseModel):
    """
    Schema for a chat response.
    """
    session_id: Optional[str] = Field(..., description="Unique session ID for the conversation.")
    answer: str = Field(..., description="The generated answer from the RAG system.")
    retrieved_context: Optional[List[RetrievedContext]] = Field(default_factory=list, description="List of context chunks retrieved and used for generating the answer.")
    # raw_llm_response: Optional[Dict[str, Any]] = Field(None, description="Raw response from the LLM for debugging or advanced use.")
    # error: Optional[str] = Field(None, description="Error message if the request failed.")

# Example for a streaming response (more complex, for future consideration)
# class StreamedChatChunk(BaseModel):
#     session_id: str
#     chunk_type: str # e.g., "text_delta", "context_info", "error", "end_of_stream"
#     payload: Any
