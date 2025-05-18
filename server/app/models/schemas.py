from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, conlist


class IndexCreate(BaseModel):
    """Schema for creating a new index"""
    name: str = Field(..., description="Name of the index")
    dimension: int = Field(..., description="Dimension of vectors to be stored")
    index_type: str = Field(default="IndexFlatL2", description="Type of FAISS index")
    tenant_id: str = Field(..., description="ID of the tenant")


class IndexInfo(BaseModel):
    """Schema for index information"""
    id: str
    name: str
    dimension: int
    index_type: str
    tenant_id: str
    vector_count: int = 0


class Vector(BaseModel):
    """Schema for a vector with metadata"""
    id: str = Field(..., description="Unique ID for the vector")
    values: List[float] = Field(..., description="Vector values")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Associated metadata")


class VectorBatch(BaseModel):
    """Schema for batch vector operations"""
    vectors: List[Vector] = Field(..., description="List of vectors to process")


class SearchRequest(BaseModel):
    """Schema for search requests"""
    vector: List[float] = Field(..., description="Query vector")
    k: int = Field(default=10, description="Number of results to return")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")


class SearchResult(BaseModel):
    """Schema for search results"""
    id: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Schema for search response"""
    results: List[SearchResult]
