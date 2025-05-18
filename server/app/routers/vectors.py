from fastapi import APIRouter, Depends, HTTPException, status, Path
from app.models.schemas import VectorBatch, SearchRequest, SearchResponse
from app.utils.auth import get_tenant_id, validate_tenant_access
from app.utils.faiss_manager import get_faiss_manager

router = APIRouter()

# Get the singleton FAISS manager
faiss_manager = get_faiss_manager()


@router.post("/v1/index/{index_id}/vectors", status_code=status.HTTP_201_CREATED)
async def add_vectors(
    index_id: str = Path(..., description="ID of the index"),
    vectors: VectorBatch = ...,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Add vectors to an index.
    """
    # Get index info to validate tenant access
    index_info = faiss_manager.get_index_info(tenant_id, index_id)
    if not index_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Index not found"
        )

    # Validate tenant access
    validate_tenant_access(tenant_id, index_info["tenant_id"])

    # Add vectors
    success_list = faiss_manager.add_vectors(tenant_id, index_id, vectors.vectors)

    # Return result
    return {
        "success": all(success_list),
        "added_count": sum(success_list),
        "failed_count": len(success_list) - sum(success_list),
    }


@router.post("/v1/index/{index_id}/search", response_model=SearchResponse)
async def search_vectors(
    index_id: str = Path(..., description="ID of the index"),
    search_request: SearchRequest = ...,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Search for similar vectors in an index.
    """
    # Get index info to validate tenant access
    index_info = faiss_manager.get_index_info(tenant_id, index_id)
    if not index_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Index not found"
        )

    # Validate tenant access
    validate_tenant_access(tenant_id, index_info["tenant_id"])

    # Search
    results = faiss_manager.search(
        tenant_id=tenant_id,
        index_id=index_id,
        vector=search_request.vector,
        k=search_request.k,
        filter_metadata=search_request.filter,
    )

    # Format response
    return SearchResponse(results=results)


# Keep the GET endpoint for backward compatibility
@router.get("/v1/index/{index_id}/search", response_model=SearchResponse)
async def search_vectors_get(
    index_id: str = Path(..., description="ID of the index"),
    search_request: SearchRequest = ...,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Search for similar vectors in an index (GET method).
    """
    return await search_vectors(index_id, search_request, tenant_id)


@router.delete(
    "/v1/index/{index_id}/vectors/{vector_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_vector(
    index_id: str = Path(..., description="ID of the index"),
    vector_id: str = Path(..., description="ID of the vector"),
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Delete a vector from an index.
    """
    # Get index info to validate tenant access
    index_info = faiss_manager.get_index_info(tenant_id, index_id)
    if not index_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Index not found"
        )

    # Validate tenant access
    validate_tenant_access(tenant_id, index_info["tenant_id"])

    # Delete vector
    success = faiss_manager.delete_vector(tenant_id, index_id, vector_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Vector not found"
        )

    return None
