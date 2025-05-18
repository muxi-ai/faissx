from fastapi import APIRouter, Depends, HTTPException, status
from app.models.schemas import IndexCreate, IndexInfo
from app.utils.auth import get_tenant_id, validate_tenant_access
from app.utils.faiss_manager import get_faiss_manager
import os

router = APIRouter()

# Get the singleton FAISS manager
faiss_manager = get_faiss_manager()


@router.post("", response_model=IndexInfo, status_code=status.HTTP_201_CREATED)
async def create_index(
    index_data: IndexCreate, tenant_id: str = Depends(get_tenant_id)
):
    """
    Create a new FAISS index.
    """
    # Validate tenant access (in this case, confirm tenant_id matches)
    validate_tenant_access(tenant_id, index_data.tenant_id)

    # Create index
    index_id = faiss_manager.create_index(
        tenant_id=tenant_id,
        name=index_data.name,
        dimension=index_data.dimension,
        index_type=index_data.index_type,
    )

    # Return index info
    index_info = faiss_manager.get_index_info(tenant_id, index_id)
    if not index_info:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Index created but info not found",
        )

    return index_info


@router.get("/{index_id}", response_model=IndexInfo)
async def get_index(index_id: str, tenant_id: str = Depends(get_tenant_id)):
    """
    Get information about an index.
    """
    # Get index info
    index_info = faiss_manager.get_index_info(tenant_id, index_id)
    if not index_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Index not found"
        )

    # Validate tenant access
    validate_tenant_access(tenant_id, index_info["tenant_id"])

    return index_info


@router.delete("/{index_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_index(index_id: str, tenant_id: str = Depends(get_tenant_id)):
    """
    Delete an index.
    """
    # Get index info to validate tenant access
    index_info = faiss_manager.get_index_info(tenant_id, index_id)
    if not index_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Index not found"
        )

    # Validate tenant access
    validate_tenant_access(tenant_id, index_info["tenant_id"])

    # Delete index
    success = faiss_manager.delete_index(tenant_id, index_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete index",
        )

    return None
