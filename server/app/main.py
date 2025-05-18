from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import index, vectors, admin

# Initialize FastAPI app
app = FastAPI(
    title="FAISS Proxy",
    description="A lightweight proxy service for FAISS vector operations",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(index.router, prefix="/v1/index", tags=["Index Management"])
app.include_router(vectors.router, tags=["Vector Operations"])
app.include_router(admin.router, prefix="/v1", tags=["Administration"])


@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint to check if the service is running"""
    return {"status": "ok", "message": "FAISS Proxy is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
