import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models.database import create_tables
from .routes.classification import router as classification_router

# Configure logging to work with uvicorn
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Get logger for this module
logger = logging.getLogger(__name__)

# Ensure uvicorn loggers show our messages
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.setLevel(logging.INFO)
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.setLevel(logging.INFO)

# Create database tables
create_tables()
logger.info("Database tables created/verified")

app = FastAPI(
    title="News Classification API",
    description="API for classifying news articles as real or fake using BERT and LLM models",
    version="1.0.0",
)

logger.info("FastAPI application initialized")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(classification_router, prefix="/api", tags=["classification"])


@app.get("/")
async def root():
    return {"message": "News Classification API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy"}


if __name__ == "__main__":
    logger.info("Starting uvicorn server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)
