# backend/api.py
"""FastAPI application for speech analysis with comprehensive input validation."""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import os
import logging
from typing import Dict, Any

from validators.file_validator import validate_audio_file
from record_audio import record_audio
from link import run_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Speech Personality Analysis API",
    description="AI-powered speech analysis with personality insights",
    version="1.0.0"
)

# Configure CORS.
# Example: ALLOWED_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
# In local development, allow the Vite dev server origins by default.
raw_origins = os.getenv("ALLOWED_ORIGINS", "")
if raw_origins:
    # Split, trim, and filter blank entries
    ALLOWED_ORIGINS = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
elif os.getenv("ENV") == "development" or os.getenv("DEBUG") == "true":
    # Only allow wildcard in explicit development mode
    ALLOWED_ORIGINS = ["*"]
    logger.warning("CORS configured with wildcard '*' in development mode")
elif os.getenv("ENV") == "production":
    # Fail closed in production unless ALLOWED_ORIGINS is explicitly set
    ALLOWED_ORIGINS = []
    logger.info("CORS configured with no allowed origins (production default)")
else:
    ALLOWED_ORIGINS = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ]
    logger.info("CORS configured for local frontend development")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "speech-analysis-api"}


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze uploaded audio file for speech patterns and personality insights.
    
    Args:
        file: Audio file upload (WAV, MP3, WebM, OGG)
        
    Returns:
        Dictionary containing analysis results including transcript,
        speech metrics, agent analyses, and final report
        
    Raises:
        HTTPException: If file validation fails or processing errors occur
    """
    # Validate the uploaded file
    await validate_audio_file(file)

    # Initialize temp file paths before try/except
    audio_path = None
    webm_path = None
    try:
        # Record/save the audio file - returns (wav_path, webm_path)
        audio_path, webm_path = await run_in_threadpool(record_audio, file)
        logger.info(f"Processing audio file: {file.filename}")

        # Run the analysis pipeline with the WAV path
        result = await run_in_threadpool(run_pipeline, audio_path)

        logger.info(f"Analysis completed successfully for: {file.filename}")
        return result

    except HTTPException:
        # Re-raise HTTP exceptions (from validation)
        raise
    except Exception as e:
        logger.exception(f"Error processing audio file {file.filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing audio file"
        )
    finally:
        # Cleanup: Remove all temporary audio files (both WAV and WebM)
        for temp_file in [audio_path, webm_path]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup file {temp_file}: {e}")

