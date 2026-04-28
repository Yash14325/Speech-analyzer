"""File validation utilities for API input security."""

from fastapi import UploadFile, HTTPException
from typing import Set
import os
import logging
from urllib.parse import unquote

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

logger = logging.getLogger(__name__)

# Allowed MIME types for audio files
ALLOWED_AUDIO_TYPES: Set[str] = {
    "audio/wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/webm",
    "audio/ogg",
    "audio/x-wav",
    "audio/wave",
    "video/webm",  # libmagic reports WebM (including audio-only) as video/webm
}

# Maximum file size (50MB)
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Minimum file size (1KB - to prevent empty files)
MIN_FILE_SIZE_BYTES = 1024


async def validate_audio_file(file: UploadFile) -> None:
    """
    Validate uploaded audio file for security and format compliance.
    
    Performs multiple validation checks:
    - Content type validation
    - File size validation
    - Filename sanitization
    - Empty file detection
    
    Args:
        file: The uploaded file from FastAPI
        
    Raises:
        HTTPException: If validation fails with appropriate status code and message
    """
    # Check if file exists
    if not file:
        logger.warning("File upload attempted with no file")
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )
    
    # Check client-provided content type first (fast check)
    if file.content_type not in ALLOWED_AUDIO_TYPES:
        logger.warning(f"Invalid file type uploaded (client header): {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed types: {', '.join(ALLOWED_AUDIO_TYPES)}"
        )

    # Perform server-side content type validation using magic bytes
    if MAGIC_AVAILABLE:
        try:
            # Read first 512 bytes for magic number detection
            file.file.seek(0)
            file_prefix = file.file.read(512)
            file.file.seek(0)  # Reset to beginning

            # Detect actual MIME type from bytes
            detected_mime = magic.from_buffer(file_prefix, mime=True)

            # Validate detected type against allowed types
            if detected_mime not in ALLOWED_AUDIO_TYPES:
                logger.warning(
                    f"Content type mismatch - client: {file.content_type}, "
                    f"detected: {detected_mime}, filename: {file.filename}"
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"File content does not match declared type. Detected: {detected_mime}"
                )

            logger.debug(f"Server-side validation passed: {detected_mime}")

        except HTTPException:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Error during magic byte validation: {e}")
            raise HTTPException(
                status_code=415,
                detail="Magic-byte validation failed"
            )
    else:
        logger.warning(
            "python-magic not available - relying on client-provided content type only. "
            "Install python-magic for enhanced security."
        )
    
    # Check filename exists
    if not file.filename:
        logger.warning("File upload attempted with no filename")
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )
    
    # Sanitize filename (prevent path traversal attacks)
    # Decode URL-encoded characters and check for path traversal
    decoded_filename = unquote(file.filename)
    if ".." in decoded_filename or "/" in decoded_filename or "\\" in decoded_filename:
        logger.warning(f"Potential path traversal attempt: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail="Invalid filename - path traversal detected"
        )
    
    # Check file size
    try:
        # Read file to check size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size == 0:
            logger.warning(f"Empty file uploaded: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Empty file - file size is 0 bytes"
            )
        
        if file_size < MIN_FILE_SIZE_BYTES:
            logger.warning(f"File too small: {file.filename} ({file_size} bytes)")
            raise HTTPException(
                status_code=400,
                detail=f"File too small. Minimum size: {MIN_FILE_SIZE_BYTES / 1024}KB"
            )
        
        if file_size > MAX_FILE_SIZE_BYTES:
            logger.warning(f"File too large: {file.filename} ({file_size} bytes)")
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
            )
        
        logger.info(f"File validation passed: {file.filename} ({file_size} bytes)")
        
    except HTTPException:
        # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Error checking file size: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error validating file"
        )


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent security issues.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for file system operations
    """
    # Get just the basename (no path components)
    filename = os.path.basename(filename)
    
    # Remove or replace dangerous characters
    dangerous_chars = ['..', '/', '\\', '\0', '<', '>', ':', '"', '|', '?', '*']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Limit filename length
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext
    
    return filename
