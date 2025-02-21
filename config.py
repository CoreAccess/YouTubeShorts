import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev'
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024 * 1024  # 8GB max file size
    # Increase timeouts for large file uploads
    UPLOAD_TIMEOUT = 3600  # 1 hour timeout for uploads
    # Add buffer size config
    MAX_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB buffer size for file operations
    UPLOAD_FOLDER = 'uploads'
