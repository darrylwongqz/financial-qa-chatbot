# app/db/firebase_init.py
import os
from pathlib import Path
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials
# Load environment variables
load_dotenv()
from app.utils.logging_utils import logger

# Track initialization state
_is_initialized = False

def initialize_firebase():
    """
    Initialize Firebase Admin SDK using a singleton pattern.
    This function can be called multiple times but will only initialize once.
    """
    global _is_initialized
    
    # If already initialized, don't initialize again
    if _is_initialized:
        logger.info("Firebase already initialized, skipping")
        return
    
    # Try to get the service account JSON from environment variable
    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    
    if service_account_json:
        try:
            import json
            import tempfile
            
            # Parse the JSON string
            service_account_dict = json.loads(service_account_json)
            
            # Initialize Firebase Admin using the service account credentials directly
            logger.info("Firebase Admin SDK initializing from environment variable...")
            cred = credentials.Certificate(service_account_dict)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Firebase Admin SDK from environment variable: {str(e)}")
    else:
        # Fall back to file-based approach for local development
        service_key_path_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if service_key_path_env:
            service_key_path = Path(service_key_path_env)
            logger.info(f"Using service key path from environment variable: {service_key_path}")
        else:
            # Determine the project root directory (two levels up from this file)
            PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
            # Build the full path to the service key file in the project root
            service_key_path = PROJECT_ROOT / "app" / "service_key.json"

        # Check if the service key file exists
        if not service_key_path.exists():
            # For development, we'll log a warning but continue without Firebase
            logger.warning(f"Service account key not found at {service_key_path}")
            logger.warning("Firebase functionality will be disabled. This is OK for development if you don't need Firestore.")
        else:
            try:
                # Initialize Firebase Admin using the service account credentials
                logger.info("Firebase Admin SDK initializing from file...")
                cred = credentials.Certificate(str(service_key_path))
                firebase_admin.initialize_app(cred)
                logger.info("Firebase Admin SDK initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Firebase Admin SDK from file: {str(e)}")
    
    # Mark as initialized
    _is_initialized = True
