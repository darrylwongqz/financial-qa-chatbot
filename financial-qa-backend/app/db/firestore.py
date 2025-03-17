# app/db/firestore.py
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1 import SERVER_TIMESTAMP

# Import our centralized logger
from app.utils.logging_utils import logger

# Global variable for the Firestore client
db = None
_is_initialized = False

def initialize_firestore():
    """
    Initialize the Firestore client using a singleton pattern.
    This function can be called multiple times but will only initialize once.
    """
    global db, _is_initialized
    
    # If already initialized, don't initialize again
    if _is_initialized:
        logger.info("Firestore already initialized, skipping")
        return
    
    try:
        # Try to get the service account JSON from environment variable
        service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        
        if service_account_json:
            # Create a temporary file to store the JSON content
            import json
            import tempfile
            
            # Create a temporary file
            fd, temp_path = tempfile.mkstemp()
            with os.fdopen(fd, 'w') as tmp:
                tmp.write(service_account_json)
            
            # Set the environment variable to point to the temporary file
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
            logger.info(f"Using service account JSON from environment variable")
            
            # Initialize Firestore client
            db = AsyncClient()
            logger.info("Successfully initialized asynchronous Firestore client from environment variable")
            
            # Clean up the temporary file
            os.unlink(temp_path)
        else:
            # Fall back to file-based approach for local development
            service_key_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "service_key.json")
            
            if not os.path.exists(service_key_path):
                raise FileNotFoundError(f"File {service_key_path} was not found.")
                
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_key_path
            logger.info(f"Using service key at: {service_key_path}")
            
            db = AsyncClient()
            logger.info("Successfully initialized asynchronous Firestore client from file")
    except Exception as e:
        logger.error(f"Failed to initialize Firestore client: {str(e)}")
        db = None
    
    # Mark as initialized
    _is_initialized = True

# Don't initialize automatically on import
# initialize_firestore()

async def add_document(
    collection: str,
    data: Dict[str, Any],
    document_id: Optional[str] = None
) -> Optional[str]:
    """
    Add a document to a Firestore collection asynchronously.
    """
    if db is None:
        logger.error("Firestore client not initialized")
        return None
    try:
        data["timestamp"] = SERVER_TIMESTAMP
        if document_id:
            await db.collection(collection).document(document_id).set(data)
            return document_id
        else:
            doc_ref = await db.collection(collection).add(data)
            return doc_ref[1].id
    except Exception as e:
        logger.error(f"Error adding document to {collection}: {str(e)}")
        return None

async def get_document(
    collection: str,
    document_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a document from a Firestore collection asynchronously.
    """
    if db is None:
        logger.error("Firestore client not initialized")
        return None
    try:
        doc_ref = db.collection(collection).document(document_id)
        doc = await doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            logger.warning(f"Document {document_id} not found in {collection}")
            return None
    except Exception as e:
        logger.error(f"Error getting document {document_id} from {collection}: {str(e)}")
        return None

async def get_documents(
    collection: str,
    order_by: str = "timestamp",
    limit: int = 100,
    descending: bool = False
) -> List[Dict[str, Any]]:
    """
    Retrieve all documents from a Firestore collection asynchronously.
    
    Args:
        collection: The collection to get documents from
        order_by: Field to order results by (default: timestamp)
        limit: Maximum number of documents to return
        descending: Whether to order in descending order (newest first)
        
    Returns:
        List of documents as dictionaries
    """
    if db is None:
        logger.error("Firestore client not initialized")
        return []
    
    try:
        logger.info(f"Retrieving documents from collection: {collection}, order_by: {order_by}, limit: {limit}, descending: {descending}")
        
        # Specify direction based on the descending flag
        direction = "DESCENDING" if descending else "ASCENDING"
        query_ref = db.collection(collection).order_by(order_by, direction=direction).limit(limit)
        results = []
        
        # Stream the results
        async for doc in query_ref.stream():
            doc_dict = doc.to_dict()
            doc_dict["id"] = doc.id
            
            # Convert timestamp to ISO format string if it exists
            if "timestamp" in doc_dict and not isinstance(doc_dict["timestamp"], str):
                doc_dict["timestamp"] = doc_dict["timestamp"].isoformat() if hasattr(doc_dict["timestamp"], "isoformat") else str(doc_dict["timestamp"])
            
            results.append(doc_dict)
        
        logger.info(f"Retrieved {len(results)} documents from collection: {collection}")
        
        # Log document IDs and conversation IDs for debugging
        if results:
            doc_ids = [doc.get("id") for doc in results]
            logger.debug(f"Document IDs: {doc_ids[:5]}{'...' if len(doc_ids) > 5 else ''}")
            
            if all("conversation_id" in doc for doc in results):
                conversation_ids = set(doc.get("conversation_id") for doc in results if doc.get("conversation_id"))
                logger.debug(f"Conversation IDs in results: {conversation_ids}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error getting documents from {collection}: {str(e)}")
        return []

async def delete_documents(
    collection: str,
    batch_size: int = 100
) -> bool:
    """
    Delete all documents in a Firestore collection asynchronously.
    
    Args:
        collection: The collection to delete documents from
        batch_size: Number of documents to delete in each batch
        
    Returns:
        True if successful, False otherwise
    """
    if db is None:
        logger.error("Firestore client not initialized")
        return False
    
    try:
        # Get a batch of documents
        query_ref = db.collection(collection).limit(batch_size)
        docs = []
        
        async for doc in query_ref.stream():
            docs.append(doc)
        
        # If no documents, we're done
        if not docs:
            return True
        
        # Delete documents in batch
        for doc in docs:
            await doc.reference.delete()
        
        # Recursively delete remaining documents
        if len(docs) >= batch_size:
            return await delete_documents(collection, batch_size)
        
        return True
    
    except Exception as e:
        logger.error(f"Error deleting documents from {collection}: {str(e)}")
        return False

async def query_documents(
    collection: str,
    field: str,
    operator: str,
    value: Any,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Query documents from a Firestore collection asynchronously.
    """
    if db is None:
        logger.error("Firestore client not initialized")
        return []
    try:
        # Firestore supports operators directly (e.g., "==", ">", etc.)
        query_ref = db.collection(collection).where(field, operator, value).limit(limit)
        results = []
        async for doc in query_ref.stream():
            doc_dict = doc.to_dict()
            doc_dict["id"] = doc.id
            results.append(doc_dict)
        return results
    except Exception as e:
        logger.error(f"Error querying documents from {collection}: {str(e)}")
        return []

async def update_document(
    collection: str,
    document_id: str,
    data: Dict[str, Any]
) -> bool:
    """
    Update a document in a Firestore collection asynchronously.
    """
    if db is None:
        logger.error("Firestore client not initialized")
        return False
    try:
        data["updated_at"] = SERVER_TIMESTAMP
        await db.collection(collection).document(document_id).update(data)
        return True
    except Exception as e:
        logger.error(f"Error updating document {document_id} in {collection}: {str(e)}")
        return False

async def delete_document(
    collection: str,
    document_id: str
) -> bool:
    """
    Delete a specific document from a Firestore collection asynchronously.
    
    Args:
        collection: The collection containing the document
        document_id: The ID of the document to delete
        
    Returns:
        True if successful, False otherwise
    """
    if db is None:
        logger.error("Firestore client not initialized")
        return False
    
    try:
        await db.collection(collection).document(document_id).delete()
        return True
    except Exception as e:
        logger.error(f"Error deleting document {document_id} from {collection}: {str(e)}")
        return False