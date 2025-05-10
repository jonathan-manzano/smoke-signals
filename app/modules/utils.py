from google.cloud import storage
import numpy as np
import tempfile
import os

def get_npy_from_gcs(bucket_name, source_blob_name):
    """
    Download a .npy file from Google Cloud Storage and return its content as a NumPy array.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    if not blob.exists():
        raise FileNotFoundError(f"Blob {source_blob_name} not found in bucket {bucket_name}")

    # Create a temporary file to store the downloaded .npy file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name
        blob.download_to_filename(temp_file_path)

    try:
        # Load the .npy file into a NumPy array
        numpy_data = np.load(temp_file_path, allow_pickle=True)
        return numpy_data
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)