# filepath: c:\Users\Aaron Sam\Documents\CS163\smoke-signals\src\app\utils.py
import pandas as pd
from google.cloud import storage
from io import StringIO

def get_csv_from_gcs(bucket_name, source_blob_name):
    """
    Download a CSV file from Google Cloud Storage and return a DataFrame.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_text()
    return pd.read_csv(StringIO(data))