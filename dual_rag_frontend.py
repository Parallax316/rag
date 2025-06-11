import streamlit as st
import requests
import os
from typing import List, Dict, Any, Optional
import time
import json

# --- Configuration ---
BACKEND_URL = os.environ.get("BACKEND_API_URL", "  http://127.0.0.1:8001")
DOCUMENTS_UPLOAD_URL = f"{BACKEND_URL}/api/v1/docs/upload-multiple"
DOCUMENTS_UPLOAD_VLM_URL = f"{BACKEND_URL}/api/v1/docs/upload-multiple-vlm"
COLLECTIONS_URL = f"{BACKEND_URL}/api/v1/collections"

VLM_LOCAL_URL = os.environ.get("VLM_LOCAL_URL", "https://99ad-213-173-108-218.ngrok-free.app")
IMAGE_INDEX_URL = f"{VLM_LOCAL_URL}/api/index/image"
PDF_INDEX_URL = f"{VLM_LOCAL_URL}/api/index/pdf"

st.set_page_config(page_title="Dual RAG Uploader", layout="wide")
st.title("Dual RAG System: Text + VLM (Separate APIs)")

# --- Helper Functions ---
def get_collections() -> List[Dict[str, Any]]:
    try:
        response = requests.get(COLLECTIONS_URL)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching collections: {str(e)}")
        return []

def create_collection(name: str) -> bool:
    try:
        response = requests.post(
            COLLECTIONS_URL,
            json={"name": name},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error creating collection: {str(e)}")
        return False

def delete_collection(name: str) -> bool:
    try:
        response = requests.delete(f"{COLLECTIONS_URL}/{name}")
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error deleting collection: {str(e)}")
        return False

# --- Sidebar for Collection and Document Management ---
with st.sidebar:
    st.header("📚 Collection & Document Management")
    collections = get_collections()
    collection_names = [col["name"] for col in collections]
    # Create new collection
    with st.expander("Create New Collection", expanded=False):
        new_collection_name = st.text_input("Collection Name")
        if st.button("Create Collection", use_container_width=True):
            if new_collection_name:
                if create_collection(new_collection_name):
                    st.success(f"Collection '{new_collection_name}' created successfully!")
                    st.rerun()
            else:
                st.warning("Please enter a collection name")
    # Select collection
    if collection_names:
        selected_collection = st.selectbox(
            "Select Collection",
            options=collection_names,
            index=0
        )
        # Delete collection option
        with st.expander("Delete Collection", expanded=False):
            st.warning("⚠️ This action cannot be undone!")
            confirm_delete = st.checkbox("I confirm I want to delete this collection")
            if st.button("Delete Collection", disabled=not confirm_delete, use_container_width=True):
                if delete_collection(selected_collection):
                    st.success(f"Collection '{selected_collection}' deleted successfully!")
                    st.rerun()
    else:
        selected_collection = None
        st.info("No collections available. Create a new collection to get started.")

# --- Document Upload Section ---
st.header("Upload and Process Document (Text + VLM)")
uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif"],
    accept_multiple_files=True
)

if uploaded_files and selected_collection:
    if st.button("Process Documents", use_container_width=True):
        processing_message = st.info("Processing documents... This may take a while.")
        for uploaded_file in uploaded_files:
            st.write(f"Processing: {uploaded_file.name}")
            # Prepare files for text RAG (as in ui.py)
            files = [("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))]
            # Upload to text RAG
            try:
                response_text = requests.post(
                    DOCUMENTS_UPLOAD_URL,
                    files=files,
                    params={"collection_name": selected_collection}
                )
                response_text.raise_for_status()
                st.success(f"Text RAG: {uploaded_file.name} uploaded and processing started!")
            except Exception as e:
                st.error(f"Text RAG error for {uploaded_file.name}: {e}")
            # Upload to VLM RAG
            try:
                if uploaded_file.type == 'application/pdf':
                    vlm_url = PDF_INDEX_URL
                else:
                    vlm_url = IMAGE_INDEX_URL
                response_vlm = requests.post(
                    vlm_url,
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={"collection": selected_collection}
                )
                response_vlm.raise_for_status()
                st.success(f"VLM RAG: {uploaded_file.name} uploaded and processing started!")
            except Exception as e:
                st.error(f"VLM RAG error for {uploaded_file.name}: {e}")
        processing_message.empty()
else:
    st.info("Please select a collection and upload a file.")
