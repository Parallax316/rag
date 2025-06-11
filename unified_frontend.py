import streamlit as st
import requests
import os

# --- Configuration ---
BACKEND_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")
COLLECTIONS_URL = f"{BACKEND_URL}/collections"
UPLOAD_URL = f"{BACKEND_URL}/upload/document"

st.set_page_config(page_title="Unified RAG Uploader", layout="wide")
st.title("Unified RAG System: Text + VLM")

# --- Collection Management ---
def get_collections():
    try:
        resp = requests.get(COLLECTIONS_URL)
        resp.raise_for_status()
        return resp.json().get("collections", [])
    except Exception as e:
        st.error(f"Error fetching collections: {e}")
        return []

def create_collection(name):
    try:
        resp = requests.post(COLLECTIONS_URL, json={"name": name})
        resp.raise_for_status()
        st.success(f"Collection '{name}' created!")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error creating collection: {e}")

# Sidebar: Collection selection/creation
with st.sidebar:
    st.header("Collection Management")
    collections = get_collections()
    if collections:
        selected_collection = st.selectbox("Select Collection", collections)
    else:
        selected_collection = None
    new_coll = st.text_input("New Collection Name")
    if st.button("Create Collection") and new_coll:
        create_collection(new_coll)

# --- File Upload ---
st.header("Upload and Process Document (Text + VLM)")
uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif"])

if uploaded_file and selected_collection:
    if st.button("Process Document"):
        with st.spinner("Uploading and triggering processing..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {"collection": selected_collection}
            try:
                resp = requests.post(UPLOAD_URL, files=files, data=data)
                resp.raise_for_status()
                st.success("Document uploaded and both RAG systems are processing it!")
                st.json(resp.json())
            except Exception as e:
                st.error(f"Error uploading/processing: {e}")
else:
    st.info("Please select a collection and upload a file.")
