import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Multimodal RAG System")

# --- Collection management ---
st.sidebar.header("Collections")

# Fetch collections from backend
@st.cache_data(show_spinner=False)
def get_collections():
    try:
        resp = requests.get(f"{API_URL}/collections")
        resp.raise_for_status()
        data = resp.json()
        # Accept both {"collections": [..]} and {"collections": [{"name": ...}]}
        collections = data.get("collections", [])
        if collections and isinstance(collections[0], dict) and "name" in collections[0]:
            return [col["name"] for col in collections]
        return collections
    except Exception as e:
        st.sidebar.error(f"Error fetching collections: {e}")
        return []

collections = get_collections()

# Dropdown for selecting collection
if collections:
    if "selected_collection" not in st.session_state:
        st.session_state.selected_collection = collections[0]
    selected_collection = st.sidebar.selectbox(
        "Select Collection",
        options=collections,
        index=collections.index(st.session_state.get("selected_collection", collections[0]))
    )
    st.session_state.selected_collection = selected_collection
else:
    st.sidebar.info("No collections available. Create one below.")
    st.session_state.selected_collection = "default"

# Create new collection
new_collection = st.sidebar.text_input("New Collection Name")
if st.sidebar.button("Create Collection"):
    if new_collection:
        resp = requests.post(f"{API_URL}/collections", json={"name": new_collection})
        if resp.status_code == 200:
            st.sidebar.success(f"Collection '{new_collection}' created.")
            st.cache_data.clear()
        else:
            st.sidebar.error(f"Failed to create collection: {resp.text}")
    else:
        st.sidebar.warning("Enter a collection name.")

# Delete selected collection
if st.sidebar.button("Delete Collection"):
    resp = requests.delete(f"{API_URL}/collections/{st.session_state.selected_collection}")
    if resp.status_code == 200:
        st.sidebar.success(f"Collection '{st.session_state.selected_collection}' deleted.")
        st.cache_data.clear()
    else:
        st.sidebar.error(f"Failed to delete collection: {resp.text}")

# --- Upload ---
st.header("Upload Data")
uploaded_file = st.file_uploader("Upload image, PDF, or text file")
upload_type = st.selectbox("Type", ["image", "pdf", "text"])
if st.button("Upload"):
    if uploaded_file:
        files = {"file": uploaded_file}
        resp = requests.post(
            f"{API_URL}/upload/{upload_type}",
            files=files,
            data={"collection": st.session_state.selected_collection}
        )
        if resp.status_code == 200:
            st.success("File uploaded and indexed.")
        else:
            st.error(f"Upload failed: {resp.text}")
    else:
        st.warning("Please upload a file.")

# --- Query ---
st.header("Query")
query = st.text_input("Enter your query")
if st.button("Search"):
    resp = requests.post(
        f"{API_URL}/query",
        json={"query": query, "collection": st.session_state.selected_collection}
    )
    if resp.status_code == 200:
        results = resp.json()
        st.write("Text Results:", results.get("text"))
        st.write("Image Results:", results.get("image"))
    else:
        st.error(f"Query failed: {resp.text}")
