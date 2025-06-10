import streamlit as st
import requests
import base64
from PIL import Image
import io
import json
import os
import time
import pandas as pd
from typing import List, Dict, Any, Optional

# Configure page FIRST before any other Streamlit commands
st.set_page_config(page_title="📷 Image RAG Demo - Colpali + Llama Vision", layout="wide")

# Configure Streamlit to listen on all network interfaces
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
os.environ['STREAMLIT_SERVER_PORT'] = '8501'

# --- Configuration ---
# Try RunPod URL first, fallback to localhost for development
LOCAL_URL = "http://127.0.0.1:8000"
BACKEND_URL =  "http://127.0.0.1:8001"

# Test connection to determine which URL to use
def test_connection(url):
    try:
        response = requests.get(f"{url}/", timeout=5)
        return response.status_code == 200
    except:
        return False


# API Endpoints
IMAGE_INDEX_URL = f"{LOCAL_URL}/api/index/image"
PDF_INDEX_URL = f"{LOCAL_URL}/api/index/pdf"
QUERY_URL = f"{LOCAL_URL}/api/query"
COLLECTIONS_URL = f"{BACKEND_URL}/api/v1/collections"

# --- Helper Functions ---
def get_collections() -> List[Dict[str, Any]]:
    """Get list of collections from the API."""
    try:
        response = requests.get(COLLECTIONS_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching collections: {str(e)}")
        return []

def create_collection(name: str) -> bool:
    """Create a new collection."""
    try:
        response = requests.post(
            COLLECTIONS_URL,
            json={"name": name},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error creating collection: {str(e)}")
        return False

def delete_collection(name: str) -> bool:
    """Delete a collection."""
    try:
        response = requests.delete(f"{COLLECTIONS_URL}/{name}", timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error deleting collection: {str(e)}")
        return False

def reset_session():
    """Reset the session state."""
    st.session_state.messages = []
    st.session_state.uploaded_file_details = []
    st.session_state.processing_message = None

def display_chat_message(role: str, content: Any):
    """Helper to display a chat message with a consistent avatar."""
    avatar_map = {"user": "👤", "assistant": "🤖"}
    avatar = avatar_map.get(role)
    with st.chat_message(role, avatar=avatar):
        if isinstance(content, str):
            st.markdown(content)
        elif isinstance(content, dict):
            # Display image if present
            if "image" in content:
                st.subheader("Most Similar Image:")
                st.write(f"Similarity Score: {content.get('similarity_score', 0):.4f}")
                
                # Decode and display image
                try:
                    img_data = base64.b64decode(content['image'])
                    image = Image.open(io.BytesIO(img_data))
                    st.image(image, caption="Retrieved Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying image: {e}")
            
            # Display AI response
            if "response" in content:
                st.subheader("AI Response:")
                st.markdown(content['response'])
            
            # Display any additional metadata
            if "image_hash" in content:
                with st.expander("Technical Details", expanded=False):
                    st.code(f"Image Hash: {content['image_hash']}")
        else:
            st.markdown(str(content))

# --- Streamlit App ---

st.title("📷 Image RAG Demo")
st.caption("Advanced Image Retrieval with Colpali + Llama Vision | Upload images/PDFs and query them with natural language")

# --- Initialize session state ---
default_session_state = {
    "messages": [],
    "uploaded_file_details": [],
    "selected_collection": None,
    "processing_message": None
}
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# --- Sidebar for Collection and Document Management ---
with st.sidebar:
    st.header("📚 Collection & Image Management")
    
    # Collection Management Section
    st.subheader("Collection Management")
    
    # Get existing collections
    collections = get_collections()
    collection_names = [col["name"] for col in collections] if collections else []
    
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
        st.session_state.selected_collection = st.selectbox(
            "Select Collection",
            options=["default"] + collection_names,
            index=0
        )
        
        # Display collection info
        if st.session_state.selected_collection != "default":
            selected_col_info = next((col for col in collections if col["name"] == st.session_state.selected_collection), None)
            if selected_col_info:
                st.info(f"📊 Documents: {selected_col_info.get('document_count', 0)}")
        
        # Delete collection option (only for non-default collections)
        if st.session_state.selected_collection != "default":
            with st.expander("Delete Collection", expanded=False):
                st.warning("⚠️ This action cannot be undone!")
                confirm_delete = st.checkbox("I confirm I want to delete this collection")
                if st.button("Delete Collection", disabled=not confirm_delete, use_container_width=True):
                    if delete_collection(st.session_state.selected_collection):
                        st.success(f"Collection '{st.session_state.selected_collection}' deleted successfully!")
                        st.rerun()
    else:
        st.info("No collections available. Using default collection.")
        st.session_state.selected_collection = "default"
    
    st.markdown("---")
    
    # Image Upload Section
    st.subheader("Image Upload")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Images or PDFs",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "gif"],
        accept_multiple_files=True,
        help="Upload images or PDF files to add to your collection"
    )
    
    if uploaded_files:
        if st.button("Process and Index", use_container_width=True):
            # Create containers for progress tracking
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
            with status_container:
                processing_message = st.info("🔄 Starting file processing...")
            
            try:
                processed_files = []
                total_files = len(uploaded_files)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    # Update progress
                    current_progress = idx / total_files
                    progress_bar.progress(current_progress)
                    progress_text.text(f"Processing file {idx + 1} of {total_files}: {uploaded_file.name}")
                    
                    with status_container:
                        processing_message.info(f"🔄 Processing: {uploaded_file.name} ({uploaded_file.type})")
                    
                    try:
                        # Add timing
                        start_time = time.time()
                        
                        # Determine file type and endpoint
                        if uploaded_file.type == 'application/pdf':
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                            with status_container:
                                processing_message.info(f"📄 Converting PDF to images and generating embeddings...")
                            response = requests.post(PDF_INDEX_URL, files=files, timeout=300)  # Increased to 5 minutes
                        else:
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                            with status_container:
                                processing_message.info(f"🖼️ Generating image embeddings with ColQwen2...")
                            response = requests.post(IMAGE_INDEX_URL, files=files, timeout=300)  # Increased to 5 minutes
                        
                        processing_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            result = response.json()
                            processed_files.append({
                                "name": uploaded_file.name,
                                "status": "✅ Success",
                                "message": result.get("message", "Processed successfully"),
                                "hash": result.get("image_hash") or result.get("image_hashes", ["N/A"])[0] if result.get("image_hashes") else "N/A",
                                "time": f"{processing_time:.1f}s"
                            })
                            with status_container:
                                processing_message.success(f"✅ {uploaded_file.name} processed successfully in {processing_time:.1f}s")
                        else:
                            error_msg = response.text
                            if response.status_code == 500:
                                error_msg = f"Server error: {error_msg}"
                            elif response.status_code == 408:
                                error_msg = "Request timeout - file processing took too long"
                            
                            processed_files.append({
                                "name": uploaded_file.name,
                                "status": "❌ Error",
                                "message": error_msg,
                                "hash": "N/A",
                                "time": f"{processing_time:.1f}s"
                            })
                            with status_container:
                                processing_message.error(f"❌ {uploaded_file.name} failed: {error_msg}")
                                
                    except requests.exceptions.Timeout:
                        processed_files.append({
                            "name": uploaded_file.name,
                            "status": "⏰ Timeout",
                            "message": "Processing timed out after 5 minutes. The file may be too large or the model is overloaded.",
                            "hash": "N/A",
                            "time": "300s+"
                        })
                        with status_container:
                            processing_message.error(f"⏰ {uploaded_file.name} timed out after 5 minutes")
                            
                    except requests.exceptions.ConnectionError:
                        processed_files.append({
                            "name": uploaded_file.name,
                            "status": "🔌 Connection Error",
                            "message": "Could not connect to the backend server. Please check if the server is running.",
                            "hash": "N/A",
                            "time": "N/A"
                        })
                        with status_container:
                            processing_message.error(f"🔌 {uploaded_file.name} - connection error")
                            
                    except Exception as e:
                        processed_files.append({
                            "name": uploaded_file.name,
                            "status": "💥 Unexpected Error",
                            "message": str(e),
                            "hash": "N/A",
                            "time": "N/A"
                        })
                        with status_container:
                            processing_message.error(f"💥 {uploaded_file.name} - unexpected error: {str(e)}")
                
                # Complete progress
                progress_bar.progress(1.0)
                progress_text.text(f"✅ Completed processing {total_files} files")
                
                # Final status
                success_count = sum(1 for f in processed_files if "Success" in f["status"])
                with status_container:
                    if success_count == total_files:
                        processing_message.success(f"🎉 All {total_files} files processed successfully!")
                    else:
                        processing_message.warning(f"⚠️ Processed {success_count}/{total_files} files successfully")
                
                # Display results table
                st.subheader("📊 Processing Results")
                results_df = pd.DataFrame(processed_files)
                st.dataframe(results_df, use_container_width=True)
                
                # Update session state
                st.session_state.uploaded_file_details.extend(processed_files)
                
                # Show success message
                st.success(f"Processed {len(processed_files)} files!")
                
                # Reset chat history
                reset_session()
                
            except Exception as e:
                processing_message.empty()
                st.error(f"Error processing files: {str(e)}")

    st.markdown("---")
    st.subheader("Processed Files Log")
    if st.session_state.uploaded_file_details:
        for detail in st.session_state.uploaded_file_details:
            with st.expander(f"{detail['status']} {detail['name']}", expanded=False):
                st.write(f"**Status:** {detail['status']}")
                st.write(f"**Message:** {detail['message']}")
                st.write(f"**Hash:** `{detail['hash']}`")
    else:
        st.info("No files processed in this session yet.")
    
    if st.button("Clear Session Data", use_container_width=True):
        reset_session()
        st.rerun()

# --- Main Chat Interface ---
st.header("💬 Query Your Images")

# Display processing message if active
if st.session_state.processing_message:
    st.info(st.session_state.processing_message)

# Display chat history
for message in st.session_state.messages:
    display_chat_message(message["role"], message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your images..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    display_chat_message("user", prompt)

    # Process query
    with st.spinner("Searching through images..."):
        try:
            # Query the image index
            data = {"query": prompt}
            response = requests.post(QUERY_URL, data=data, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": result})
                display_chat_message("assistant", result)
                
            elif response.status_code == 404:
                error_msg = "No images found in the index. Please upload some images first."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                display_chat_message("assistant", error_msg)
            else:
                error_msg = f"Error: {response.text}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                display_chat_message("assistant", error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "The request timed out. The server might be busy. Please try again."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            display_chat_message("assistant", error_msg)
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            display_chat_message("assistant", error_msg)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        <p>📷 Image RAG Demo | Powered by Colpali + Llama Vision</p>
        <p>Upload images or PDFs and query them using natural language</p>
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    pass