import streamlit as st
import requests
import base64
from PIL import Image
import io
import json

# API endpoint
API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Image RAG Demo", page_icon="📷", layout="wide")

def main():
    st.title("📷 Image RAG Demo")
    st.subheader("Colpali + Llama Vision")
    
    # Use st.radio for tab selection
    tab = st.radio("Navigation", ["➕ Add to Index", "🔍 Query Index"])

    if tab == "➕ Add to Index":
        st.header("Add Images to Index")
        
        # File uploader
        uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'pdf'])
        
        if uploaded_files and st.button("Process and Index"):
            with st.spinner("Processing files..."):
                for uploaded_file in uploaded_files:
                    # Determine file type
                    if uploaded_file.type == 'application/pdf':
                        st.info(f"Processing PDF: {uploaded_file.name}")
                        # Call PDF indexing endpoint
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        response = requests.post(f"{API_URL}/index/pdf", files=files)
                    else:
                        st.info(f"Processing image: {uploaded_file.name}")
                        # Call image indexing endpoint
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{API_URL}/index/image", files=files)
                    
                    # Check response
                    if response.status_code == 200:
                        result = response.json()
                        st.success(result["message"])
                    else:
                        st.error(f"Error: {response.text}")
            
            st.success("All files processed successfully!")

    elif tab == "🔍 Query Index":
        st.header("Query Index")
        
        query = st.text_input("Enter your query")
        
        if query and st.button("Search"):
            with st.spinner("Searching..."):
                # Call query endpoint
                data = {"query": query}
                response = requests.post(f"{API_URL}/query", data=data)
                
                # Check response
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display image
                    st.subheader("Most similar image:")
                    st.write(f"Similarity Score: {result['similarity_score']:.4f}")
                    
                    # Decode and display image
                    img_data = base64.b64decode(result['image'])
                    image = Image.open(io.BytesIO(img_data))
                    st.image(image)
                    
                    # Display AI response
                    st.subheader("AI Response:")
                    st.write(result['response'])
                elif response.status_code == 404:
                    st.warning("No images found in the index. Please add images first.")
                else:
                    st.error(f"Error: {response.text}")

if __name__ == "__main__":
    main()