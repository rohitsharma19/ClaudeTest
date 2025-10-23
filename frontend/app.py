"""
Streamlit Frontend for RAG Application.
"""

import streamlit as st
import requests
import json
from typing import List, Dict, Any
import os

# Backend API URL
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:5000')


def init_session_state():
    """Initialize session state variables."""
    if 'session_token' not in st.session_state:
        st.session_state.session_token = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""


def login(username: str, password: str) -> bool:
    """
    Login user.

    Args:
        username: Username
        password: Password

    Returns:
        True if login successful, False otherwise
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/auth/login",
            json={"username": username, "password": password},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            st.session_state.session_token = data['session_token']
            st.session_state.username = data['username']
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return False


def logout():
    """Logout user."""
    try:
        if st.session_state.session_token:
            requests.post(
                f"{API_BASE_URL}/api/auth/logout",
                json={"session_token": st.session_state.session_token},
                timeout=10
            )

        st.session_state.session_token = None
        st.session_state.username = None
        st.session_state.chat_history = []
    except Exception as e:
        st.error(f"Logout failed: {str(e)}")


def upload_documents(files: List) -> Dict[str, Any]:
    """
    Upload documents to backend.

    Args:
        files: List of uploaded files

    Returns:
        Response data from backend
    """
    try:
        files_data = []
        for uploaded_file in files:
            files_data.append(
                ('files', (uploaded_file.name, uploaded_file, uploaded_file.type))
            )

        response = requests.post(
            f"{API_BASE_URL}/api/documents/upload",
            files=files_data,
            data={'session_token': st.session_state.session_token},
            timeout=300
        )

        return response.json()
    except Exception as e:
        return {'error': str(e)}


def list_documents() -> List[Dict[str, Any]]:
    """
    Get list of uploaded documents.

    Returns:
        List of documents
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/documents/",
            params={'session_token': st.session_state.session_token},
            timeout=10
        )

        if response.status_code == 200:
            return response.json().get('documents', [])
        else:
            return []
    except Exception as e:
        st.error(f"Failed to fetch documents: {str(e)}")
        return []


def delete_document(doc_id: str) -> bool:
    """
    Delete a document.

    Args:
        doc_id: Document ID to delete

    Returns:
        True if deletion successful
    """
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/documents/{doc_id}",
            params={'session_token': st.session_state.session_token},
            timeout=10
        )

        return response.status_code == 200
    except Exception as e:
        st.error(f"Failed to delete document: {str(e)}")
        return False


def submit_query(question: str, top_k: int = None) -> Dict[str, Any]:
    """
    Submit a question to the backend.

    Args:
        question: User question
        top_k: Number of documents to retrieve

    Returns:
        Response data with answer and sources
    """
    try:
        payload = {
            "session_token": st.session_state.session_token,
            "question": question
        }

        if top_k:
            payload["top_k"] = top_k

        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {'error': response.json().get('error', 'Query failed')}
    except Exception as e:
        return {'error': str(e)}


def get_chat_history() -> List[Dict[str, Any]]:
    """
    Get chat history from backend.

    Returns:
        List of chat history entries
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/chat/history",
            params={'session_token': st.session_state.session_token},
            timeout=10
        )

        if response.status_code == 200:
            return response.json().get('history', [])
        else:
            return []
    except Exception as e:
        st.error(f"Failed to fetch chat history: {str(e)}")
        return []


def get_config() -> Dict[str, Any]:
    """
    Get current configuration.

    Returns:
        Configuration dictionary
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/config/",
            params={'session_token': st.session_state.session_token},
            timeout=10
        )

        if response.status_code == 200:
            return response.json().get('config', {})
        else:
            return {}
    except Exception as e:
        st.error(f"Failed to fetch configuration: {str(e)}")
        return {}


def update_config(updates: Dict[str, Any]) -> bool:
    """
    Update configuration.

    Args:
        updates: Configuration updates

    Returns:
        True if update successful
    """
    try:
        response = requests.put(
            f"{API_BASE_URL}/api/config/",
            json={
                "session_token": st.session_state.session_token,
                "updates": updates
            },
            timeout=10
        )

        return response.status_code == 200
    except Exception as e:
        st.error(f"Failed to update configuration: {str(e)}")
        return False


def render_login_page():
    """Render login page."""
    st.title("RAG Application - Login")

    st.markdown("### Welcome to the RAG Application")
    st.markdown("Please login to continue.")

    with st.form("login_form"):
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password", value="admin123")
        submit = st.form_submit_button("Login")

        if submit:
            if login(username, password):
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")


def render_main_page():
    """Render main application page."""
    # Header
    st.title("RAG Application")

    # Sidebar
    with st.sidebar:
        st.markdown(f"**Logged in as:** {st.session_state.username}")

        if st.button("Logout", type="secondary"):
            logout()
            st.rerun()

        st.markdown("---")

        # Configuration Panel
        st.markdown("### Configuration")

        with st.expander("RAG Parameters", expanded=False):
            config = get_config()

            chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=5000,
                value=config.get('chunking', {}).get('chunk_size', 1000),
                step=100
            )

            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=1000,
                value=config.get('chunking', {}).get('chunk_overlap', 200),
                step=50
            )

            top_k = st.number_input(
                "Top K Retrieval",
                min_value=1,
                max_value=20,
                value=config.get('retrieval', {}).get('top_k', 5),
                step=1
            )

            if st.button("Save Configuration"):
                updates = {
                    'chunking': {
                        'chunk_size': chunk_size,
                        'chunk_overlap': chunk_overlap
                    },
                    'retrieval': {
                        'top_k': top_k
                    }
                }

                if update_config(updates):
                    st.success("Configuration updated!")
                else:
                    st.error("Failed to update configuration")

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["Chat", "Document Upload", "Documents"])

    # Chat Tab
    with tab1:
        st.markdown("### Ask Questions")

        # Load chat history
        if not st.session_state.chat_history:
            st.session_state.chat_history = get_chat_history()

        # Display chat history
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(entry['question'])

            with st.chat_message("assistant"):
                st.markdown(entry['answer'])

                # Show sources in expander
                if entry.get('sources'):
                    with st.expander(f"Sources ({len(entry['sources'])} documents)"):
                        for i, source in enumerate(entry['sources'], 1):
                            st.markdown(f"**Source {i}:** {source['filename']}")
                            if 'page_number' in source:
                                st.markdown(f"*Page {source['page_number']}*")
                            elif 'row_number' in source:
                                st.markdown(f"*Row {source['row_number']}*")

                            st.markdown(f"**Confidence:** {source['score']:.3f}")
                            st.text(source['text'][:300] + "..." if len(source['text']) > 300 else source['text'])
                            st.markdown("---")

        # Chat input
        question = st.chat_input("Ask a question about your documents...")

        if question:
            # Display user message
            with st.chat_message("user"):
                st.markdown(question)

            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = submit_query(question)

                    if 'error' in response:
                        st.error(f"Error: {response['error']}")
                    else:
                        st.markdown(response['answer'])

                        # Show sources
                        if response.get('sources'):
                            with st.expander(f"Sources ({len(response['sources'])} documents)"):
                                for i, source in enumerate(response['sources'], 1):
                                    st.markdown(f"**Source {i}:** {source['filename']}")
                                    if 'page_number' in source:
                                        st.markdown(f"*Page {source['page_number']}*")
                                    elif 'row_number' in source:
                                        st.markdown(f"*Row {source['row_number']}*")

                                    st.markdown(f"**Confidence:** {source['score']:.3f}")
                                    st.text(source['text'][:300] + "..." if len(source['text']) > 300 else source['text'])
                                    st.markdown("---")

                        # Add to chat history
                        st.session_state.chat_history.append(response)

    # Document Upload Tab
    with tab2:
        st.markdown("### Upload Documents")

        uploaded_files = st.file_uploader(
            "Choose PDF or Excel files",
            type=['pdf', 'xlsx', 'xls'],
            accept_multiple_files=True
        )

        if st.button("Upload", type="primary"):
            if uploaded_files:
                with st.spinner("Uploading and processing documents..."):
                    result = upload_documents(uploaded_files)

                    if 'error' in result:
                        st.error(f"Upload failed: {result['error']}")
                    else:
                        if result.get('uploaded'):
                            st.success(f"Successfully uploaded {len(result['uploaded'])} document(s)!")

                            for doc in result['uploaded']:
                                st.markdown(f"- **{doc['filename']}** ({doc['chunks']} chunks)")

                        if result.get('errors'):
                            st.error("Some files failed to upload:")
                            for error in result['errors']:
                                st.markdown(f"- {error['filename']}: {error['error']}")
            else:
                st.warning("Please select files to upload")

    # Documents Tab
    with tab3:
        st.markdown("### Uploaded Documents")

        if st.button("Refresh", type="secondary"):
            st.rerun()

        documents = list_documents()

        if documents:
            for doc in documents:
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(f"**{doc.get('filename', 'Unknown')}**")
                    st.caption(f"Type: {doc.get('file_type', 'unknown')} | Uploaded: {doc.get('upload_date', 'N/A')}")

                with col2:
                    doc_id = doc.get('doc_id') or doc.get('filename')
                    if st.button("Delete", key=f"delete_{doc_id}"):
                        if delete_document(doc_id):
                            st.success("Document deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete document")

                st.markdown("---")
        else:
            st.info("No documents uploaded yet. Go to the 'Document Upload' tab to upload documents.")


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="RAG Application",
        page_icon="📚",
        layout="wide"
    )

    init_session_state()

    # Check if user is logged in
    if not st.session_state.session_token:
        render_login_page()
    else:
        render_main_page()


if __name__ == "__main__":
    main()
