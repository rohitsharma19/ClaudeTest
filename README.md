# RAG Application

A configurable Retrieval-Augmented Generation (RAG) application with a Streamlit frontend and Flask backend. The system is modular, allowing easy swapping of components (LLM providers, vector databases, embedding models) through configuration files.

## Features

- **Multiple LLM Providers**: OpenAI, Anthropic (Claude), Azure OpenAI, Ollama
- **Multiple Vector Databases**: ChromaDB, FAISS
- **Multiple Embedding Providers**: OpenAI Embeddings, HuggingFace, Sentence Transformers
- **Document Processing**: PDF and Excel file support
- **Interactive Chat Interface**: Ask questions about your documents
- **Session-based Authentication**: Simple username/password authentication
- **Configurable Parameters**: Adjust chunk size, overlap, and retrieval settings
- **Chat History**: Persistent conversation history per user
- **Source Attribution**: View source documents with confidence scores

## Architecture

### Technology Stack

- **Frontend**: Streamlit
- **Backend**: Flask REST API
- **Configuration**: JSON
- **Language**: Python 3.9+

### Project Structure

```
rag-application/
├── backend/
│   ├── app.py                    # Flask app initialization
│   ├── config/
│   │   ├── config.json           # Configuration file
│   │   └── config_manager.py     # Configuration manager
│   ├── managers/
│   │   ├── llm_manager.py        # LLM providers
│   │   ├── embedding_manager.py  # Embedding providers
│   │   ├── vector_db_manager.py  # Vector database managers
│   │   ├── retrieval_manager.py  # Retrieval orchestration
│   │   └── auth_manager.py       # Authentication
│   ├── processors/
│   │   ├── base_processor.py     # Base processor class
│   │   ├── pdf_processor.py      # PDF document processor
│   │   └── excel_processor.py    # Excel document processor
│   ├── routes/
│   │   ├── auth_routes.py        # Authentication endpoints
│   │   ├── document_routes.py    # Document management endpoints
│   │   ├── query_routes.py       # Query endpoints
│   │   └── config_routes.py      # Configuration endpoints
│   └── utils/
│       ├── logger.py             # Logging utilities
│       └── helpers.py            # Helper functions
├── frontend/
│   └── app.py                    # Streamlit application
├── data/
│   ├── uploads/                  # Uploaded documents
│   ├── vectordb/                 # Vector database storage
│   └── chat_history/             # Chat history files
├── logs/                         # Application logs
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) API keys for LLM providers (OpenAI, Anthropic, etc.)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-application
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Application

Edit `backend/config/config.json` to configure your LLM provider, embeddings, and other settings.

**Important**: Add your API keys to the configuration file or set them as environment variables.

#### Example Configuration

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "api_key": "your-openai-api-key",
    "base_url": ""
  },
  "embeddings": {
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "api_key": "your-openai-api-key"
  },
  "vector_db": {
    "type": "chromadb",
    "persist_directory": "./data/vectordb"
  },
  "chunking": {
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "retrieval": {
    "top_k": 5
  },
  "system_prompt": "You are a helpful assistant...",
  "authentication": {
    "username": "admin",
    "password": "admin123"
  },
  "logging": {
    "level": "INFO",
    "log_to_file": true,
    "log_file_path": "./logs/app.log"
  }
}
```

## Running the Application

### Start the Flask Backend

```bash
# From the project root directory
cd backend
python app.py
```

The backend will start on `http://localhost:5000`

### Start the Streamlit Frontend

In a new terminal:

```bash
# From the project root directory
cd frontend
streamlit run app.py
```

The frontend will open in your browser at `http://localhost:8501`

## Usage

### 1. Login

- Default credentials:
  - Username: `admin`
  - Password: `admin123`

### 2. Upload Documents

- Navigate to the "Document Upload" tab
- Select PDF or Excel files
- Click "Upload" to process and index the documents

### 3. Ask Questions

- Go to the "Chat" tab
- Type your question in the chat input
- View the answer along with source documents and confidence scores

### 4. Manage Documents

- Navigate to the "Documents" tab
- View all uploaded documents
- Delete documents as needed

### 5. Configure Settings

- Open the "Configuration" panel in the sidebar
- Adjust:
  - Chunk Size
  - Chunk Overlap
  - Top K Retrieval
- Click "Save Configuration" to apply changes

## Configuration Guide

### LLM Providers

#### OpenAI

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "api_key": "your-api-key"
  }
}
```

#### Anthropic (Claude)

```json
{
  "llm": {
    "provider": "anthropic",
    "model": "claude-3-sonnet-20240229",
    "api_key": "your-api-key"
  }
}
```

#### Azure OpenAI

```json
{
  "llm": {
    "provider": "azure_openai",
    "model": "your-deployment-name",
    "api_key": "your-api-key",
    "base_url": "https://your-resource.openai.azure.com/"
  }
}
```

#### Ollama (Local)

```json
{
  "llm": {
    "provider": "ollama",
    "model": "llama2",
    "base_url": "http://localhost:11434"
  }
}
```

### Embedding Providers

#### OpenAI

```json
{
  "embeddings": {
    "provider": "openai",
    "model": "text-embedding-ada-002",
    "api_key": "your-api-key"
  }
}
```

#### HuggingFace

```json
{
  "embeddings": {
    "provider": "huggingface",
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

#### Sentence Transformers

```json
{
  "embeddings": {
    "provider": "sentence_transformers",
    "model": "all-MiniLM-L6-v2"
  }
}
```

### Vector Databases

#### ChromaDB

```json
{
  "vector_db": {
    "type": "chromadb",
    "persist_directory": "./data/vectordb"
  }
}
```

#### FAISS

```json
{
  "vector_db": {
    "type": "faiss",
    "persist_directory": "./data/vectordb"
  }
}
```

## Environment Variables

You can use environment variables instead of hardcoding API keys in the configuration file:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

## API Endpoints

### Authentication

- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `POST /api/auth/validate` - Validate session

### Documents

- `POST /api/documents/upload` - Upload documents
- `GET /api/documents/` - List documents
- `DELETE /api/documents/<id>` - Delete document

### Query

- `POST /api/query` - Submit question
- `GET /api/chat/history` - Get chat history
- `DELETE /api/chat/history` - Clear chat history

### Configuration

- `GET /api/config/` - Get configuration
- `PUT /api/config/` - Update configuration

## Troubleshooting

### Backend Won't Start

1. Check that all dependencies are installed: `pip install -r requirements.txt`
2. Verify Python version: `python --version` (should be 3.9+)
3. Check the configuration file for syntax errors
4. Review logs in `./logs/app.log`

### Frontend Can't Connect to Backend

1. Ensure the backend is running on `http://localhost:5000`
2. Check for CORS errors in browser console
3. Verify the `API_BASE_URL` in `frontend/app.py`

### Documents Not Uploading

1. Check file format (only PDF and Excel supported)
2. Verify file size (max 100MB)
3. Check backend logs for processing errors
4. Ensure sufficient disk space

### LLM Errors

1. Verify API key is correct
2. Check API key has sufficient credits/quota
3. Ensure correct model name in configuration
4. Review error messages in logs

### Vector Database Errors

1. Check persist directory exists and is writable
2. For ChromaDB: ensure `chromadb` package is installed
3. For FAISS: ensure `faiss-cpu` package is installed
4. Try deleting the vector database directory and re-uploading documents

## Limitations

Current version does not include:

- Unit tests
- Hybrid search (BM25 + semantic)
- Re-ranking mechanisms
- Chat memory in LLM context
- Multi-sheet Excel processing
- Image extraction from PDFs
- Multi-user/multi-tenant support

## License

This project is provided as-is for educational and development purposes.

## Support

For issues and questions, please refer to the troubleshooting section or check the application logs.
