# Path to JSON knowledge base
DATA_PATH = "data/knowledge.json"

# Folder to store indexes
INDEX_DIR = "indexes"

# Path to your local Orca-Mini model
MODEL_PATH = "models/q5_0-orca-mini-3b.gguf"

# Sentence-transformer model for embeddings (vector search)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking config
CHUNK_SIZE = 200   # characters per chunk

# Retrieval config
TOP_K = 5          # number of top results to retrieve
