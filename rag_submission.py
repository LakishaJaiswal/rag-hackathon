# rag_submission.py
import os
import json
import zipfile
from rag import retrieve, load_index_and_meta  # Your offline RAG functions
from config import INDEX_DIR

# Config
QUERIES_FILE = "all_queries.json"   # Your 10k queries
OUTPUT_FOLDER = "submission_jsons"
ZIP_NAME = "startup_PS4.zip"
TOP_K = 5  # top chunks to include per query

def main():
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load queries
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # Load RAG index and metadata
    index, meta, bm25 = load_index_and_meta()

    # Process each query
    for q in queries:
        query_num = q["query_num"]
        query_text = q["query"]

        # Retrieve top chunks
        hits = retrieve(query_text, index, meta, bm25, top_k=TOP_K)

        # Build response list
        response_files = [f"Doc{h['doc_id']}_Chunk{h['chunk_id']}" for h in hits]

        # Create JSON for this query
        output_json = {
            "query": query_text,
            "response": response_files
        }

        # Save JSON file
        json_path = os.path.join(OUTPUT_FOLDER, f"{query_num}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(output_json, jf, indent=2, ensure_ascii=False)

    # Zip all JSONs
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(OUTPUT_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=file)  # arcname avoids full path in zip

    print(f"✅ All JSON files generated and zipped as {ZIP_NAME}")

if __name__ == "__main__":
    main()
