import os
import json
import zipfile
from rag import retrieve, load_index_and_meta
from config import INDEX_DIR

# Config
QUERIES_FILE = os.path.join("data", "queries.json")
KNOWLEDGE_FILE = os.path.join("data", "knowledge.json")
OUTPUT_FOLDER = "submission_jsons"
ZIP_NAME = "startup_PS4.zip"
TOP_K = 10  # more top chunks to increase relevance

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load queries
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"✅ Loaded {len(queries)} queries")

    # Load index and BM25
    index, meta, bm25 = load_index_and_meta(KNOWLEDGE_FILE)
    print(f"✅ Loaded RAG index with {len(meta)} documents")

    # Process each query
    for q in queries:
        query_num = q["query_num"]
        query_text = q["query"]

        hits = retrieve(query_text, index, meta, bm25, top_k=TOP_K)

        # Format response as Doc{doc_id}_Chunk{chunk_id}
        response_files = [f"Doc{h['doc_id']}_Chunk{h['chunk_id']}" for h in hits]

        output_json = {
            "query": query_text,
            "response": response_files
        }

        json_path = os.path.join(OUTPUT_FOLDER, f"{query_num}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(output_json, jf, indent=2, ensure_ascii=False)

    # Zip all JSONs
    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(OUTPUT_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=file)

    print(f"✅ All JSON files generated and zipped as {ZIP_NAME}")

if __name__ == "__main__":
    main()
