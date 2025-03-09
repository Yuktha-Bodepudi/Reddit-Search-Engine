import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_path):
    """
    Loads a FAISS index from the specified file path.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    index = faiss.read_index(index_path)
    return index

def load_posts_data(data_dir):
    """
    Loads post data from all files in the specified directory.
    Assumes that each file (e.g., reddit_batch_1.txt) contains one JSON record per line.
    Each record should include keys such as "title", "body", and either "permalink" or "url".
    """
    posts = []
    # Process files in sorted order (to maintain the same order as when indexing)
    for filename in sorted(os.listdir(data_dir)):
        if filename.startswith("reddit_batch_") and filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        post = json.loads(line.strip())
                    except Exception:
                        # If not valid JSON, store as raw text.
                        post = {"title": "Untitled", "body": line.strip(), "permalink": "", "url": ""}
                    posts.append(post)
    return posts

def load_bert_model(model_name="sentence-transformers/all-distilroberta-v1"):
    """
    Loads a SentenceTransformer model.
    Make sure this is the same model (or compatible) used to create the embeddings.
    """
    model = SentenceTransformer(model_name)
    return model

def get_query_embedding(model, query, normalize=False):
    """
    Converts a text query into a BERT embedding.
    Optionally normalizes the embedding.
    Returns a numpy array of shape (1, embedding_dim) of type float32.
    """
    embedding = model.encode(query, convert_to_numpy=True)
    if normalize:
        norm = np.linalg.norm(embedding)
        embedding = embedding / norm if norm != 0 else embedding
    return embedding.reshape(1, -1).astype(np.float32)

def search_bert(query, top_k, index, posts_data, model, normalize=False):
    """
    Searches the FAISS index for the top_k most similar posts for a given query.
    Converts the query to an embedding, performs the FAISS search, and uses the returned doc_ids
    to look up metadata (Title, Body, URL) from posts_data.
    
    Returns:
        dict: { "totalHits": <int>, "results": [ { "doc_id": ..., "Title": ..., "Body": ..., "URL": ..., "score": ... }, ... ] }
    """
    try:
        top_k = int(top_k)
        # Convert the query into an embedding.
        query_vector = get_query_embedding(model, query, normalize=normalize)
        # Perform FAISS search.
        distances, indices = index.search(query_vector, top_k)
        results = []
        for rank in range(top_k):
            doc_id = indices[0][rank]
            score = float(distances[0][rank])
            if doc_id < len(posts_data):
                post = posts_data[doc_id]
                title = post.get("title", "No Title")
                body = post.get("body", "No Body")
                permalink = post.get("permalink", "")
                # Construct full URL from permalink if necessary.
                if permalink and not permalink.startswith("http"):
                    url = f"https://www.reddit.com{permalink}"
                else:
                    url = post.get("url", "No URL")
            else:
                title = "Unknown"
                body = "Unknown"
                url = "Unknown"
            results.append({
                "doc_id": int(doc_id),
                "Title": title,
                "Body": body,
                "URL": url,
                "score": score
            })
        return {"totalHits": len(results), "results": results}
    
    except Exception as e:
        return {"error": "Exception occurred", "details": str(e)}

# For testing this module directly.
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_FILE = os.path.join(BASE_DIR, "bert.index")
    # Assuming the data folder is one level up from utils
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")
    faiss_index = load_faiss_index(INDEX_FILE)
    posts_data = load_posts_data(DATA_DIR)
    bert_model = load_bert_model("sentence-transformers/all-distilroberta-v1")
    query = "Who is Steve Jobs?"
    top_k = 4
    results = search_bert(query, top_k, faiss_index, posts_data, bert_model)
    print(json.dumps(results, indent=2))
