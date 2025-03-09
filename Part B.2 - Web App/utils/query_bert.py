import os
import json
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# --------------------------
# Data Loading
# --------------------------
def load_reddit_posts(data_dir):
    """
    Loads Reddit posts from all files in the specified directory.
    Assumes each file (e.g., reddit_batch_1.txt) contains one JSON record per line.
    Each record must have at least 'title' and 'body', and optionally 'permalink'.
    
    Returns:
      posts_texts: List of combined post texts (for embedding)
      posts_data:  List of post metadata dictionaries.
    """
    posts_texts = []
    posts_data = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.startswith("reddit_batch_") and filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    try:
                        post = json.loads(line.strip())
                        # Combine title and body (add comments if desired)
                        post_text = post['title'] + " " + post['body']
                        for comment in post.get('comments', []):
                            post_text += " " + comment.get('body', "")
                        posts_texts.append(post_text)
                        posts_data.append({
                            "title": post.get("title", "Untitled"),
                            "body": post.get("body", ""),
                            "permalink": post.get("permalink", "")
                        })
                    except Exception as e:
                        print(f"Error processing line in {filename}: {e}")
    return posts_texts, posts_data

# --------------------------
# Embedding Generation
# --------------------------
def generate_embeddings_batch(texts, tokenizer, model, batch_size=32):
    """
    Generates embeddings for a list of texts in batches using mean pooling.
    Returns a torch.Tensor of shape (num_texts, hidden_dim).
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = {"input_ids": [], "attention_mask": []}
        for text in batch:
            new_tokens = tokenizer.encode_plus(
                text,
                max_length=512,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            tokens["input_ids"].append(new_tokens["input_ids"][0])
            tokens["attention_mask"].append(new_tokens["attention_mask"][0])
        tokens["input_ids"] = torch.stack(tokens["input_ids"])
        tokens["attention_mask"] = torch.stack(tokens["attention_mask"])
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        attention_mask = tokens["attention_mask"]  # (batch, seq_len)
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, dim=1)
        summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / summed_mask  # (batch, hidden_dim)
        all_embeddings.append(mean_pooled)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

def normalize_embeddings(embeddings):
    """
    Converts a tensor of embeddings to a NumPy array and normalizes each vector to unit length.
    """
    embeddings_np = embeddings.cpu().numpy()
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    normalized = embeddings_np / norms
    return normalized

# --------------------------
# FAISS Indexing
# --------------------------
def build_faiss_index(embeddings):
    """
    Builds a FAISS index (using inner product) from the given normalized embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def load_faiss_index(index_path):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    index = faiss.read_index(index_path)
    return index

# --------------------------
# Query Embedding
# --------------------------
def convert_to_embedding(query, tokenizer, model):
    """
    Converts a query string into an embedding using mean pooling.
    Follows the same procedure as used during indexing.
    
    Returns:
      A torch.Tensor of shape (hidden_dim,).
    """
    tokens = tokenizer.encode_plus(
        query,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
    attention_mask = tokens['attention_mask']  # (1, seq_len)
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, dim=1)
    summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    mean_pooled = summed / summed_mask
    return mean_pooled[0]

# --------------------------
# Searching
# --------------------------
def search_bert(query, top_k, index, posts_data, tokenizer, model):
    """
    Converts the query to an embedding (using the same mean pooling),
    normalizes it, performs a FAISS search over a large set of neighbors,
    filters results by a similarity threshold, and returns the top_k results
    along with the total number of hits that exceed the threshold.

    Returns a dictionary in the format:
      {
          "totalHits": <int>,
          "results": [
             { "doc_id": <int>, "Title": <str>, "Body": <str>, "URL": <str>, "score": <float> },
             ...
          ]
      }
    """
    top_k = int(top_k)
    # For total hit count, search over all indexed vectors
    top_n = index.ntotal
    
    # Compute query embedding.
    query_embedding = convert_to_embedding(query, tokenizer, model)
    query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
    query_embedding_normalized = query_embedding_np / np.linalg.norm(query_embedding_np)
    
    # Perform FAISS search for top_n results.
    distances, indices = index.search(query_embedding_normalized, top_n)
    
    # Debug: Print the top similarity score.
    max_sim = distances[0][0]
    # Dynamic Threshold
    cut_off = 0.5 if max_sim > 0.9 else 0.4 if max_sim > 0.8 else 0.3 if max_sim > 0.6 else 0.2 if max_sim > 0.4 else 0.1 if max_sim > 0.2 else 0.05

    threshold = max_sim - cut_off
    print(f"Max similarity: {distances[0][0]} and Threshold: {threshold}")
    
    # Filter results by threshold.
    filtered_results = []
    for i in range(len(distances[0])):
        if distances[0][i] >= threshold:
            filtered_results.append((indices[0][i], distances[0][i]))
    
    total_hits = len(filtered_results)
    
    # Only take the top_k filtered results for display.
    filtered_results = filtered_results[:top_k]
    
    results = []
    for doc_id, score in filtered_results:
        if doc_id < len(posts_data):
            post = posts_data[doc_id]
            title = post.get("title", "No Title")
            body = post.get("body", "No Body")
            permalink = post.get("permalink", "")
            if permalink and not permalink.startswith("http"):
                url = f"https://www.reddit.com{permalink}"
            else:
                url = post.get("url", "No URL")
        else:
            title, body, url = "Unknown", "Unknown", "Unknown"
        results.append({
            "doc_id": int(doc_id),
            "Title": title,
            "Body": body,
            "URL": url,
            "score": float(score)
        })
    
    return {"totalHits": total_hits, "results": results}


# --------------------------
# Model Loading
# --------------------------
def load_transformers_model(model_name="sentence-transformers/all-distilroberta-v1"):
    """
    Loads the Hugging Face AutoTokenizer and AutoModel.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# --------------------------
# Reindexing Function
# --------------------------
def reindex_data(data_dir, model_name="sentence-transformers/all-distilroberta-v1", batch_size=32):
    """
    Reindexes all Reddit posts from the data directory.
      - Loads posts and their texts.
      - Generates embeddings in batches.
      - Normalizes the embeddings.
      - Builds and returns a FAISS index along with the posts metadata.
    """
    tokenizer, model = load_transformers_model(model_name)
    posts_texts, posts_data = load_reddit_posts(data_dir)
    embeddings = generate_embeddings_batch(posts_texts, tokenizer, model, batch_size=batch_size)
    normalized = normalize_embeddings(embeddings)
    index = build_faiss_index(normalized)
    return index, posts_data

# --------------------------
# For Testing / Standalone Query
# --------------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    # Save/load index in the utils folder (adjust as needed)
    index_save_path = os.path.join(base_dir, "bert.index")
    
    print("Reindexing data...")
    index, posts_data = reindex_data(data_dir)
    print("Index built. Number of vectors:", index.ntotal)
    
    # Save the FAISS index to disk.
    faiss.write_index(index, index_save_path)
    print("Index saved to:", index_save_path)
    
    # Load the model for query processing.
    tokenizer, model = load_transformers_model()
    
    query = input("Enter your query: ")
    top_k = int(input("Enter top K results: "))
    results = search_bert(query, top_k, index, posts_data, tokenizer, model)
    
    print("\nResults:")
    for i, res in enumerate(results["results"]):
        print(f"Rank {i+1}:")
        print(f"Title: {res['Title']}")
        print(f"Score: {res['score']}")
        print(f"URL: {res['URL']}")
        print(f"Body: {res['Body']}")
        print("-----")
