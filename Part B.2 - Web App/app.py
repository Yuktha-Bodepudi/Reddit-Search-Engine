import os
import subprocess
import json
from flask import Flask, render_template, request
import platform
import numpy as np
from utils import query_bert

app = Flask(__name__)

# Define base directories.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "utils", "bert.index")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load resources once at startup.
faiss_index = query_bert.load_faiss_index(INDEX_FILE)
posts_data = query_bert.load_posts_data(DATA_DIR)
bert_model = query_bert.load_bert_model("sentence-transformers/all-distilroberta-v1")

def search_lucene(query, top_k):
    # Determine OS for correct classpath separator
    separator = ";" if platform.system() == "Windows" else ":"

    # Get the base directory dynamically (directory where this script is located)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Dependencies folder where Lucene & JSON-simple JARs are located
    dependencies_dir = os.path.join(BASE_DIR, "dependencies")

    # Java class to execute
    lucene_search_class = "LuceneSearch"

    # Construct the Java command
    command = [
        "java",
        "-cp",
        f"{BASE_DIR}{separator}{dependencies_dir}/*",
        lucene_search_class,
        str(top_k),
        query  # Query as a single argument
    ]

    print("Executing command:", " ".join(command))  # Debugging output

    try:
        # Run the Java program and capture output
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            return {"error": "Error executing Java program", "details": result.stderr}

        # Parse JSON output from Java program
        output = result.stdout.strip()
        return json.loads(output)
    
    except Exception as e:
        return {"error": "Exception occurred", "details": str(e)}



@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}
    if request.method == 'POST':
        query = request.form['query']
        top_k = request.form.get('top_k', 10)  # Default: 10
        index_type = request.form['index_type']
        
        if index_type == 'lucene':
            results = search_lucene(query, top_k)
        elif index_type == 'bert':
            results = query_bert.search_bert(query, top_k, faiss_index, posts_data, bert_model)

    return render_template('index.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)
