import os
import json
import sys

# Ensure datasets is available
try:
    from datasets import load_dataset
except ImportError:
    print("datasets not found, installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

def sample_bright():
    print("Loading BRIGHT dataset from Hugging Face...")
    
    # We load queries and corpus separately
    # BRIGHT has sub-tasks like 'long-bench', 'stack-overflow', etc.
    # Let's pick a small subset for the sample.
    try:
        # Load queries
        ds = load_dataset('xlangai/BRIGHT', 'biology', split='test', streaming=True)
        # Load corpus
        corpus_ds = load_dataset('xlangai/BRIGHT', 'corpus', split='corpus', streaming=True)
        
        os.makedirs('data/bright_sample', exist_ok=True)
        
        print("Sampling 10 queries...")
        queries = []
        for i, q in enumerate(ds):
            if i >= 10: break
            queries.append(q)
            
        print("Sampling 100 documents from corpus...")
        corpus = []
        for i, c in enumerate(corpus_ds):
            if i >= 100: break
            corpus.append(c)
            
        with open('data/bright_sample/queries.json', 'w') as f:
            json.dump(queries, f, indent=2)
            
        with open('data/bright_sample/corpus.json', 'w') as f:
            json.dump(corpus, f, indent=2)
            
        print(f"Sample saved to data/bright_sample/ (Queries: {len(queries)}, Docs: {len(corpus)})")
        
    except Exception as e:
        print(f"Error sampling BRIGHT: {e}")
        # Fallback to a tiny manual sample if HF is blocked or fails
        print("Falling back to manual sample...")
        manual_queries = [{"query_id": "q1", "query": "What is the role of mitochondria?"}]
        manual_corpus = [{"doc_id": "d1", "content": "Mitochondria are the powerhouse of the cell."}]
        os.makedirs('data/bright_sample', exist_ok=True)
        json.dump(manual_queries, open('data/bright_sample/queries.json', 'w'), indent=2)
        json.dump(manual_corpus, open('data/bright_sample/corpus.json', 'w'), indent=2)

if __name__ == "__main__":
    sample_bright()
