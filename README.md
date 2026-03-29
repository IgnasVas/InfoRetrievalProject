# Information Retrieval Pipeline Study

Empirical study comparing sparse (BM25), dense (Sentence-BERT), and reranking pipelines on the ANTIQUE collection.

## Requirements

- Python 3.8+
- Java (OpenJDK, required for PyTerrier/BM25)

## Setup on macOS

1. **Install OpenJDK** (if not already installed):
   ```bash
   brew install openjdk
   ```

2. **Configure Java environment** (add to `~/.zshrc`):
   ```bash
   export JAVA_HOME=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home
   export JVM_PATH=/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib
   ```

3. **Reload shell**:
   ```bash
   source ~/.zshrc
   ```

## Quick Start

```bash
# Install dependencies
python -m pip install -r requirements.txt
# or .venv/bin/python -m pip install -r requirements.txt

# Download and preprocess data
python scripts/01_download_antique.py
# or .venv/bin/python scripts/01_download_antique.py            

python scripts/02_preprocess_data.py
# or .venv/bin/python scripts/02_preprocess_data.py

# Run experiments
python scripts/03_evaluate.py
# or .venv/bin/python scripts/03_evaluate.py
```

Results saved to `results/results.json`.

## Project Structure

```
├── data/           # Corpus, queries, relevance judgments
├── src/            # Modules (config, utils, retrievers, reranker, pipeline, evaluation)
├── scripts/        # Data processing and experiments
├── results/        # Experiment results (JSON)
├── config.yaml     # Configuration
└── requirements.txt
```

## Pipelines Evaluated

1. **BM25** - Sparse retrieval
2. **Dense** - Sentence-BERT embeddings  
3. **BM25 + Reranker** - BM25 then cross-encoder
4. **Dense + Reranker** - Dense then cross-encoder

## Metrics

- Recall@k
- nDCG@10
- MRR@10

## Configuration

Edit `config.yaml` to change:
- Candidate depths (20, 50, 100)
- Model names
- Batch sizes
- Device (GPU/CPU)

## Research Questions

1. How do the four pipelines compare?
2. How does candidate depth affect reranking?
3. Query-type specific performance (per-query results saved)
