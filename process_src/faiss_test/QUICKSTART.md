# QUICK START GUIDE
## FAISS Testing for PyTorch Lightning Dataset - Windows 11

⏱️ **Time Required:** 15-30 minutes for quick test, 1-2 hours for full test

---

## 🎯 Quick Start (3 Commands)

```bash
# 1. Install (5 minutes)
python 01_setup_faiss_windows.py

# 2. Test with 1000 documents (5-10 minutes)
python 02_test_faiss_retrieval.py pl_dataset/basic/train.jsonl --max-docs 1000 --output-dir faiss_test

# 3. Evaluate (1 minute)
python 03_create_manual_baseline.py pl_dataset/basic/train.jsonl --auto
python 04_evaluate_retrieval.py faiss_test --baseline manual_baseline.json --visualize
```

---

## 📁 Files You Need

**Before starting, make sure you have:**
- `pl_dataset/basic/train.jsonl` - Your PyTorch Lightning dataset
- All 4 Python scripts in the same directory
- Python 3.8+ installed

---

## 🚀 Option 1: Automated (Easiest)

**Windows Users:**
```bash
# Double-click this file in Windows Explorer:
run_all_tests.bat

# Or run from command prompt:
run_all_tests.bat
```

This batch script will:
1. Install all dependencies
2. Ask you to choose quick or full test
3. Run FAISS testing
4. Create baseline (auto or manual)
5. Generate evaluation results

---

## 🔧 Option 2: Manual Step-by-Step

### Step 1: Install Dependencies (5 minutes)

```bash
python 01_setup_faiss_windows.py
```

**Expected output:**
```
✓ Python version is compatible
✓ FAISS installed successfully
✓ FAISS is working correctly!
```

### Step 2: Run FAISS Test (10-60 minutes depending on dataset size)

**For quick testing (recommended first time):**
```bash
python 02_test_faiss_retrieval.py pl_dataset/basic/train.jsonl ^
    --max-docs 1000 ^
    --output-dir faiss_quick_test ^
    --k 5
```

**For full dataset:**
```bash
python 02_test_faiss_retrieval.py pl_dataset/basic/train.jsonl ^
    --output-dir faiss_full_test ^
    --k 5
```

**What you'll see:**
```
Loading dataset from: pl_dataset/basic/train.jsonl
✓ Loaded 1000 documents
Generating embeddings using sentence-transformer...
100%|███████████████████| 1000/1000
✓ Generated embeddings shape: (1000, 384)
Building FAISS index...
✓ Index built with 1000 vectors

Running Test Queries
Query 1: How to fix CUDA out of memory error...
  Top result: adjust_memory_allocation (score: 0.892)
  Latency: 14.23ms
...
✓ Results saved to faiss_quick_test/retrieval_results.json
```

### Step 3: Create Manual Baseline (2-10 minutes)

**Option A: Quick automatic baseline**
```bash
python 03_create_manual_baseline.py pl_dataset/basic/train.jsonl ^
    --auto ^
    --output manual_baseline.json
```

**Option B: Interactive (higher quality but slower)**
```bash
python 03_create_manual_baseline.py pl_dataset/basic/train.jsonl ^
    --output manual_baseline.json
```

### Step 4: Evaluate Results (1 minute)

```bash
python 04_evaluate_retrieval.py faiss_quick_test ^
    --baseline manual_baseline.json ^
    --visualize
```

**Output:**
```
AGGREGATE METRICS
====================================================================
Total Queries: 15
Average Latency: 12.45ms (±3.21ms)

Retrieval Quality:
  MRR (Mean Reciprocal Rank): 0.6234
  NDCG@10: 0.6789
  Precision@5: 0.4800

✓ Evaluation results saved to evaluation_metrics.json
✓ Saved metrics plot to visualizations/metrics_overview.png
```

---

## 📊 What You Get

### 1. Retrieval Results
**File:** `faiss_quick_test/retrieval_results.json`
- All 15 test queries
- Top-5 retrieved documents per query
- Similarity scores
- Latency measurements

### 2. Manual Baseline
**File:** `manual_baseline.json`
- Ground truth relevant documents
- Used for evaluation

### 3. Evaluation Metrics
**File:** `faiss_quick_test/evaluation_metrics.json`
- Precision@1, @5, @10
- Recall@1, @5, @10
- MRR (Mean Reciprocal Rank)
- NDCG@10
- MAP (Mean Average Precision)
- Performance by query type

### 4. Visualizations (if --visualize used)
**Folder:** `faiss_quick_test/visualizations/`
- `metrics_overview.png` - Bar charts of all metrics
- `performance_by_type.png` - Performance by query type

---

## ✅ Phase 1 Submission Checklist

For your project Phase 1, you need:

- [ ] **Dataset:** `pl_dataset/` (✓ You already have this)
- [ ] **Test Queries:** Built into script (✓ 15 diverse queries)
- [ ] **Retrieval Results:** `retrieval_results.json` ← Run Step 2
- [ ] **Manual Baseline:** `manual_baseline.json` ← Run Step 3
- [ ] **Evaluation Metrics:** `evaluation_metrics.json` ← Run Step 4
- [ ] **Evaluator Code:** `04_evaluate_retrieval.py` (✓ Provided)
- [ ] **Phase 1 Report:** Write your report with these results

---

## 🎓 What to Report in Phase 1

Include in your report:

### Dataset Statistics
- Total documents: ___ (from your dataset)
- Document types: Code functions with docstrings
- Data source: PyTorch Lightning repository

### RAG System Details
- Vector store: FAISS
- Embedding method: Sentence Transformers (all-MiniLM-L6-v2)
- Index type: IndexFlatL2 (exact search)
- Retrieved results: Top-5 per query

### Performance Metrics
Copy from `evaluation_metrics.json`:
- MRR: ___
- NDCG@10: ___
- Precision@5: ___
- Average latency: ___ms

### Query Analysis
- Total test queries: 15
- Query types: debugging (20%), API usage (33%), implementation (27%), conceptual (20%)
- Performance by type: (from evaluation results)

### Example Success/Failure
Pick one good and one poor retrieval from results and explain why.

---

## ⚡ Performance Tuning

### If tests are too slow:
```bash
# Use TF-IDF (faster but less accurate)
--embedding tfidf

# Reduce documents
--max-docs 500

# Use smaller model
--model paraphrase-MiniLM-L3-v2
```

### If you need better accuracy:
```bash
# Use better model (slower)
--model all-mpnet-base-v2

# Retrieve more results
--k 10

# Use full dataset (no --max-docs)
```

### If you have a GPU:
```bash
# Install GPU version
pip install faiss-gpu

# The scripts will automatically use GPU if available
```

---

## 🐛 Common Issues

### "ModuleNotFoundError: No module named 'faiss'"
```bash
pip install faiss-cpu
```

### "File not found: pl_dataset/basic/train.jsonl"
Update the path to match your dataset location:
```bash
python 02_test_faiss_retrieval.py path/to/your/dataset.jsonl
```

### "Out of memory"
Reduce dataset size:
```bash
--max-docs 500
```

### "CUDA out of memory" (if using GPU)
Switch to CPU version:
```bash
pip uninstall faiss-gpu
pip install faiss-cpu
```

---

## 📞 Need Help?

1. Check the [README.md](README.md) for detailed documentation
2. Review error messages - they usually explain the issue
3. Try with smaller dataset first (`--max-docs 100`)
4. Verify your Python version: `python --version` (need 3.8+)

---

## 🎯 Success Criteria

You're ready for Phase 1 submission when you have:
- ✅ All 4 output files generated
- ✅ MRR > 0.5 (acceptable for Phase 1)
- ✅ Evaluation completes without errors
- ✅ Results visualizations created (optional but recommended)

**Typical good results for Phase 1:**
- MRR: 0.5 - 0.7
- NDCG@10: 0.6 - 0.8
- Latency: < 50ms per query

---

## 🚀 Next Steps After Phase 1

For Phase 2, you'll improve your RAG system:
1. Try different embedding models
2. Implement hybrid search (semantic + keyword)
3. Add reranking
4. Optimize chunking strategies
5. Compare with other vector stores (Qdrant, Weaviate)

---

**Good luck! 🎉**
