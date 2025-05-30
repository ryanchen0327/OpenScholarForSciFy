# OpenScholar Context Initialization Fix

## Issue Description

The OpenScholar system was encountering an `IndexError: list index out of range` when trying to run demos with empty contexts. This happened because:

1. **Demo Input Format**: The demo input data (`demo_input.jsonl`) contains only basic question data without pre-retrieved contexts
2. **Reranking Expectation**: The system was trying to rerank documents (`--ranking_ce` flag) when no documents existed yet
3. **Empty List Access**: The `rerank_paragraphs_bge()` function attempted to access `paragraph_texts[0]` on an empty list

## Root Cause

The error occurred in this sequence:
```
run.py:728 → reranking_passages_cross_encoder() → rerank_paragraphs_bge() → paragraph_texts[0]
```

When starting with empty contexts (`item["ctxs"] = []`), the reranking function tried to access the first element of an empty list.

## Solution Applied

### 1. Fixed Empty Context Handling in `rerank_paragraphs_bge()`

**File**: `src/open_scholar.py` (lines 41-47)

```python
def rerank_paragraphs_bge(query, paragraphs, reranker, norm_cite=False, start_index=0, use_abstract=False):
    paragraphs = [p for p in paragraphs if p["text"] is not None]
    
    # Return empty results if no paragraphs to rerank
    if len(paragraphs) == 0:
        print("No paragraphs to rerank - returning empty results")
        return [], {}, {}
    
    # ... rest of function
```

### 2. Added Smart Reranking Skip in `run()` Method

**File**: `src/open_scholar.py` (lines 726-738)

```python
if ranking_ce is True:
    # Check if there are contexts to rerank
    if len(item["ctxs"]) == 0:
        print("⚠️  Skipping reranking: No contexts available to rerank")
        print("   This is normal when starting with empty contexts - documents will be retrieved during feedback phase")
        item["ranked_results"] = {}
        item["id_mapping"] = {}
    else:
        item["ctxs"], ranked_results, id_mapping = self.reranking_passages_cross_encoder(item, batch_size=1, llama3_chat=llama3_chat, task_name=task_name, use_abstract=False)
        item["ranked_results"] = ranked_results
        item["id_mapping"] = id_mapping
```

### 3. Fixed Input Data Processing

**File**: `run.py` (lines 67-88)

```python
def process_input_data(data, use_contexts=True):
    # ... existing code ...
    
    elif use_contexts is True and "ctxs" not in item:
        # Initialize empty ctxs for RAG pipeline when contexts will be retrieved
        item["ctxs"] = []
        item["original_ctxs"] = []
```

And updated the function call:
```python
# process input data
data = process_input_data(data, use_contexts=args.use_contexts)
```

## Expected Behavior Now

### ✅ **Working Demo Flow**

1. **Input Processing**: Demo questions are loaded with empty `ctxs` arrays
2. **Reranking Skip**: System detects empty contexts and skips reranking with helpful message
3. **Initial Generation**: May generate basic response without contexts (depending on configuration)
4. **Feedback Retrieval**: If `--feedback` is enabled, retrieves relevant documents from configured sources
5. **Document Reranking**: Retrieved documents are then reranked and filtered
6. **Final Generation**: Improved response using retrieved contexts

### 📝 **Demo Configuration Options**

**Option 1: Basic RAG (No Feedback)**
```bash
python run.py \
  --input_file demo_input.jsonl \
  --model_name gpt-4o-mini \
  --use_contexts \
  --ranking_ce \
  --reranker OpenScholar/OpenScholar_Reranker \
  --output_file demo_basic_output.json
```
- Skips reranking (no contexts)
- Generates response without retrieval
- Useful for testing basic functionality

**Option 2: Enhanced Multi-Source RAG**
```bash
python run.py \
  --input_file demo_input.jsonl \
  --model_name gpt-4o-mini \
  --use_contexts \
  --ranking_ce \
  --reranker OpenScholar/OpenScholar_Reranker \
  --feedback \
  --ss_retriever \
  --use_pes2o_feedback \
  --output_file demo_enhanced_output.json
```
- Skips initial reranking
- Uses feedback retrieval to find documents
- Reranks and filters retrieved documents
- Generates improved response with citations

## Alternative Approaches

### Option A: Pre-Retrieval Workflow

For users who want to pre-populate contexts, use `src/use_search_apis.py`:

```bash
# Step 1: Retrieve documents
python src/use_search_apis.py \
  --input_file demo_input.jsonl \
  --output_file demo_with_contexts.jsonl \
  --use_semantic_scholar

# Step 2: Run OpenScholar with pre-retrieved contexts  
python run.py \
  --input_file demo_with_contexts.jsonl \
  --use_contexts \
  --ranking_ce \
  --output_file demo_output.json
```

### Option B: Zero-Shot Mode

For testing without retrieval:
```bash
python run.py \
  --input_file demo_input.jsonl \
  --model_name gpt-4o-mini \
  --zero_shot \
  --output_file demo_zero_shot.json
```

## Summary

The fix ensures OpenScholar gracefully handles the common use case of starting with questions-only input data, which is the typical workflow for users who want the system to handle document retrieval automatically during the feedback phase.

**Key Benefits:**
- ✅ No more crashes on empty contexts
- ✅ Clear user messaging about system behavior  
- ✅ Supports both pre-retrieval and feedback-retrieval workflows
- ✅ Backward compatible with existing data formats
- ✅ Maintains all existing functionality 