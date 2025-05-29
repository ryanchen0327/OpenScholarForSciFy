# Score-Based Document Filtering for OpenScholar

OpenScholar now supports intelligent score-based document filtering that adapts to the actual quality distribution of retrieved documents, replacing the traditional fixed top-N approach with dynamic thresholds based on OpenScholar reranker scores.

## 🎯 Overview

Instead of always selecting exactly N documents regardless of their relevance scores, score-based filtering uses statistical thresholds to automatically filter out documents that fall below a certain quality threshold. This ensures that only genuinely relevant documents are used for generation.

**🔧 Consistent Filtering**: Score-based filtering is applied consistently to **both initial retrieval** and **feedback-retrieved documents** in the self-reflective generation pipeline, ensuring uniform quality standards throughout the process.

## 📊 Available Threshold Types

### 1. Average Threshold (`average`)
**Description**: Filters documents that score below the average OpenScholar reranker score.

**Use Case**: Best for general-purpose filtering where you want to exclude below-average documents.

**Example**:
```bash
python run.py --input_file input.jsonl --model_name gpt2 \
              --output_file output.json --use_contexts \
              --ranking_ce --reranker OpenScholar/OpenScholar_Reranker \
              --use_score_threshold --score_threshold_type average
```

**Behavior**: If OpenScholar scores are [6.5, 2.0, 1.4, 0.9], average = 2.7, so only documents scoring ≥ 2.7 are kept (2 documents).

---

### 2. Median Threshold (`median`)
**Description**: Filters documents that score below the median OpenScholar reranker score.

**Use Case**: More robust to outliers than average; good when you have extreme high/low scores.

**Example**:
```bash
python run.py --input_file input.jsonl --model_name gpt2 \
              --output_file output.json --use_contexts \
              --ranking_ce --reranker OpenScholar/OpenScholar_Reranker \
              --use_score_threshold --score_threshold_type median
```

**Behavior**: If OpenScholar scores are [6.5, 2.0, 1.4, 0.9], median = 1.7, so documents scoring ≥ 1.7 are kept (3 documents).

---

### 3. 25th Percentile (`percentile_25`)
**Description**: Filters documents that score below the 25th percentile (keeps top 75%).

**Use Case**: Liberal filtering - keeps most documents while removing only the worst performers.

**Example**:
```bash
python run.py --input_file input.jsonl --model_name gpt2 \
              --output_file output.json --use_contexts \
              --ranking_ce --reranker OpenScholar/OpenScholar_Reranker \
              --use_score_threshold --score_threshold_type percentile_25
```

**Behavior**: Removes bottom 25% of documents based on OpenScholar scores.

---

### 4. 50th Percentile (`percentile_50`)
**Description**: Filters documents that score below the 50th percentile (keeps top 50%).

**Use Case**: Balanced filtering - equivalent to median threshold.

**Example**:
```bash
python run.py --input_file input.jsonl --model_name gpt2 \
              --output_file output.json --use_contexts \
              --ranking_ce --reranker OpenScholar/OpenScholar_Reranker \
              --use_score_threshold --score_threshold_type percentile_50
```

**Behavior**: Keeps top half of documents based on OpenScholar scores.

---

### 5. 75th Percentile (`percentile_75`)
**Description**: Filters documents that score below the 75th percentile (keeps top 25%).

**Use Case**: Aggressive filtering - only keeps the highest quality documents.

**Example**:
```bash
python run.py --input_file input.jsonl --model_name gpt2 \
              --output_file output.json --use_contexts \
              --ranking_ce --reranker OpenScholar/OpenScholar_Reranker \
              --use_score_threshold --score_threshold_type percentile_75
```

**Behavior**: Removes bottom 75% of documents, keeping only the top quartile.

---

### 6. 90th Percentile (`percentile_90`)
**Description**: Filters documents that score below the 90th percentile (keeps top 10%).

**Use Case**: Very aggressive filtering - only keeps the absolute best documents.

**Example**:
```bash
python run.py --input_file input.jsonl --model_name gpt2 \
              --output_file output.json --use_contexts \
              --ranking_ce --reranker OpenScholar/OpenScholar_Reranker \
              --use_score_threshold --score_threshold_type percentile_90
```

**Behavior**: Removes bottom 90% of documents, keeping only the top 10%.

## 🔍 Comparison: Score-Based vs Top-N Filtering

### Traditional Top-N Approach
```bash
# Always selects exactly 3 documents, regardless of quality
python run.py --input_file input.jsonl --top_n 3 --ranking_ce
```

### Score-Based Approach
```bash
# Adapts to quality distribution - could select 1, 2, 4, or more documents
python run.py --input_file input.jsonl --use_score_threshold --score_threshold_type average
```

### Example Comparison
Given OpenScholar scores: [6.5, 2.0, 1.4, 0.9]

| Approach | Documents Selected | Reasoning |
|----------|-------------------|-----------|
| `--top_n 3` | 3 documents | Fixed count regardless of quality |
| `--score_threshold_type average` | 2 documents | Threshold = 2.7, keeps [6.5, 2.0] |
| `--score_threshold_type median` | 3 documents | Threshold = 1.7, keeps [6.5, 2.0, 1.4] |
| `--score_threshold_type percentile_75` | 1 document | Threshold = 4.9, keeps [6.5] |

## 📈 Filter Statistics

When using score-based filtering, detailed statistics are automatically saved in the output:

```json
{
  "filter_stats": {
    "original_count": 4,
    "filtered_count": 2,
    "threshold": 2.675,
    "threshold_type": "average",
    "scores_above_threshold": [6.5352, 2.002],
    "scores_below_threshold": [1.387, 0.953]
  }
}
```

## 🚀 Quick Start Examples

### Conservative Filtering (Keep Most Documents)
```bash
python run.py --input_file input.jsonl --model_name gpt2 \
              --output_file output.json --use_contexts \
              --ranking_ce --reranker OpenScholar/OpenScholar_Reranker \
              --use_score_threshold --score_threshold_type percentile_25
```

### Balanced Filtering (Recommended)
```bash
python run.py --input_file input.jsonl --model_name gpt2 \
              --output_file output.json --use_contexts \
              --ranking_ce --reranker OpenScholar/OpenScholar_Reranker \
              --use_score_threshold --score_threshold_type average
```

### Aggressive Filtering (High Quality Only)
```bash
python run.py --input_file input.jsonl --model_name gpt2 \
              --output_file output.json --use_contexts \
              --ranking_ce --reranker OpenScholar/OpenScholar_Reranker \
              --use_score_threshold --score_threshold_type percentile_75
```

## 🎛️ Complete Command Reference

### Required Arguments
- `--use_score_threshold`: Enable score-based filtering
- `--ranking_ce`: Enable OpenScholar reranking (required for scores)
- `--reranker OpenScholar/OpenScholar_Reranker`: Specify the OpenScholar reranker model

### Optional Arguments
- `--score_threshold_type {average,median,percentile_25,percentile_50,percentile_75,percentile_90}`: Choose threshold type (default: average)

### Full Example with All Options
```bash
python run.py \
  --input_file test_input.jsonl \
  --output_file test_output.json \
  --model_name gpt2 \
  --use_contexts \
  --ranking_ce \
  --reranker OpenScholar/OpenScholar_Reranker \
  --use_score_threshold \
  --score_threshold_type average \
  --zero_shot \
  --max_tokens 2000
```

## 🔬 Why Use OpenScholar Reranker?

The **OpenScholar/OpenScholar_Reranker** is specifically trained for scientific literature and provides:

- ✅ **Better Relevance**: Higher scores for scientifically relevant content
- ✅ **Domain Expertise**: Understands scientific terminology and concepts  
- ✅ **Quality Control**: More selective filtering for higher-quality results
- ✅ **Optimized Performance**: Designed specifically for academic paper ranking

**Score Range Comparison**:
- **General BGE**: -10 to +2 (limited positive range)
- **OpenScholar**: -10 to +7 (broader positive range for high-quality content)

## 💡 Choosing the Right Threshold

| Scenario | Recommended Threshold | Reasoning |
|----------|----------------------|-----------|
| High-stakes applications (medical, legal) | `percentile_75` or `percentile_90` | Need highest quality documents only |
| General knowledge questions | `average` | Good balance of quality and coverage |
| Exploratory research | `percentile_25` or `median` | Want broader perspective |
| Limited context window | `percentile_75` | Maximize quality when space is limited |
| Comprehensive coverage needed | `percentile_25` | Include more diverse perspectives |

## 🔧 Advanced Usage

### Combining with Citation Filtering
```bash
python run.py --input_file input.jsonl \
              --use_score_threshold --score_threshold_type average \
              --min_citation 100 \
              --ranking_ce --reranker OpenScholar/OpenScholar_Reranker
```

### With Feedback Loop
```bash
python run.py --input_file input.jsonl \
              --use_score_threshold --score_threshold_type average \
              --feedback \
              --ranking_ce --reranker OpenScholar/OpenScholar_Reranker
```

### With Post-hoc Attribution
```bash
python run.py --input_file input.jsonl \
              --use_score_threshold --score_threshold_type average \
              --posthoc_at \
              --ranking_ce --reranker OpenScholar/OpenScholar_Reranker
```

### Complete Self-Reflective Pipeline
```bash
python run.py --input_file input.jsonl \
              --model_name OpenScholar/Llama-3.1_OpenScholar-8B \
              --use_contexts --output_file output.json \
              --llama3 --ranking_ce --reranker OpenScholar/OpenScholar_Reranker \
              --posthoc_at --feedback --ss_retriever \
              --use_abstract --norm_cite --zero_shot --max_per_paper 3 \
              --use_score_threshold --score_threshold_type average
```

## 🚨 Important Notes

1. **OpenScholar Reranker Required**: Score-based filtering requires `--ranking_ce` and `--reranker OpenScholar/OpenScholar_Reranker` for optimal scientific literature ranking.

2. **Consistent Filtering**: Score-based filtering is now applied consistently to both initial pre-indexed retrieval and feedback-retrieved documents during self-reflective generation, ensuring uniform quality standards.

3. **Minimum Documents**: The system ensures at least one document is always selected, even if all scores fall below the threshold.

4. **Performance**: Score-based filtering adds minimal computational overhead compared to reranking.

5. **Deterministic**: Results are deterministic for the same input and threshold type.

6. **Statistics**: Filter statistics are automatically included in the output for analysis.

## 🔄 Filtering in Self-Reflective Generation

When using the feedback loop (`--feedback`) with score-based filtering, the system now applies the same filtering logic to both phases:

### Phase 1: Initial Retrieval
```bash
Score-based filtering: 10 -> 4 docs (threshold: -5.234)
Using score-based filtering with 4 documents
```

### Phase 2: Feedback Retrieval
```bash
Feedback filtering: 8 -> 3 docs (threshold: -4.891)
```

This ensures that lower-quality documents retrieved during feedback don't compromise the final answer quality.

## 📊 Expected Behavior

Score-based filtering with OpenScholar reranker will typically:
- **Select fewer documents** when retrieval quality is poor
- **Select more documents** when retrieval quality is high  
- **Provide consistent quality** across different scientific queries
- **Adapt automatically** to score distributions
- **Improve answer quality** by excluding irrelevant content
- **Leverage scientific domain knowledge** for better relevance assessment

This adaptive approach with domain-specific reranking often leads to significantly better generation quality compared to fixed top-N selection with general-purpose rerankers. 