# Multi-Source Feedback Retrieval for OpenScholar

OpenScholar now supports retrieving documents from multiple data sources during the self-reflective generation feedback loop, expanding beyond just Semantic Scholar API to include dense retrieval and web search.

## 🎯 Overview

During the self-reflective generation process, when the model provides feedback that includes new questions or information needs, OpenScholar can now retrieve relevant documents from multiple sources:

1. **Semantic Scholar API** (existing) - Real-time academic paper search
2. **peS2o Dense Retrieval** (new) - Pre-indexed scientific literature  
3. **Google Search** (new) - Web-based academic content discovery
4. **You.com Search** (new) - Academic-focused web search

All retrieved documents are reranked using the OpenScholar reranker and filtered using score-based thresholds for consistent quality.

## 🚀 Available Data Sources

### 1. Semantic Scholar API (`--ss_retriever`)
**Description**: Real-time search of academic papers via Semantic Scholar API.
- **Pros**: Fresh content, comprehensive metadata, citation counts
- **Cons**: API rate limits, network dependency
- **Best for**: Recent papers, citation analysis

### 2. peS2o Dense Retrieval (`--use_pes2o_feedback`)
**Description**: Dense retrieval from pre-indexed peS2o scientific literature corpus.
- **Pros**: Fast retrieval, high-quality scientific content, offline capability
- **Cons**: Pre-indexed (not real-time), requires local index
- **Best for**: Comprehensive scientific literature coverage

### 3. Google Search (`--use_google_feedback`) 
**Description**: Academic content discovery via Google search with ArXiv/PubMed filtering.
- **Pros**: Broad coverage, finds diverse sources
- **Cons**: Requires API key, potential noise
- **Best for**: Finding papers across different platforms

### 4. You.com Search (`--use_youcom_feedback`)
**Description**: Academic-focused web search specifically targeting scientific sources.
- **Pros**: Academic-optimized, good ArXiv/PubMed integration
- **Cons**: Requires API key, newer service
- **Best for**: Academic content with web context

## 📋 Usage Examples

### Basic Multi-Source Feedback
```bash
python run.py \
  --input_file input.jsonl \
  --model_name OpenScholar/Llama-3.1_OpenScholar-8B \
  --use_contexts \
  --feedback \
  --ss_retriever \
  --use_pes2o_feedback \
  --reranker OpenScholar/OpenScholar_Reranker \
  --ranking_ce \
  --output_file output.json
```

### With Score-Based Filtering
```bash
python run.py \
  --input_file input.jsonl \
  --model_name OpenScholar/Llama-3.1_OpenScholar-8B \
  --use_contexts \
  --feedback \
  --ss_retriever \
  --use_pes2o_feedback \
  --use_google_feedback \
  --reranker OpenScholar/OpenScholar_Reranker \
  --ranking_ce \
  --use_score_threshold \
  --score_threshold_type average \
  --output_file output.json
```

### All Sources Enabled
```bash
python run.py \
  --input_file input.jsonl \
  --model_name OpenScholar/Llama-3.1_OpenScholar-8B \
  --use_contexts \
  --feedback \
  --ss_retriever \
  --use_pes2o_feedback \
  --use_google_feedback \
  --use_youcom_feedback \
  --reranker OpenScholar/OpenScholar_Reranker \
  --ranking_ce \
  --use_score_threshold \
  --score_threshold_type percentile_75 \
  --output_file output.json
```

## 🔧 Configuration Options

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--ss_retriever` | Enable Semantic Scholar API retrieval | False |
| `--use_pes2o_feedback` | Enable peS2o dense retrieval during feedback | False |
| `--use_google_feedback` | Enable Google search during feedback | False |
| `--use_youcom_feedback` | Enable You.com search during feedback | False |
| `--feedback_threshold_type` | Manual override for feedback threshold (auto-selected if not specified) | Auto-adaptive |

### API Requirements

- **Google Search**: Requires Google Custom Search API key
- **You.com Search**: Requires You.com API key  
- **Semantic Scholar**: Uses S2_API_KEY environment variable (can be dummy for basic usage)
- **peS2o**: Requires local peS2o index setup

## 📊 Processing Pipeline

1. **Feedback Analysis**: Extract questions/information needs from model feedback
2. **Multi-Source Retrieval**: Query enabled data sources in parallel
3. **Content Processing**: Normalize document format across sources
4. **Deduplication**: Remove duplicate content based on text similarity
5. **Reranking**: Apply OpenScholar reranker for relevance scoring
6. **Score Filtering**: Apply threshold-based filtering (if enabled)
7. **Integration**: Merge filtered documents with existing context

## 🎯 Best Practices

### Source Selection Strategy
- **For comprehensive coverage**: Enable all sources
- **For speed**: Use only peS2o + Semantic Scholar
- **For fresh content**: Prioritize Google + You.com + Semantic Scholar
- **For reliability**: Use peS2o + Semantic Scholar only

### Performance Optimization
- Use score-based filtering to maintain quality while expanding sources
- Consider API rate limits when enabling multiple web sources
- Monitor deduplication effectiveness with diverse sources

### Quality Assurance
- Always use `--ranking_ce` with `--reranker OpenScholar/OpenScholar_Reranker`
- Apply appropriate score thresholds (`average` or `percentile_75` recommended)
- Review document types in output to ensure source diversity

## 🔥 Adaptive Feedback Thresholds

OpenScholar automatically applies **stricter thresholds** for feedback-retrieved documents when multiple sources are enabled, ensuring quality doesn't degrade with volume:

### Automatic Threshold Selection

| Sources Enabled | Feedback Threshold | Selectivity | Reasoning |
|-----------------|-------------------|-------------|-----------|
| 1 source | `percentile_50` | ~50% kept | Moderate filtering for single source |
| 2 sources | `percentile_75` | ~25% kept | Strict filtering for dual sources |  
| 3+ sources | `percentile_90` | ~10% kept | Very strict filtering for multi-source |

### Example Behavior

```bash
# Single source: Moderate feedback filtering
--ss_retriever
# → Feedback threshold: percentile_50

# Dual source: Strict feedback filtering  
--ss_retriever --use_pes2o_feedback
# → Feedback threshold: percentile_75

# Multi-source: Very strict feedback filtering
--ss_retriever --use_pes2o_feedback --use_google_feedback
# → Feedback threshold: percentile_90
```

### Manual Override

You can override the automatic selection:

```bash
python run.py \
  --ss_retriever --use_pes2o_feedback --use_google_feedback \
  --use_score_threshold --score_threshold_type average \
  --feedback_threshold_type percentile_75  # Manual override
```

### Filtering Statistics Output

The system provides detailed logging:

```
📊 Sources enabled: 3 → Using stricter threshold for feedback
Feedback filtering (percentile_90): 55 → 5 docs (threshold: -2.156)
```

## 🛠️ Implementation Details

The multi-source feedback retrieval system:

1. **Maintains Source Attribution**: Each document includes a `type` field indicating its source
2. **Handles Failures Gracefully**: If one source fails, others continue working
3. **Preserves Existing Functionality**: All existing features remain unchanged
4. **Ensures Consistent Filtering**: Same reranking and filtering applied to all sources
5. **Provides Detailed Logging**: Shows retrieval statistics for each source

## 📝 Example Output Statistics
```
🔍 Retrieving from Semantic Scholar API...
📄 Semantic Scholar: Retrieved 15 papers
🔍 Retrieving from peS2o dense index...  
📄 peS2o: Retrieved 28 papers
🔍 Retrieving from Google search...
📄 Google: Retrieved 12 papers
📊 Total feedback papers from all sources: 55
before deduplication: 55
after deduplication: 42
Feedback filtering: 42 -> 8 docs (threshold: -6.245)
```

## 🚨 Requirements & Dependencies

- **OpenScholar reranker**: Required for consistent scoring across sources
- **API Keys**: Required for Google and You.com search (optional)
- **peS2o Index**: Required for dense retrieval (optional)
- **Network Access**: Required for web-based sources
- **Score-based filtering**: Recommended for multi-source scenarios

This enhancement significantly expands OpenScholar's ability to gather relevant information during self-reflective generation, leading to more comprehensive and well-sourced responses. 