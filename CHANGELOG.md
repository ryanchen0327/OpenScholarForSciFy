# OpenScholar Enhancement Changelog

## Version 2.0.0 - Multi-Source Feedback Retrieval & Adaptive Filtering

### 🎯 Major Features Added

#### 1. Multi-Source Feedback Retrieval System
- **Enhanced feedback loop** to support multiple data sources during self-reflective generation
- **Four data sources** now available during feedback retrieval:
  - Semantic Scholar API (existing, enhanced)
  - peS2o Dense Retrieval (new)
  - Google Search (new) 
  - You.com Search (new)
- **Graceful error handling** - if one source fails, others continue working
- **Source attribution** - each document tagged with source type for transparency
- **Detailed logging** showing retrieval statistics per source

#### 2. Adaptive Feedback Threshold System
- **Intelligent threshold selection** that automatically becomes stricter as more sources are enabled
- **Quality preservation** - prevents degradation when document volume increases
- **Adaptive logic**:
  - 1 source: `percentile_50` (moderate filtering, ~50% kept)
  - 2 sources: `percentile_75` (strict filtering, ~25% kept)
  - 3+ sources: `percentile_90` (very strict filtering, ~10% kept)
- **Manual override capability** via `--feedback_threshold_type` parameter

#### 3. Enhanced Score-Based Filtering
- **Consistent filtering** applied to both initial and feedback-retrieved documents
- **Fixed inconsistency** where feedback documents bypassed score filtering
- **Improved quality control** throughout the entire pipeline

### 🔧 Technical Implementation

#### New Files Added
- `SCORE_FILTERING_README.md` - Comprehensive documentation for score-based filtering
- `MULTI_SOURCE_FEEDBACK_README.md` - Complete guide for multi-source feedback retrieval

#### Modified Files

**src/open_scholar.py**
- Added multi-source feedback retrieval support
- Implemented adaptive threshold selection logic (`_get_default_feedback_threshold()`)
- Enhanced feedback processing with multiple data sources
- Added consistent score-based filtering for feedback documents
- New parameters: `use_pes2o_feedback`, `use_google_feedback`, `use_youcom_feedback`, `feedback_threshold_type`

**run.py**
- Added command-line arguments for multi-source feedback configuration
- Added `--feedback_threshold_type` parameter for manual threshold override
- Updated OpenScholar initialization to support new parameters

**src/use_search_apis.py** (imports)
- Enhanced imports to support Google and You.com search functions

### 🚀 New Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_pes2o_feedback` | flag | False | Enable peS2o dense retrieval during feedback |
| `--use_google_feedback` | flag | False | Enable Google search during feedback |
| `--use_youcom_feedback` | flag | False | Enable You.com search during feedback |
| `--feedback_threshold_type` | choice | Auto-adaptive | Manual override for feedback threshold |

### 📊 Performance Improvements

#### Quality Assurance
- **Stricter filtering** for high-volume multi-source scenarios
- **Consistent quality standards** across initial and feedback retrieval
- **Source diversity** while maintaining relevance thresholds

#### Processing Efficiency
- **Parallel retrieval** from multiple sources
- **Intelligent deduplication** across diverse content sources
- **Fault-tolerant design** with graceful degradation

### 🎯 Usage Examples

#### Basic Multi-Source Setup
```bash
python run.py \
  --input_file input.jsonl \
  --model_name OpenScholar/Llama-3.1_OpenScholar-8B \
  --use_contexts --feedback \
  --ss_retriever --use_pes2o_feedback \
  --reranker OpenScholar/OpenScholar_Reranker \
  --ranking_ce --output_file output.json
```

#### All Sources with Adaptive Filtering
```bash
python run.py \
  --input_file input.jsonl \
  --model_name OpenScholar/Llama-3.1_OpenScholar-8B \
  --use_contexts --feedback \
  --ss_retriever --use_pes2o_feedback \
  --use_google_feedback --use_youcom_feedback \
  --reranker OpenScholar/OpenScholar_Reranker \
  --ranking_ce --use_score_threshold \
  --score_threshold_type average \
  --output_file output.json
```

### 🔍 Backward Compatibility

- **Fully backward compatible** - all existing functionality preserved
- **Optional features** - new sources disabled by default
- **Existing workflows** continue to work without modification
- **Progressive enhancement** - users can adopt features incrementally

### 📋 Testing & Validation

#### Features Tested
- ✅ Single-source feedback retrieval (Semantic Scholar)
- ✅ Multi-source feedback retrieval (SS + peS2o + Google)
- ✅ Adaptive threshold selection logic
- ✅ Score-based filtering consistency
- ✅ Error handling and graceful degradation
- ✅ Source attribution and logging

#### Performance Metrics
- **Quality**: Stricter filtering maintains relevance standards
- **Coverage**: Multi-source retrieval increases document diversity
- **Efficiency**: Adaptive thresholds optimize context usage
- **Reliability**: Fault-tolerant design ensures system stability

### 🚨 Requirements & Dependencies

#### New Dependencies
- No new package dependencies required
- Existing FlagEmbedding and OpenScholar reranker dependencies

#### API Requirements (Optional)
- Google Custom Search API key (for Google search)
- You.com API key (for You.com search)
- Semantic Scholar API key (existing, can be dummy for basic usage)
- peS2o index setup (for dense retrieval)

### 📖 Documentation

#### New Documentation Files
- **SCORE_FILTERING_README.md**: Complete guide to score-based filtering
- **MULTI_SOURCE_FEEDBACK_README.md**: Comprehensive multi-source setup guide
- **CHANGELOG.md**: This detailed change documentation

#### Updated Documentation
- **README.md**: Enhanced with new features and examples
- Inline code comments for new functionality
- Command-line help text updates

### 🎉 Benefits Summary

1. **Enhanced Coverage**: Access to diverse scientific literature sources
2. **Quality Assurance**: Adaptive filtering maintains standards at scale
3. **Fault Tolerance**: Robust system with graceful error handling
4. **Transparency**: Clear source attribution and detailed logging
5. **Flexibility**: Configurable source selection and threshold override
6. **Efficiency**: Intelligent filtering optimizes context window usage

### 🔄 Migration Guide

#### For Existing Users
1. No changes required for basic usage
2. Add new flags to enable enhanced features
3. Review new documentation for advanced configurations

#### For Advanced Users
1. Experiment with multi-source combinations
2. Tune threshold settings for specific use cases
3. Monitor filtering statistics for optimal performance

---

**License**: Apache 2.0  
**Compatibility**: OpenScholar v1.x compatible  
**Python Version**: 3.8+  
**Date**: December 2024 