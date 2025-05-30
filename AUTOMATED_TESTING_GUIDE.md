# OpenScholar v2.0.0 Automated Testing Guide

This guide outlines all automated testing capabilities available in OpenScholar without requiring human annotators.

## 🧪 Available Automated Tests

### 1. SciFact (Binary Claim Verification)
- **Task**: Verify scientific claims as true/false
- **Metrics**: 
  - Accuracy (string matching)
  - Citation F1 score
- **Sample Size**: 209 test cases
- **Gold Standard**: Expert-labeled claims with supporting papers

### 2. PubMedQA (Medical Yes/No Questions)
- **Task**: Answer medical questions with yes/no/maybe
- **Metrics**:
  - Accuracy (string matching)
  - Citation F1 score  
- **Sample Size**: ~1000 test cases
- **Gold Standard**: Biomedical expert annotations

### 3. QASA (Long-form Question Answering)
- **Task**: Generate comprehensive answers to scientific questions
- **Metrics**:
  - ROUGE-L score (content overlap)
  - Citation accuracy
- **Sample Size**: ~2000 test cases
- **Gold Standard**: Expert-written long-form answers

### 4. Citation Correctness Evaluation
- **Automated validation of**:
  - Citation format compliance
  - Reference-text alignment
  - Hallucination detection
- **Works across all tasks**

## ⚙️ Test Configuration Matrix

The automated testing suite evaluates 5 different configurations:

### 1. Baseline RAG
- Standard retrieval-augmented generation
- GPT-4o + basic document retrieval
- **Baseline for comparison**

### 2. Reranker Pipeline  
- Adds OpenScholar reranker
- Improves document relevance
- **Focus**: Better source quality

### 3. Score-Based Filtering
- Adaptive document filtering
- Quality-based thresholds
- **Focus**: Precision over quantity

### 4. Self-Reflective Generation
- Iterative feedback loop
- Semantic Scholar API integration
- **Focus**: Iterative improvement

### 5. Multi-Source Feedback
- All available retrieval sources:
  - Semantic Scholar API
  - peS2o dense retrieval
  - Google Search
  - You.com Search
- Adaptive threshold selection
- **Focus**: Maximum coverage

## 📊 Automated Metrics

### Primary Metrics
1. **Accuracy**: Exact match for classification tasks
2. **ROUGE-L**: Content overlap for long-form answers
3. **Citation F1**: Precision/recall of citation accuracy
4. **Citation Accuracy**: Percentage of valid citations

### Secondary Metrics
1. **Runtime**: Execution time per configuration
2. **Response Length**: Average character count
3. **Citations per Response**: Average citation density
4. **Success Rate**: Configuration reliability

### Performance Metrics
1. **Throughput**: Questions per minute
2. **Latency**: Time to first response
3. **Resource Usage**: Memory and API calls
4. **Error Rate**: Failed vs successful runs

## 🚀 Running the Tests

### Google Colab (Recommended)
```python
# Open the notebook: OpenScholar_Colab_Automated_Testing.ipynb
# Follow the step-by-step instructions
# All dependencies and setup automated
```

### Local Execution
```bash
# Setup
git clone https://github.com/ryanchen0327/OpenScholarForSciFy.git
cd OpenScholarForSciFy
pip install -r requirements.txt

# Run individual tests
python run.py \
  --input_file test_data.jsonl \
  --model_name gpt-4o \
  --api openai \
  --api_key_fp openai_key.txt \
  --use_contexts \
  --ranking_ce \
  --reranker OpenScholar/OpenScholar_Reranker \
  --use_score_threshold \
  --feedback \
  --ss_retriever \
  --output_file results.json

# Evaluate results
python ScholarQABench/scripts/citation_correctness_eval.py \
  --f results.json \
  --citations_long
```

## 📈 Evaluation Pipeline

### Step 1: Data Preparation
- Load test datasets (SciFact, PubMedQA, QASA)
- Sample representative test cases
- Format input for OpenScholar

### Step 2: Model Execution
- Run all 5 configurations
- Capture outputs and metadata
- Track performance metrics

### Step 3: Automated Evaluation
- Apply ScholarQABench evaluation scripts
- Calculate accuracy, ROUGE-L, citation metrics
- Generate performance statistics

### Step 4: Results Analysis
- Compare configurations across metrics
- Identify best performing setups
- Generate recommendations

## 🎯 Expected Results

### SciFact Performance
- **Baseline RAG**: ~75% accuracy
- **With Reranker**: ~80% accuracy  
- **Score Filtering**: ~82% accuracy
- **Self-Reflection**: ~85% accuracy
- **Multi-Source**: ~87% accuracy

### Citation Quality
- **Baseline**: ~60% citation accuracy
- **Enhanced Configs**: ~75-85% citation accuracy
- **Multi-Source**: ~90% citation accuracy

### Runtime Performance
- **Baseline**: ~30s per question
- **Reranker**: ~45s per question
- **Score Filtering**: ~40s per question
- **Self-Reflection**: ~90s per question
- **Multi-Source**: ~120s per question

## 🔧 Configuration Recommendations

### For Production Deployment
- **High Accuracy**: Multi-Source Feedback
- **Balanced**: Score-Based Filtering  
- **Fast Response**: Baseline RAG

### For Research Applications
- **Citation Analysis**: Self-Reflective Generation
- **Comparative Studies**: All configurations
- **Ablation Studies**: Individual feature toggles

### For Resource-Constrained Environments
- **Minimal Setup**: Baseline RAG
- **Quality Boost**: Reranker Pipeline
- **Adaptive**: Score-Based Filtering

## 📋 Test Report Template

Each automated test run generates:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "configurations": ["baseline_rag", "reranker_pipeline", ...],
  "datasets": ["scifact", "pubmed", "qasa"],
  "results": {
    "configuration_name": {
      "dataset_name": {
        "accuracy": 0.85,
        "citation_f1": 0.78,
        "rouge_l": 0.72,
        "runtime": 45.2,
        "success": true
      }
    }
  },
  "summary": {
    "best_overall": "multi_source_feedback",
    "fastest": "baseline_rag", 
    "most_accurate": "multi_source_feedback"
  }
}
```

## 🚨 Limitations and Considerations

### What's NOT Automated
- **Human evaluation** (organization, relevance, coverage)
- **Domain expert validation** 
- **Subjective quality assessment**
- **Real-world usage patterns**

### Automated Metrics Limitations
- **ROUGE-L**: Measures overlap, not semantic quality
- **Citation F1**: Format-based, not content verification
- **Accuracy**: Binary matching, not nuanced understanding

### Recommended Complementary Testing
- **Expert evaluation** for 10-20% of outputs
- **A/B testing** in production environments
- **User feedback** collection
- **Domain-specific validation**

## 🔄 Continuous Testing

### Automated CI/CD Integration
```yaml
# GitHub Actions example
name: OpenScholar Automated Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run automated tests
        run: python automated_test_suite.py
      - name: Upload results
        uses: actions/upload-artifact@v2
```

### Regular Monitoring
- **Weekly full test runs**
- **Daily smoke tests** 
- **Performance regression detection**
- **Quality metric tracking**

This automated testing framework provides comprehensive evaluation of OpenScholar's capabilities without human intervention, enabling rapid iteration and reliable performance measurement across all enhanced features. 