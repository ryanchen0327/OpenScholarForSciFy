# 🚀 OpenScholar v2.0.0 - Google Colab Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryanchen0327/OpenScholarForSciFy/blob/main/OpenScholar_Colab_Setup.ipynb)

## 🎯 Quick Setup (2 Options)

### Option 1: Jupyter Notebook (Recommended)
1. Click the "Open in Colab" badge above
2. Follow the step-by-step notebook
3. Run all cells in sequence

### Option 2: Python Script
Copy and paste this into a Colab cell:

```python
# Quick setup - run this in a single cell
!git clone https://github.com/ryanchen0327/OpenScholarForSciFy.git
%cd OpenScholarForSciFy
!pip install -r requirements.txt
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers accelerate sentence-transformers FlagEmbedding
!python -m spacy download en_core_web_sm

# Run the automated setup
!python colab_setup.py
```

## ⚡ One-Line Demo

Test OpenScholar immediately with this command:

```python
# Create a test question and run basic demo
import json
test_q = {"question": "What are the latest developments in transformer models?", "id": "test"}
with open("quick_test.jsonl", "w") as f:
    json.dump(test_q, f)

# Run OpenScholar
!python run.py --input_file quick_test.jsonl --model_name gpt2 --use_contexts --use_score_threshold --output_file quick_output.json --zero_shot

# View results
with open("quick_output.json", "r") as f:
    result = json.load(f)[0]
    print(f"Question: {result['question']}")
    print(f"Answer: {result['output']}")
    print(f"Sources: {len(result.get('ctxs', []))} documents")
```

## 🔑 API Keys (Optional)

For enhanced multi-source retrieval, set these in Colab:

```python
import os

# Semantic Scholar (can use dummy)
os.environ['S2_API_KEY'] = 'your_key_or_dummy'

# Google Custom Search (optional)
os.environ['GOOGLE_API_KEY'] = 'your_google_api_key'
os.environ['GOOGLE_CX'] = 'your_custom_search_engine_id'

# You.com Search (optional)  
os.environ['YOUR_API_KEY'] = 'your_youcom_api_key'
```

## 🚀 Enhanced Features Demo

Once setup is complete, try these advanced features:

### Multi-Source Feedback Retrieval
```python
!python run.py \
  --input_file quick_test.jsonl \
  --model_name gpt2 \
  --use_contexts --feedback \
  --ss_retriever \
  --use_score_threshold \
  --score_threshold_type average \
  --output_file enhanced_output.json \
  --zero_shot
```

### With All Sources (if APIs available)
```python
!python run.py \
  --input_file quick_test.jsonl \
  --model_name gpt2 \
  --use_contexts --feedback \
  --ss_retriever \
  --use_google_feedback \
  --use_youcom_feedback \
  --use_score_threshold \
  --score_threshold_type percentile_75 \
  --output_file full_demo.json \
  --zero_shot
```

## 📊 Key Features

### ✅ What Works in Colab
- **Basic RAG Pipeline**: ✅ Full support
- **Score-Based Filtering**: ✅ All threshold types
- **Semantic Scholar API**: ✅ With dummy or real key
- **Self-Reflective Generation**: ✅ Full feedback loop
- **Google/You.com Search**: ✅ With API keys
- **Adaptive Thresholds**: ✅ Automatic quality control

### 🎯 New in v2.0.0
- **Multi-Source Retrieval**: 4 data sources
- **Adaptive Filtering**: Stricter thresholds for more sources  
- **Source Attribution**: Know where each document came from
- **Enhanced Quality**: Consistent filtering throughout pipeline

## 🔧 Troubleshooting

### Common Issues

**"Model not found" Error:**
```python
# Use a smaller model for testing
!python run.py --input_file test.jsonl --model_name gpt2 --use_contexts --output_file output.json --zero_shot
```

**"CUDA out of memory":**
```python
# Reduce batch size or use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
```

**"API rate limit exceeded":**
```python
# Use dummy key for Semantic Scholar
os.environ['S2_API_KEY'] = 'dummy_key'
```

### Memory Optimization
```python
# Clear cache between runs
import torch
torch.cuda.empty_cache()

# Or restart runtime: Runtime > Restart runtime
```

## 📋 Example Notebooks

1. **Basic Usage**: Simple RAG with score filtering
2. **Multi-Source Demo**: All data sources with adaptive filtering  
3. **Custom Research**: Interactive question testing
4. **Comparative Analysis**: Different threshold comparisons

## 🎉 Quick Success Check

After setup, verify everything works:

```python
# Test all major features
commands = [
    "python run.py --input_file quick_test.jsonl --model_name gpt2 --use_contexts --output_file test1.json --zero_shot",
    "python run.py --input_file quick_test.jsonl --model_name gpt2 --use_contexts --use_score_threshold --output_file test2.json --zero_shot", 
    "python run.py --input_file quick_test.jsonl --model_name gpt2 --use_contexts --feedback --ss_retriever --output_file test3.json --zero_shot"
]

for i, cmd in enumerate(commands, 1):
    print(f"🧪 Test {i}: {'✅' if os.system(cmd) == 0 else '❌'}")
```

## 📚 Resources

- **📖 Full Documentation**: See README.md in repository
- **🔧 Multi-Source Guide**: MULTI_SOURCE_FEEDBACK_README.md  
- **📊 Filtering Guide**: SCORE_FILTERING_README.md
- **🔄 What's New**: CHANGELOG.md
- **⚖️ License**: LICENSE_COMPLIANCE.md

## 🆘 Need Help?

1. **Issues**: https://github.com/ryanchen0327/OpenScholarForSciFy/issues
2. **Discussions**: Check repository discussions
3. **Documentation**: All README files in the repository

---

**🎓 Happy researching with OpenScholar v2.0.0 in Google Colab!** 