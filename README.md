# OpenScholar

This repository contains the implementation of OpenScholar, a retrieval-augmented language model designed for scientific literature synthesis.

[**Blog**](https://allenai.org/blog/openscholar) | [**Demo**](https://open-scholar.allen.ai/) |
[**Paper**](https://arxiv.org/abs/2411.14199) | [**Model checkpoints and data**](https://huggingface.co/collections/OpenScholar/openscholar-v1-67376a89f6a80f448da411a6) | [**ScholarQABench**](https://github.com/AkariAsai/ScholarQABench/) | [**Expert Evaluation**](https://github.com/AkariAsai/OpenScholar_ExpertEval) | 
[**Slides**](https://akariasai.github.io/assets/pdf/open_scholar_slides.pdf) 
 
### Table of Contents
1. [Overview of OpenScholar](#overview-of-openscholar)
2. [Repository Organization](#repository-organization)
3. [Installation](#installation)
4. [Running OpenScholar](#running-openscholar-inference)
5. [Training OpenScholar-8B](#training)
6. [Running Retriever](#running-retriever)
7. [Configuration Options](#configuration-options)
8. [Citation](#citation)

## Overview of OpenScholar

Scientific progress depends on researchers' ability to synthesize the growing body of literature. However, the exponential growth of scientific publications—with millions of papers published annually—has made it increasingly challenging for scientists to find relevant information and stay current with developments in their fields.

**OpenScholar** is a retrieval-augmented language model designed to address these challenges by searching for relevant papers in the literature and generating responses grounded in those sources. The system combines dense retrieval, reranking, and iterative self-feedback to provide accurate, citation-backed answers to scientific queries.

![Overview of OpenScholar](imgs/open_scholar.png)

## 🚀 Enhanced Features (v2.0.0)

### Multi-Source Feedback Retrieval
OpenScholar supports multiple data sources during self-reflective generation, significantly expanding retrieval coverage:

- **Semantic Scholar API** - Real-time academic paper search
- **peS2o Dense Retrieval** - Pre-indexed scientific literature
- **Google Search** - Web-based academic content discovery  
- **You.com Search** - Academic-focused web search

### Adaptive Score-Based Filtering
Intelligent threshold selection that automatically adjusts filtering strictness based on the number of enabled sources:
- **1 source**: moderate filtering (~50% documents retained)
- **2 sources**: strict filtering (~25% documents retained)
- **3+ sources**: very strict filtering (~10% documents retained)

### Enhanced Quality Control
- **Consistent filtering** across initial and feedback retrieval
- **Source attribution** for transparency
- **Fault-tolerant design** with graceful error handling

📖 **See [MULTI_SOURCE_FEEDBACK_README.md](MULTI_SOURCE_FEEDBACK_README.md) and [SCORE_FILTERING_README.md](SCORE_FILTERING_README.md) for detailed guides.**

## Repository Organization

This repository contains the core implementation for OpenScholar inference:

- [`src/`](src): Main source code for OpenScholar
- [`training/`](training): Training code for Llama 3.1 8B using processed data (modified version of `torchtune`)
- [`retriever/`](retriever): Code for offline retrieval and hosting retrieval servers

For evaluation and benchmarking:
- **ScholarQABench**: [ScholarQABench repository](https://github.com/AkariAsai/ScholarQABench/)
- **Human Evaluation**: [OpenScholar_ExpertEval repository](https://github.com/AkariAsai/OpenScholar_ExpertEval)

## Installation 

To run OpenScholar inference, ensure all necessary dependencies are installed:

```bash
conda create -n os_env python=3.10.0
conda activate os_env
pip install -r requirements.txt
python -m spacy download en_core_web_sm
``` 

### API Key Configuration

Set the required API keys:

```bash
export S2_API_KEY=YOUR_S2_API_KEY
```
Obtain API keys from the [Semantic Scholar API Page](https://www.semanticscholar.org/product/api).

For web search functionality, configure You.com API:
```bash
export YOUR_API_KEY=YOUR_YOU_COM_API_KEY
```

For detailed information on training and retriever components, refer to the [`training/`](training/) and [`retriever/`](retriever) directories.

## Running OpenScholar Inference

OpenScholar combines offline retrieval results with real-time API-based retrieval from Semantic Scholar and web search APIs. Offline retrieval results are available at [Google Drive](https://drive.google.com/drive/folders/1lOloYPOveKesD-37lD4Dlju96tc0XIm9?usp=sharing).

### Using Open Language Models

#### Standard RAG Pipeline
```bash
python run.py \
    --input_file YOUR_INPUT_FILE \
    --model_name OpenScholar/Llama-3.1_OpenScholar-8B \
    --use_contexts \
    --output_file OUTPUT_FILE_PATH \
    --top_n 10 --llama3 --zero_shot
```

#### Retriever + Reranker Pipeline
```bash
python run.py \
    --input_file YOUR_INPUT_FILE \
    --model_name OpenScholar/Llama-3.1_OpenScholar-8B \
    --use_contexts \
    --ranking_ce \
    --reranker OpenScholar/OpenScholar_Reranker \
    --output_file OUTPUT_FILE_PATH \
    --top_n 10 --llama3 --zero_shot
```

#### Score-Based Document Filtering
```bash
python run.py \
    --input_file YOUR_INPUT_FILE \
    --model_name OpenScholar/Llama-3.1_OpenScholar-8B \
    --use_contexts \
    --ranking_ce \
    --reranker OpenScholar/OpenScholar_Reranker \
    --use_score_threshold \
    --score_threshold_type average \
    --output_file OUTPUT_FILE_PATH \
    --llama3 --zero_shot
```

#### Self-Reflective Generation Pipeline
```bash
python run.py \
    --input_file YOUR_INPUT_FILE \
    --model_name OpenScholar/Llama-3.1_OpenScholar-8B \
    --use_contexts --output_file OUTPUT_FILE_NAME \
    --top_n 10 --llama3 --use_contexts \
    --ranking_ce --reranker OpenScholar/OpenScholar_Reranker \
    --posthoc --feedback --ss_retriever \
    --use_abstract --norm_cite --zero_shot --max_per_paper 3
```

#### Multi-Source Feedback Retrieval
```bash
python run.py \
    --input_file YOUR_INPUT_FILE \
    --model_name OpenScholar/Llama-3.1_OpenScholar-8B \
    --use_contexts --feedback \
    --ss_retriever --use_pes2o_feedback --use_google_feedback \
    --ranking_ce --reranker OpenScholar/OpenScholar_Reranker \
    --use_score_threshold --score_threshold_type average \
    --output_file OUTPUT_FILE_NAME \
    --llama3 --zero_shot
```

#### Complete Pipeline with All Sources
```bash
python run.py \
    --input_file YOUR_INPUT_FILE \
    --model_name OpenScholar/Llama-3.1_OpenScholar-8B \
    --use_contexts --feedback \
    --ss_retriever --use_pes2o_feedback \
    --use_google_feedback --use_youcom_feedback \
    --ranking_ce --reranker OpenScholar/OpenScholar_Reranker \
    --use_score_threshold --score_threshold_type average \
    --posthoc_at --use_abstract --norm_cite \
    --output_file OUTPUT_FILE_NAME \
    --llama3 --zero_shot
```

### Using Proprietary Language Models

OpenScholar can be combined with proprietary LLMs such as OpenAI GPT-4o:

```bash
python run.py \
    --input_file YOUR_INPUT_FILE \
    --model_name "gpt-4o" \
    --api "openai" \
    --api_key_fp PATH_TO_YOUR_OPENAI_KEY \
    --use_contexts \
    --output_file OUTPUT_FILE_PATH \
    --top_n 10 --llama3 --zero_shot
```

## Configuration Options

### Core Parameters
- `top_n`: Number of passages fed to the language model (default: 10 for multi-paper tasks)
- `use_score_threshold`: Enable adaptive score-based document filtering instead of fixed top-N
- `score_threshold_type`: Threshold type for score-based filtering (`average`, `median`, `percentile_25`, `percentile_50`, `percentile_75`, `percentile_90`)
- `feedback`: Enable self-feedback loop during generation
- `posthoc_at`: Enable post-hoc citation attribution
- `zero_shot`: Run inference in zero-shot manner

### Retrieval and Reranking
- `ranking_ce`: Use reranking model for passage reranking
- `reranker`: Path to reranker model (use `OpenScholar/OpenScholar_Reranker` for OpenScholar reranker)
- `min_citation`: Minimum citation count threshold for paper inclusion
- `ss_retriever`: Use Semantic Scholar API during feedback generation

### Multi-Source Retrieval (New)
- `use_pes2o_feedback`: Enable peS2o dense retrieval during feedback loop
- `use_google_feedback`: Enable Google search during feedback loop
- `use_youcom_feedback`: Enable You.com search during feedback loop
- `feedback_threshold_type`: Manual override for adaptive feedback threshold selection

### Additional Options
- `use_abstract`: Include abstracts to enhance reranking
- `max_per_paper`: Maximum number of passages per paper during inference
- `task_name`: Task specification (`claim_full` for SciFact, `boolean_question_full` for PubmedQA, `single_qa` for QASA)

## Training

[OpenScholar-8B](https://huggingface.co/OpenScholar/OpenScholar_Llama-3.1-8B) is trained using the [OpenScholar Training Dataset](https://huggingface.co/datasets/OpenScholar/OS_Train_Data), consisting of 13K instruction-tuning examples. Training uses a modified version of torchtune on 8×A100 GPUs.

See detailed training instructions in the [`training/`](training) directory.

## Running Retriever

Both peS2o v2 and v3 datastores (chunked text + index) are available:
- [OpenScholar-DataStore-V2](https://huggingface.co/datasets/OpenScholar/OpenScholar-DataStore-V2)
- [OpenScholar-DataStore-V3](https://huggingface.co/datasets/OpenScholar/OpenScholar-DataStore-V3)

See instructions in the [`retriever/`](retriever) directory for local deployment. Note that the peS2o index requires substantial CPU memory (200M+ embeddings from 45M papers).

## Citation

If you use OpenScholar in your research, please cite:

```bibtex
@article{openscholar,
  title={{OpenScholar}: Synthesizing Scientific Literature with Retrieval-Augmented Language Models},
  author={Asai, Akari and He*, Jacqueline and Shao*, Rulin and Shi, Weijia and Singh, Amanpreet and Chang, Joseph Chee  and Lo,  Kyle and Soldaini, Luca and Feldman, Tian, Sergey and Mike, D'arcy and Wadden, David and Latzke, Matt and Minyang and Ji, Pan and Liu, Shengyan and Tong, Hao and Wu, Bohao and Xiong, Yanyu and Zettlemoyer, Luke and Weld, Dan and Neubig, Graham and Downey, Doug and Yih, Wen-tau and Koh, Pang Wei and Hajishirzi, Hannaneh},
  journal={Arxiv},
  year={2024},
}
```
