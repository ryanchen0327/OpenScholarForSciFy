#!/usr/bin/env python3
"""
Demo script to show how OpenScholar's reranker scores documents
"""

from FlagEmbedding import FlagReranker
import json

def main():
    print("=== OpenScholar Reranker Scoring Demo ===\n")
    
    # Initialize the BGE reranker (same as used in OpenScholar)
    print("1. Loading BGE Reranker...")
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
    print("   ✅ Loaded BAAI/bge-reranker-base\n")
    
    # Sample query and documents (from our test)
    query = "What are the main advantages of retrieval-augmented generation over traditional language models?"
    
    documents = [
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "text": "We explore retrieval-augmented generation (RAG), which uses a pre-trained parametric memory (a seq2seq model) and a non-parametric memory (a dense vector index of Wikipedia) to generate responses. RAG models can generate more specific, diverse and factual responses than seq2seq models. For knowledge-intensive tasks, we achieve state-of-the-art results on three open-domain QA datasets.",
            "citation_counts": 1500
        },
        {
            "title": "FiD: Leveraging Passage Retrieval with Generative Models", 
            "text": "Fusion-in-Decoder (FiD) leverages the power of retrieval with the generation capabilities of large language models. By processing multiple retrieved passages independently in the encoder and fusing information in the decoder, FiD achieves strong performance on knowledge-intensive tasks while maintaining computational efficiency.",
            "citation_counts": 800
        },
        {
            "title": "REALM: Retrieval-Augmented Language Model Pre-Training",
            "text": "REALM pre-trains a language model as a dense retriever and a knowledge-augmented encoder. This approach allows the model to retrieve and attend over documents from a large corpus such as Wikipedia, improving performance on knowledge-intensive tasks. The retrieval component is learned end-to-end as part of the training process.",
            "citation_counts": 900
        },
        {
            "title": "RAG vs Parametric Models",
            "text": "Traditional parametric language models store knowledge in their weights, which can lead to hallucinations and outdated information. Retrieval-augmented models address these issues by accessing external knowledge bases during inference, providing more accurate and up-to-date information while maintaining the flexibility of neural generation.",
            "citation_counts": 200
        }
    ]
    
    print("2. Query:")
    print(f"   '{query}'\n")
    
    print("3. Documents to score:")
    for i, doc in enumerate(documents):
        print(f"   [{i}] {doc['title']}")
        print(f"       Citations: {doc['citation_counts']}")
        print(f"       Text: {doc['text'][:100]}...\n")
    
    # Step 1: Prepare texts for scoring (title + text)
    print("4. Preparing texts for reranker (title + text):")
    paragraph_texts = []
    for i, doc in enumerate(documents):
        combined_text = f"{doc['title']} {doc['text']}"
        paragraph_texts.append(combined_text)
        print(f"   [{i}] Length: {len(combined_text)} chars")
        print(f"       Preview: {combined_text[:120]}...\n")
    
    # Step 2: Create query-document pairs for scoring
    print("5. Creating query-document pairs for BGE reranker:")
    query_doc_pairs = [[query, text] for text in paragraph_texts]
    print(f"   Created {len(query_doc_pairs)} pairs: [query, document_text]\n")
    
    # Step 3: Get relevance scores from BGE reranker
    print("6. Computing BGE relevance scores...")
    scores = reranker.compute_score(query_doc_pairs, batch_size=100)
    print("   ✅ BGE scoring complete!\n")
    
    # Step 4: Show raw BGE scores
    print("7. Raw BGE Relevance Scores:")
    raw_results = {}
    for i, score in enumerate(scores):
        raw_results[i] = score
        print(f"   Document [{i}]: {score:.6f}")
        print(f"      Title: {documents[i]['title']}")
        print(f"      Why: {explain_score(score)}\n")
    
    # Step 5: Optional citation normalization (norm_cite=True)
    print("8. Citation-Normalized Scores (if enabled):")
    max_citations = max(doc['citation_counts'] for doc in documents)
    print(f"   Max citations: {max_citations}")
    
    normalized_results = {}
    for i, score in enumerate(scores):
        citation_boost = documents[i]['citation_counts'] / max_citations
        normalized_score = score + citation_boost
        normalized_results[i] = normalized_score
        print(f"   Document [{i}]: {score:.3f} + {citation_boost:.3f} = {normalized_score:.3f}")
        print(f"      Citations: {documents[i]['citation_counts']} / {max_citations}")
    print()
    
    # Step 6: Rank documents by score
    print("9. Final Ranking (highest to lowest):")
    ranked_docs = sorted(raw_results.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (doc_id, score) in enumerate(ranked_docs):
        print(f"   Rank {rank + 1}: Document [{doc_id}] - Score: {score:.6f}")
        print(f"             Title: {documents[doc_id]['title']}")
        print(f"             Relevance: {explain_relevance(rank + 1)}")
    print()
    
    # Step 7: Compare with our actual results
    print("10. Comparison with OpenScholar results:")
    actual_results = {"0": 1.38671875, "1": 0.95263671875, "2": 2.001953125, "3": 4.5625}
    print("    Actual ranking from our test:")
    actual_ranked = sorted(actual_results.items(), key=lambda x: x[1], reverse=True)
    for rank, (doc_id, score) in enumerate(actual_ranked):
        print(f"    Rank {rank + 1}: Doc [{doc_id}] = {score} ({documents[int(doc_id)]['title'][:30]}...)")

def explain_score(score):
    """Explain what a BGE score means"""
    if score > 3.0:
        return "Very High Relevance - Strong semantic match"
    elif score > 1.0:
        return "High Relevance - Good semantic match"  
    elif score > 0.0:
        return "Moderate Relevance - Some semantic overlap"
    else:
        return "Low Relevance - Little semantic connection"

def explain_relevance(rank):
    """Explain ranking position"""
    if rank == 1:
        return "Most relevant - Best answers the question"
    elif rank == 2:
        return "Second most relevant - Strong but not perfect match"
    elif rank == 3:
        return "Third most relevant - Moderate relevance"
    else:
        return "Lower relevance - Less directly related"

if __name__ == "__main__":
    main() 