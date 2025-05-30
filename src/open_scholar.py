from tqdm import tqdm
import os
import re
import spacy
from src.use_search_apis import search_paper_via_query, retrieve_pes2o_passages, search_google_non_restricted, search_youcom_non_restricted

import numpy as np
import os
from nltk import sent_tokenize
import vllm
import src.instructions as instructions
from FlagEmbedding import FlagReranker

nlp = spacy.load('en_core_web_sm')

# To compute API costs based on October 2023 pricing available at https://openai.com/ja-JP/api/pricing/
price_per_million = {"gpt-4o": 2.50, "gpt-4o-2024-08-06": 2.50, "gpt-4o-2024-05-13": 5.00, "gpt-4o-mini": 0.15, "gpt-4o-mini-2024-07-18": 0.15, "gpt-4-turbo": 10.0, "gpt-3.5-turbo-0125": 0.50} 
price_per_million_output = {"gpt-4o": 10.00, "gpt-4o-2024-08-06": 10.00,  "gpt-4o-2024-05-13": 15.00, "gpt-4o-mini": 0.600, "gpt-4o-mini-2024-07-18": 0.600, "gpt-4-turbo": 30.0, "gpt-3.5-turbo-0125": 1.50} 

def calculate_openai_api_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """
    Calculate OpenAI API cost based on the number of input and output tokens.
    
    Args:
    - input_tokens (int): Number of tokens in the input.
    - output_tokens (int): Estimated number of tokens in the output.
    - price_per_million_tokens (float): Cost per 1 million tokens (e.g., 0.02 for GPT-4).

    Returns:
    - float: The total API cost.
    """
    total_cost_input = (input_tokens / 1000000) * price_per_million[model_name]
    total_cost_output =  (output_tokens / 1000000) * price_per_million_output[model_name]
    total_cost = total_cost_input + total_cost_output
    return round(total_cost, 6)

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")
    
def rerank_paragraphs_bge(query, paragraphs, reranker, norm_cite=False, start_index=0, use_abstract=False):
    paragraphs = [p for p in paragraphs if p["text"] is not None]
    
    # Return empty results if no paragraphs to rerank
    if len(paragraphs) == 0:
        print("No paragraphs to rerank - returning empty results")
        return [], {}, {}
    
    if use_abstract is True:
        paragraph_texts = [p["title"] + "\n" + p["abstract"] + "\n" + p["text"] if "title" in p and "abstract" in p else p["text"] for p in paragraphs]
    else:
        paragraph_texts = [p["title"] + " " + p["text"] if "title" in p and p["title"] is not None else p["text"] for p in paragraphs]
    
    print(paragraph_texts[0])
    scores = reranker.compute_score([[query, p] for p in paragraph_texts], batch_size=100)
    if type(scores) is float:
        result_dic = {0: scores}
    else:
        result_dic = {p_id: score for p_id, score in enumerate(scores)}
    if norm_cite is True and len([item["citation_counts"] for item in paragraphs if "citation_counts" in item and item["citation_counts"] is not None]) > 0:
        # add normalized scores
        max_citations = max([item["citation_counts"] for item in paragraphs if "citation_counts" in item and item["citation_counts"] is not None])
        for p_id in result_dic:
            if "citation_counts" in paragraphs[p_id] and paragraphs[p_id]["citation_counts"] is not None:
                result_dic[p_id] = result_dic[p_id] + (paragraphs[p_id]["citation_counts"] / max_citations)
    p_ids = sorted(result_dic.items(), key=lambda x: x[1], reverse=True)
    new_orders = []
    id_mapping = {}
    for i, p_id in enumerate(p_ids):
        new_orders.append(paragraphs[p_id[0]])
        id_mapping[i] = int(p_id[0])
    return new_orders, result_dic, id_mapping

def create_prompt_with_llama3_format(prompt, system_message="You are a helpful AI assistant for scientific literature review. Please carefully follow user's instruction and help them to understand the most recent papers."):
    if system_message is not None:
        formatted_text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{0}<|eot_id|>".format(system_message)
    else:
        formatted_text = "<|begin_of_text|>"
    formatted_text += "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|>"
    formatted_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted_text

def load_hf_tokenizer(
        model_name_or_path,
        tokenizer_name_or_path=None,
        use_fast_tokenizer=True,
        padding_side="left",
        token=os.getenv("HF_TOKEN", None),
    ):
        from transformers import AutoTokenizer

        # Need to explicitly import the olmo tokenizer.
        if not tokenizer_name_or_path:
            tokenizer_name_or_path = model_name_or_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer, token=token)
        except:
            # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, token=token)
        # set padding side to left for batch generation
        tokenizer.padding_side = padding_side
        # set pad token to eos token if pad token is not set (as is the case for llama models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
   
class OpenScholar(object):
    def __init__(self, model, tokenizer, client=None, api_model_name=None, use_contexts=True, top_n=8, reranker=None, min_citation=None, norm_cite=False, ss_retriever=False, use_score_threshold=False, score_threshold_type="average", use_pes2o_feedback=False, use_google_feedback=False, use_youcom_feedback=False, feedback_threshold_type=None):
        self.model = model
        self.tokenizer = tokenizer
        self.client = client
        self.model_name = api_model_name
        self.top_n = top_n
        self.no_retrieval = not use_contexts
        self.reranker = reranker
        self.min_citation = min_citation
        self.norm_cite = norm_cite
        self.ss_retriever = ss_retriever
        self.use_contexts = use_contexts
        # New parameters for score-based filtering
        self.use_score_threshold = use_score_threshold
        self.score_threshold_type = score_threshold_type  # "average", "median", "percentile_X"
        # New parameters for multi-source feedback retrieval
        self.use_pes2o_feedback = use_pes2o_feedback
        self.use_google_feedback = use_google_feedback
        self.use_youcom_feedback = use_youcom_feedback
        # Stricter threshold for feedback documents (defaults to percentile_75 if multi-source enabled)
        self.feedback_threshold_type = feedback_threshold_type or self._get_default_feedback_threshold()

    def _get_default_feedback_threshold(self):
        """
        Determine default feedback threshold based on enabled sources.
        Uses stricter thresholds when multi-source feedback is enabled to handle higher volume.
        """
        # Count enabled feedback sources
        feedback_sources_enabled = sum([
            self.ss_retriever,
            self.use_pes2o_feedback, 
            self.use_google_feedback,
            self.use_youcom_feedback
        ])
        
        if feedback_sources_enabled >= 3:
            # Multi-source: Use very strict threshold (top 10%)
            return "percentile_90"
        elif feedback_sources_enabled == 2:
            # Dual-source: Use strict threshold (top 25%)
            return "percentile_75"
        elif feedback_sources_enabled == 1:
            # Single-source: Use moderate threshold (top 50%)
            return "percentile_50"
        else:
            # No feedback sources: Use same as initial threshold
            return self.score_threshold_type

    # Reranking: We rerank passages based on the LMs' predictions on how useful passages are.
    def process_ranking_results(self, result):
        ratings = {int(match.group(1)): int(match.group(2)) for match in re.finditer(r'\[(\d+)\] Rating: (\d)', result)}
        return ratings

    def reranking_passages_cross_encoder(self, item, batch_size=5, llama3_chat=False, task_name="default", use_abstract=False):
        
        if self.min_citation is not None:
            ctx_above_threshold = [p for p in item["ctxs"] if "citation_counts" in p and p["citation_counts"] >= self.min_citation]
            if len(ctx_above_threshold) > self.top_n:
                item["ctxs"] = ctx_above_threshold
                print("after filtering -- number of ctxs: {0}".format(len(item["ctxs"])))
                
        reranked_contexts, sorted_results, id_mapping = rerank_paragraphs_bge(item["input"], item["ctxs"], self.reranker, norm_cite=self.norm_cite, use_abstract=use_abstract)
        return reranked_contexts, sorted_results, id_mapping
    
    def reranking_passages_cross_encoder_supplemental(self, item, passages, batch_size=5, llama3_chat=False, task_name="default"):
        
        if self.min_citation is not None:
            ctx_above_threshold = [p for p in passages if "citation_counts" in p and p["citation_counts"] >= self.min_citation]
            if len(ctx_above_threshold) > self.top_n:
                passages = ctx_above_threshold
                print("after filtering -- number of ctxs: {0}".format(len(passages)))
                
        reranked_contexts, sorted_results, id_mapping = rerank_paragraphs_bge(item["input"], passages, self.reranker, norm_cite=False, start_index=len(item["ctxs"]))
        return reranked_contexts, sorted_results, id_mapping
    
    def filter_documents_by_score_threshold(self, reranked_contexts, sorted_results, threshold_type="average"):
        """
        Filter documents based on score thresholds instead of top-N.
        
        Args:
            reranked_contexts: List of reranked documents
            sorted_results: Dictionary of {doc_id: score}
            threshold_type: "average", "median", "percentile_X" (e.g., "percentile_75")
        
        Returns:
            filtered_contexts: Documents above the threshold
            threshold_value: The computed threshold value
            stats: Statistics about filtering
        """
        if not sorted_results or len(sorted_results) == 0:
            return reranked_contexts, 0.0, {"original_count": 0, "filtered_count": 0, "threshold": 0.0}
        
        scores = list(sorted_results.values())
        original_count = len(scores)
        
        # Calculate threshold based on type
        if threshold_type == "average":
            threshold_value = sum(scores) / len(scores)
        elif threshold_type == "median":
            import statistics
            threshold_value = statistics.median(scores)
        elif threshold_type.startswith("percentile_"):
            percentile = int(threshold_type.split("_")[1])
            import numpy as np
            threshold_value = np.percentile(scores, percentile)
        else:
            # Default to average
            threshold_value = sum(scores) / len(scores)
        
        # Filter documents above threshold
        filtered_contexts = []
        filtered_count = 0
        
        for i, (doc_id, score) in enumerate(sorted_results.items()):
            if score >= threshold_value:
                if i < len(reranked_contexts):  # Safety check
                    filtered_contexts.append(reranked_contexts[i])
                    filtered_count += 1
        
        # Ensure at least one document if available
        if filtered_count == 0 and len(reranked_contexts) > 0:
            filtered_contexts.append(reranked_contexts[0])  # Take the highest scoring one
            filtered_count = 1
        
        stats = {
            "original_count": original_count,
            "filtered_count": filtered_count,
            "threshold": threshold_value,
            "threshold_type": threshold_type,
            "scores_above_threshold": [score for score in scores if score >= threshold_value],
            "scores_below_threshold": [score for score in scores if score < threshold_value]
        }
        
        print(f"Score-based filtering: {original_count} -> {filtered_count} docs (threshold: {threshold_value:.3f})")
        print(f"Threshold type: {threshold_type}")
        print(f"Score range: {min(scores):.3f} to {max(scores):.3f}")
        
        return filtered_contexts, threshold_value, stats

    def retrieve_keywords(self, question):
        prompt = [instructions.keyword_extraction_prompt.format_map({"question": question})]
        
        if  self.client is not None:
            result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user",
                            "content": prompt[0]},
                    ],
                    temperature=0.9,
                    max_tokens=1000,
                )
            raw_output = result.choices[0].message.content
            outputs = raw_output
        
        else:
            sampling_params = vllm.SamplingParams(
                temperature=0.9,  # greedy decoding
                max_tokens=1000,
                stop_token_ids=[128009]
            )
            outputs = self.model.generate(prompt, sampling_params)
            outputs = [it.outputs[0].text for it in outputs][0]
        raw_output = [t.split("[Response_End]")[0]  for t  in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
 
        queries = raw_output.split(", ")[:3]
        queries = [query.replace("Search queries: " , "") for query in queries if len(query) > 0]
        return queries

    # Generation: Generate output based on query, passages
    def generate_response(self, item, max_tokens=3000, llama3_chat=False,  task_name="default", zero_shot=False):
        ranked_results = {}
        print("zero-shot?: {}".format(zero_shot))
        print(item["input"])
        
        if self.use_contexts is False:
            ctxs = []
            # support more task
            if task_name in instructions.task_instructions:
                if zero_shot is True:
                    input_query = instructions.task_instructions[task_name][0] + instructions.task_instructions[task_name][1] + item["input"]
                else:
                    demonstration = instructions.demonstrations[task_name]
                    input_query = instructions.task_instructions[task_name][0] + demonstration + instructions.task_instructions[task_name][1] + item["input"]
            elif task_name == "single_qa":
                input_query = instructions.generation_instance_prompts_w_references_single_paper_no_context.format_map({"input": item["input"]})
            else:
                # Fallback case for default or unknown task_name
                input_query = item["input"]
        else:
            ctxs = ""
            # Apply score-based filtering if enabled
            if self.use_score_threshold and 'ranked_results' in item and item.get('ranked_results'):
                # Use score-based filtering
                filtered_contexts, threshold_value, filter_stats = self.filter_documents_by_score_threshold(
                    item["ctxs"], item['ranked_results'], self.score_threshold_type
                )
                item["filter_stats"] = filter_stats  # Store stats for analysis
                docs_to_use = filtered_contexts
                print("Using score-based filtering with {} documents".format(len(docs_to_use)))
            else:
                # Use traditional top-N filtering
                docs_to_use = item["ctxs"][:self.top_n]
                print("Using traditional top-N filtering with {} documents".format(len(docs_to_use)))
            
            for doc_idx, doc in enumerate(docs_to_use):
                if "title" in doc and len(doc["title"]) > 0:
                    ctxs += "[{0}] Title: {1} Text: {2}\n".format(doc_idx, doc["title"], doc["text"])
                else:
                    ctxs += "[{0}] {1}\n".format(doc_idx,  doc["text"])
            item["final_passages"] = ctxs
            
            if task_name =="summarization":
                if zero_shot is True:
                    input_query = instructions.prompts_w_references_summarization_zero_shot.format_map({"context": ctxs, "input": item["input"]})
                else:
                    input_query = instructions.generation_instance_prompts_summarization.format_map({"context": ctxs, "input": item["input"]})
            elif task_name == "single_qa":
                if zero_shot is True:
                    input_query = instructions.generation_instance_prompts_w_references_single_paper_zero_shot.format_map({"context": ctxs, "input": item["input"]})
                else:
                    input_query = instructions.generation_instance_prompts_w_references_single_paper.format_map({"context": ctxs, "input": item["input"]})
            
            elif task_name in instructions.task_instructions:
                task_instruction = instructions.task_instructions[task_name][0]
                instance_header = instructions.task_instructions[task_name][1]
                if zero_shot is True:
                    input_query = "{0}\nReferences:\n{1}\n{2}{3}".format(task_instruction, ctxs, instance_header, item["input"])
                else:
                    demonstration = instructions.demonstrations[task_name]
                    input_query = "{0}{1}\nReferences:\n{2}\n{3}{4}".format(task_instruction, demonstration, ctxs, instance_header, item["input"])
                    
            else:
                if zero_shot is True:
                    input_query = instructions.generation_instance_prompts_w_references_zero_shot.format_map({"context": ctxs, "input": item["input"]})
                else:
                    input_query = instructions.generation_instance_prompts_w_references.format_map({"context": ctxs, "input": item["input"]})

        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)
            
        if self.client is not None: 
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user",
                        "content": input_query},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            raw_output = result.choices[0].message.content
            outputs = raw_output
            cost = calculate_openai_api_cost(len(input_query.split(" ")),len(raw_output.split(" ")), self.model_name)
        else:
            sampling_params = vllm.SamplingParams(
                temperature=0.7,  # greedy decoding
                max_tokens=max_tokens,
                stop_token_ids=[128009]
            )
            outputs = self.model.generate([input_query], sampling_params)
            outputs = [it.outputs[0].text for it in outputs][0]
            cost = 0
        raw_output = [t.split("[Response_End]")[0] for t in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs

        if "References:" in raw_output:
            raw_output = raw_output.split("References:")[0]
        item["output"] = raw_output
        return raw_output, ctxs, cost

    # Feedback: send feedback on model' predictions.
    def process_feedback(self, response):
        feedbacks_and_questions = re.findall(r'Feedback: (.*?)(?:Question: (.*?))?\n', response)
        ratings = [(feedback.strip(), question.strip() if question else "") for feedback, question in feedbacks_and_questions]
        return ratings

    def get_feedback(self, item, llama3_chat):
        input_query = instructions.feedback_example_instance_prompt.format_map({"question": item["input"], "passages": item["final_passages"], "answer": item["output"]})
        # TODO: check if the llama3 chat format is helpful or not. 
        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)
        
        if self.client is not None:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user",
                        "content": input_query},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            outputs = result.choices[0].message.content
            cost = calculate_openai_api_cost(len(input_query.split(" ")),len(outputs.split(" ")), self.model_name)
        else:
            sampling_params = vllm.SamplingParams(
                temperature=0.7,  # greedy decoding
                max_tokens=2000,
                stop_token_ids=[128009]
            )

            outputs = self.model.generate([input_query], sampling_params)
            outputs = [it.outputs[0].text for it in outputs][0]
            cost = 0
        raw_output = [t.split("[Response_End]")[0]  for t  in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
        feedbacks = self.process_feedback(raw_output)
        return feedbacks, cost

    def edit_with_feedback(self, item, feedback, max_tokens=3000, llama3_chat=False):
        input_query = instructions.editing_instance_prompt.format_map({"question": item["input"], "passages": item["final_passages"], "answer": item["output"], "feedback": feedback})
        
        # TODO: check if the llama3 chat format is helpful or not. 
        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)
        
        if self.client is not None: 
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user",
                        "content": input_query},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            raw_output = result.choices[0].message.content
            outputs = raw_output
            cost = calculate_openai_api_cost(len(input_query.split(" ")),len(outputs.split(" ")), self.model_name)
        else:
            sampling_params = vllm.SamplingParams(
                temperature=0.7,  # greedy decoding
                max_tokens=max_tokens,
                stop_token_ids=[128009]
            )
            outputs = self.model.generate([input_query], sampling_params)
            outputs = [it.outputs[0].text for it in outputs][0]
            cost = 0
        raw_output = [t.split("[Response_End]")[0]  for t  in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
        print("orig answer: {}".format( item["output"]))
        print("feedback: {}".format(feedback))
        print("updated answer: {}".format(raw_output))
        return raw_output, cost

    def edit_with_feedback_retrieval(self, item, feedback, passages, passage_start_index, max_tokens=2000, llama3_chat=False):
        processed_passages = ""
        for doc_idx, doc in enumerate(passages[:self.top_n]):
            if "title" in doc and len(doc["title"]) > 0:
                processed_passages += "[{0}] Title: {1} Text: {2}\n".format(passage_start_index+doc_idx, doc["title"], doc["text"])
            else:
                processed_passages += "[{0}] {1}\n".format(passage_start_index+doc_idx + len(item["ctxs"]), doc["text"])

        input_query = instructions.editing_with_retrieval_instance_prompt.format_map({"question": item["input"], "retrieved_passages": processed_passages, "answer": item["output"], "feedback": feedback})
        if llama3_chat is True:
            input_query = create_prompt_with_llama3_format(input_query)
                
        if self.client is not None: 
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user",
                        "content": input_query},
                ],
                temperature=0.7,
                max_tokens=3000,
            )
            raw_output = result.choices[0].message.content
            outputs = raw_output
            cost = calculate_openai_api_cost(len(input_query.split(" ")),len(outputs.split(" ")), self.model_name)
        else:
            sampling_params = vllm.SamplingParams(
                temperature=0.7,  # greedy decoding
                max_tokens=3000,
                stop_token_ids=[128009]
            )
            outputs = self.model.generate([input_query], sampling_params)
            outputs = [it.outputs[0].text for it in outputs][0]
            cost = 0
        raw_output = [t.split("[Response_End]")[0]  for t in outputs.split("[Response_Start]") if "[Response_End]" in t][0] if "[Response_End]" in outputs else outputs
        return raw_output, cost

    def insert_attributions_posthoc_paragraph(self, item, llama3_chat=False):
        text = item["output"]
        if "final_passages" in item:
            passages = item["final_passages"] 
        else:
            ctxs = item["ctxs"]
            passages = ""
            for idx, p in enumerate(ctxs):
                passages += "[{0}] {1}\n".format(idx, p)

        print(text)
        sentences = text.split("\n")
        print(sentences)
        # post process sentences 
        updated_sentences = []
        post_hoc_sentence = {}

        for s_index, statement in enumerate(sentences):
            if len(statement) < 10:
                if len(updated_sentences) > 0 and len(statement) > 0 and statement[0] == "[":
                    updated_sentences[-1] = updated_sentences[-1] + " " + statement
                else:
                    updated_sentences.append(statement)
            
            else:
                # cases where citations are included
                if "[" in statement or (s_index < len(sentences) - 1 and len(sentences[s_index+1]) > 0 and sentences[s_index+1][0] == "["):
                    updated_sentences.append(statement)
                else:
                    updated_sentences.append("[replace_{}]".format(s_index))
                    post_hoc_sentence["[replace_{}]".format(s_index)] = statement

        if len(post_hoc_sentence) > 0:
            print("{0} sentences require attributions, e..g, {1}".format(len(post_hoc_sentence), list(post_hoc_sentence.values())[0] ))
            prompts = []
            for s in list(post_hoc_sentence.values()):    
                input_query = instructions.posthoc_attributions_paragraph.format_map({"statement": s, "passages": passages})

                if llama3_chat is True:
                    input_query = create_prompt_with_llama3_format(input_query)
                
                prompts.append(input_query)
            
            if self.client is not None: 
                outputs = []
                for input_query in prompts:
                    result = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user",
                                "content": input_query},
                        ],
                        temperature=0.7,
                        max_tokens=2000,
                    )
                    raw_output = result.choices[0].message.content
                    outputs.append(raw_output)
            else:
                sampling_params = vllm.SamplingParams(
                    temperature=0.7,  # greedy decoding
                    max_tokens=2000,
                    stop_token_ids=[128009]
                )
                outputs = self.model.generate(prompts, sampling_params)
                outputs = [it.outputs[0].text for it in outputs]
            
            # Postprocess Output
            for output, sentence_key in zip(outputs, list(post_hoc_sentence.keys())):
                if len([t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t]) == 0:
                    post_hoc_sentence[sentence_key] = post_hoc_sentence[sentence_key]
                else:
                    processed_output = [t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t][0]
                    post_hoc_sentence[sentence_key] = processed_output
                
            final_processed_outputs = []
            for item in updated_sentences:
                if item in post_hoc_sentence:
                    final_processed_outputs.append(post_hoc_sentence[item])
                else:
                    final_processed_outputs.append(item)
            updated_sentences = final_processed_outputs
            
        return "\n".join(updated_sentences)
    
    def insert_attributions_posthoc(self, item, llama3_chat=False):
        text = item["output"]
        passages = item["final_passages"]

        sentences = sent_tokenize(text)
        # post process sentences 
        updated_sentences = []
        post_hoc_sentence = {}

        for s_index, statement in enumerate(sentences):
            if len(statement) < 10:
                if statement[0] == "[":
                    updated_sentences[-1]  = updated_sentences[-1] + " " + statement
                else:
                    updated_sentences.append(statement)
            
            else:
                # cases where citations are included
                if "[" in statement or (s_index < len(sentences) - 1 and sentences[s_index+1][0] =="["):
                    updated_sentences.append(statement)
                else:
                    updated_sentences.append("[replace_{}]".format(s_index))
                    post_hoc_sentence["[replace_{}]".format(s_index)] = statement

        if len(post_hoc_sentence) > 0:
                        
            print("{0} sentences require attributions, e..g, {1}".format(len(post_hoc_sentence), list(post_hoc_sentence.values())[0] ))
            prompts = []
            for s in list(post_hoc_sentence.values()):    
                input_query = instructions.posthoc_attributions.format_map({"statement": s, "passages": passages})

                if llama3_chat is True:
                    input_query = create_prompt_with_llama3_format(input_query)
                
                prompts.append(input_query)
            
            if self.client is not None: 
                outputs = []
                for input_query in prompts:
                    result = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user",
                                "content": input_query},
                        ],
                        temperature=0.7,
                        max_tokens=2000,
                    )
                    raw_output = result.choices[0].message.content
                    outputs.append(raw_output)
            else:
                sampling_params = vllm.SamplingParams(
                    temperature=0.7,  # greedy decoding
                    max_tokens=2000,
                    stop_token_ids=[128009]
                )
                outputs = self.model.generate(prompts, sampling_params)
                outputs = [it.outputs[0].text for it in outputs]
            
            # process_output
            for output, sentence_key in zip(outputs, list(post_hoc_sentence.keys())):
                if len([t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t]) == 0:
                    post_hoc_sentence[sentence_key] = post_hoc_sentence[sentence_key]
                else:
                    processed_output = [t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t][0]
                    post_hoc_sentence[sentence_key] = processed_output
                
            final_processed_outputs = []
            for item in updated_sentences:
                if item in post_hoc_sentence:
                    final_processed_outputs.append(post_hoc_sentence[item])
                else:
                    final_processed_outputs.append(item)
            updated_sentences = final_processed_outputs
            
        return " ".join(updated_sentences)

    def insert_attributions_posthoc_paragraph_all(self, item, llama3_chat=False):
        text = item["output"]
        if "final_passages" in item:
            passages = item["final_passages"] 
        else:
            ctxs = item["ctxs"]
            passages = ""
            for idx, p in enumerate(ctxs):
                passages += "[{0}] {1}\n".format(idx, p)

        sentences = text.split("\n")
        print(sentences)
        updated_sentences = []
        post_hoc_sentence = {}
        prompts = []

        for s_index, statement in enumerate(sentences):
            if len(statement) < 10:
                if len(updated_sentences) > 0 and len(statement) > 0 and statement[0] == "[":
                    updated_sentences[-1] = updated_sentences[-1] + " " + statement
                else:
                    updated_sentences.append(statement)
            
            else:
                updated_sentences.append("[replace_{}]".format(s_index))
                post_hoc_sentence["[replace_{}]".format(s_index)] = statement

        for s in list(post_hoc_sentence.values()):    
            input_query = instructions.posthoc_attributions_paragraph_all.format_map({"statement": s, "passages": passages})

            if llama3_chat is True:
                input_query = create_prompt_with_llama3_format(input_query)
            
            prompts.append(input_query)
        
        if self.client is not None: 
            outputs = []
            cost = 0
            for input_query in prompts:
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user",
                            "content": input_query},
                    ],
                    temperature=0.7,
                    max_tokens=1000,
                )
                raw_output = result.choices[0].message.content
                outputs.append(raw_output)
                cost += calculate_openai_api_cost(len(input_query.split(" ")),len(raw_output.split(" ")), self.model_name)
        else:
            sampling_params = vllm.SamplingParams(
                temperature=0.7,
                max_tokens=1000,
                stop_token_ids=[128009]
            )
            outputs = self.model.generate(prompts, sampling_params)
            outputs = [it.outputs[0].text for it in outputs]
            cost = 0
        
        # process_output
        for output, sentence_key in zip(outputs, list(post_hoc_sentence.keys())):
            if len([t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t]) == 0:
                post_hoc_sentence[sentence_key] = post_hoc_sentence[sentence_key]
            else:
                processed_output = [t.split("[Response_End]")[0] for t in output.split("[Response_Start]") if "[Response_End]" in t][0]
                post_hoc_sentence[sentence_key] = processed_output
            
        final_processed_outputs = []
        for item in updated_sentences:
            if item in post_hoc_sentence:
                final_processed_outputs.append(post_hoc_sentence[item])
            else:
                final_processed_outputs.append(item)
        updated_sentences = final_processed_outputs
        
        return "\n".join(updated_sentences), cost

    def run(self, item, ranking_ce=False, use_feedback=False, skip_generation=False, posthoc_at=False, llama3_chat=False, task_name="default", zero_shot=False, max_per_paper=None, use_abstract=False, max_tokens=3000):
        print("llama3 chat format? {0}".format(llama3_chat))
        print("use feedback: {}".format(use_feedback))
        total_cost = 0
            
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

        if max_per_paper is not None:
            filtered_ctxs = []
            title_to_count = {}
            for ctx in item["ctxs"]:
                if "title" not in ctx or ctx["title"] is None:
                    ctx["title"] = ""
                title_to_count.setdefault(ctx["title"], 0)
                if title_to_count[ctx["title"]] > max_per_paper:
                    # print("We have already aded the paper {0} {1} times".format(ctx["title"], max_per_paper))
                    continue
                else:
                    filtered_ctxs.append(ctx)
                    title_to_count[ctx["title"]] += 1
                    
            item["ctxs"] = filtered_ctxs
            
        if skip_generation is False:
            generated_result, passages, gen_cost = self.generate_response(item, max_tokens=max_tokens, llama3_chat=llama3_chat, task_name=task_name, zero_shot=zero_shot)
            if "\n\n References":
                generated_result = generated_result.split("\n\n References")[0]
            item["initial_result"] = generated_result
            total_cost += gen_cost

        if use_feedback is True:
            print("generating feedback")
            feedbacks, feedback_cost = self.get_feedback(item, llama3_chat=llama3_chat)[:3]
            total_cost += feedback_cost
            item["feedbacks"] = feedbacks
            for feedback_idx, feedback in tqdm(enumerate(feedbacks[:3])):
                # currently only supports non retrieval feedback
                if len(feedback[1]) == 0:
                    edited_answer, edited_cost = self.edit_with_feedback(item, feedback[0], llama3_chat=llama3_chat)
                    if "Here is the revised answer:\n\n" in edited_answer:
                        edited_answer = edited_answer.split("Here is the revised answer:\n\n")[1]
                    total_cost += edited_cost
                    if len(item["output"]) > 0 and len(edited_answer) / len(item["output"]) > 0.9:
                        item["output"] = edited_answer
                        item["edited_answer_{}".format(feedback_idx)] = edited_answer
                    else:
                        print("skipping as edited answers got too short")
                else:
                    new_papers = []
                    feedback_query = feedback[1][0] if len(feedback[1]) > 0 else ""
                    
                    # 1. Semantic Scholar API retrieval (existing)
                    if self.ss_retriever is True:
                        print("🔍 Retrieving from Semantic Scholar API...")
                        new_keywords = self.retrieve_keywords(feedback_query)
                        paper_list = {}
                        if len(new_keywords) > 0:
                            for keyword in new_keywords:    
                                top_papers = search_paper_via_query(keyword)
                                print(top_papers)
                                if top_papers is None:
                                    print(keyword)
                                else:
                                    for paper in top_papers:
                                        if paper["paperId"] not in paper_list:
                                            paper["text"] = paper["abstract"]
                                            paper["citation_counts"] = paper["citationCount"]
                                            paper["type"] = "semantic_scholar_feedback"
                                            paper_list[paper["paperId"]] = paper
                            new_papers += list(paper_list.values())
                            print(f"📄 Semantic Scholar: Retrieved {len(list(paper_list.values()))} papers")
                    
                    # 2. peS2o dense retrieval (new)
                    if self.use_pes2o_feedback is True:
                        print("🔍 Retrieving from peS2o dense index...")
                        try:
                            pes2o_papers = retrieve_pes2o_passages(feedback_query, 20, "pes2o")
                            for paper in pes2o_papers:
                                paper["type"] = "pes2o_feedback"
                            new_papers += pes2o_papers
                            print(f"📄 peS2o: Retrieved {len(pes2o_papers)} papers")
                        except Exception as e:
                            print(f"❌ peS2o retrieval failed: {e}")
                    
                    # 3. Google search (new)
                    if self.use_google_feedback is True:
                        print("🔍 Retrieving from Google search...")
                        try:
                            google_papers = search_google_non_restricted(feedback_query)
                            for paper in google_papers:
                                paper["type"] = "google_feedback"
                            new_papers += google_papers
                            print(f"📄 Google: Retrieved {len(google_papers)} papers")
                        except Exception as e:
                            print(f"❌ Google search failed: {e}")
                    
                    # 4. You.com search (new)
                    if self.use_youcom_feedback is True:
                        print("🔍 Retrieving from You.com search...")
                        try:
                            youcom_papers = search_youcom_non_restricted(feedback_query)
                            for paper in youcom_papers:
                                paper["type"] = "youcom_feedback"
                            new_papers += youcom_papers
                            print(f"📄 You.com: Retrieved {len(youcom_papers)} papers")
                        except Exception as e:
                            print(f"❌ You.com search failed: {e}")
                    
                    print(f"📊 Total feedback papers from all sources: {len(new_papers)}")
                    
                    if len(new_papers) > 0:
                        print("before deduplication: {}".format(len(new_papers)))
                        new_papers_dicts = {paper["text"][:100] + paper["title"]: paper for paper in new_papers if paper is not None and type(paper["text"]) is str}
                        new_papers = list(new_papers_dicts.values())
                        print("after deduplication: {}".format(len(new_papers)))
                        # add new papers when and only when we have the new papers. 
                        if len(new_papers) > 0:
                            new_passages_reranked, new_ranked_results, new_id_mapping = self.reranking_passages_cross_encoder_supplemental(item, new_papers, batch_size=10, llama3_chat=llama3_chat, task_name=task_name)
                            passages_start_index = len(item["ctxs"])

                            # Apply score-based filtering to feedback-retrieved documents if enabled
                            if self.use_score_threshold and new_ranked_results:
                                filtered_new_passages, threshold_value, filter_stats = self.filter_documents_by_score_threshold(
                                    new_passages_reranked, new_ranked_results, self.feedback_threshold_type
                                )
                                print(f"Feedback filtering ({self.feedback_threshold_type}): {len(new_passages_reranked)} -> {len(filtered_new_passages)} docs (threshold: {threshold_value:.3f})")
                                print(f"📊 Sources enabled: {sum([self.ss_retriever, self.use_pes2o_feedback, self.use_google_feedback, self.use_youcom_feedback])} → Using stricter threshold for feedback")
                                new_passages_to_add = filtered_new_passages
                            else:
                                # Use traditional top-N filtering for feedback documents
                                new_passages_to_add = new_passages_reranked[:self.top_n]
                                print(f"Feedback top-N filtering: {len(new_passages_reranked)} -> {len(new_passages_to_add)} docs")

                            edited_answer, edited_cost = self.edit_with_feedback_retrieval(item, feedback[0], new_passages_to_add, passages_start_index)
                            total_cost += edited_cost
                            if len(item["output"]) > 0 and len(edited_answer) / len(item["output"]) > 0.9:
                                item["ctxs"] += new_passages_to_add
                                item["edited_answer_{}".format(feedback_idx)] = edited_answer
                                item["output"] = edited_answer
                                item["edited_answer_{}".format(feedback_idx)] = edited_answer
                            elif len(item["output"]) == 0 and len(edited_answer) > 0:
                                item["ctxs"] += new_passages_to_add
                                item["edited_answer_{}".format(feedback_idx)] = edited_answer
                                item["output"] = edited_answer
                                item["edited_answer_{}".format(feedback_idx)] = edited_answer
                            else:
                                print("skipping as edited answers got too short")

        if posthoc_at is True:
            # attributed_results = self.insert_attributions_posthoc(item, llama3_chat=llama3_chat)
            # attributed_results = self.insert_attributions_posthoc_paragraph(item, llama3_chat=llama3_chat)
            attributed_results, attributed_cost =  self.insert_attributions_posthoc_paragraph_all(item, llama3_chat=llama3_chat)
            total_cost += attributed_cost
            item["output"] = attributed_results
        
        item["output"] = item["output"].replace("[Response_Start]", "").replace("[Response_End]", "")

        print(item["output"])

        if "\n### References" in item["output"]:
            item["output"] = item["output"].split("\n### References")[0]
        return item, total_cost

def process_paragraph(text):
    text = text.replace("<cit.>", "")
    text = remove_citations(text)
    return text

def process_input_data(data, use_contexts=True):
    processed_data = []
    for item in data:
        if "answer" not in item:
            item["answer"] = ""
        if "input" not in item:
            if "question" in item:
                item["input"] = item["question"]
            if "query" in item:
                item["input"] = item["query"]

        new_ctxs = []
        if use_contexts is True:
            # normalize ctx format for different retrieval APIs
            for ctx in item["ctxs"]:
                if type(ctx) is list:
                    for c in ctx:
                        if type(c) is dict:
                            new_ctxs.append(c)
                if type(ctx) is dict:
                    new_ctxs.append(ctx)
            item["ctxs"] = new_ctxs

            # remove duplicated contexts
            processed_paras = []
            for ctx in tqdm(item["ctxs"]):
                if "retrieval text" in ctx:
                    ctx["text"] = ctx["retrieval text"]
                if ctx["text"] is None or len(ctx["text"]) ==0:
                    continue
                if type(ctx["text"]) != str:
                    ctx["text"] = " ".join(ctx["text"]["contexts"])
                ctx["text"] = process_paragraph(ctx["text"])
                if "title" not in ctx:
                    ctx["title"] = ""
                processed_paras.append(ctx)

            processed_paras_dicts = {paper["text"][:100] + paper["title"]: paper for paper in processed_paras}
            processed_paras = list(processed_paras_dicts.values())

            item["ctxs"] = processed_paras
            item["original_ctxs"] = processed_paras
        processed_data.append(item)
    return processed_data