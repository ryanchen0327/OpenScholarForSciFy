import jsonlines
import json
import argparse
import re 
from run_utils import load_jsonlines, save_file_jsonl
from tqdm import tqdm

def process_references(output, ctxs):
    citations = extract_citations(output)
    used_ctxs = [(i, c) for i, c in enumerate(ctxs) if i in citations]
    reference_text = ""
    reference_list = []
    for c in tqdm(used_ctxs):
        # if "title" in c and len(c["title"]) > 0:
        #     paper_data = check_paper_details(c["title"])
        #     if paper_data is not None:
        #         reference_text += "[{0}] *Title: {1}* ([link to paper]({2})) {3}\n".format(c[0], c[1]["title"], paper_data["url"], c[1]["text"])
        #     else:
        #         reference_text += "[{0}] *Title: {1}* {2}\n".format(c[0], c[1]["title"], c[1]["text"])
        # else:
        #     reference_text += "[{0}] {1}\n\n".format(c[0], c[1]["text"])
        reference_list.append({"title": c[1]["title"] if "title" in c[1] else "", "text": c[1]["text"], "id": c[0], "url": c[1]["url"] if "url" in c[1] else ""})
    return reference_list

def extract_citations(text):
    # Regular expression to match [number] or [number_1, number_2, number_3]
    citation_pattern = r'\[(\d+(?:,\s*\d+)*)\]'
    # Find all matches in the text
    matches = re.findall(citation_pattern, text)
    # Extract individual numbers and convert them to integers
    citations = []
    for match in matches:
        # Split by commas, strip any extra whitespace, and convert to integers
        citations.extend([int(num.strip()) for num in match.split(',')])
    return citations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True, type=str)
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()
    if args.pred_file.endswith(".json"):
        preds = json.load(open(args.pred_file))["data"]
    else:
        preds = load_jsonlines(args.pred_file)
    data = json.load(open(args.data_file))
    q2id = {item["initial_prompt"]: item["case_id"] for item in data}
    final_data = []
    for pred, instance in zip(preds, data):
        question = pred["input"] if "input" in pred else pred["query"]
        case_id = q2id[question]
        answer = pred["output"]
        if type(pred["ctxs"]) is dict:
            ctxs = []
            for ctx in pred["ctxs"].keys():
                ctxs.append({"title": ctx[1], "text": ctx[0]})
            pred["ctxs"] = ctxs
        references = process_references(answer, pred["ctxs"])
        reference_text = "\nReferences:\n"
        for id, ref in enumerate(references):
            reference_text += "[{0}] {1} {2}\n\n".format(id, ref["title"], ref["text"])
        answer += reference_text
        
        final_data.append({"case_id": case_id, "answer_text": answer})
        
    save_file_jsonl(final_data, args.output_file)
    
    
if __name__ == '__main__':
    main()
