import os
import glob
import json
import re
import csv
import string
from collections import defaultdict
from itertools import combinations

# --- Configuration ---
CQS_DIR = "cqs"

# Single Bridge Config
SINGLE_OUTPUTS_DIR = "kg-sel-outputs-single-bridge"
SINGLE_ROW_CSV = "kg_sel_single_bridge_row_results.csv"
SINGLE_SUMMARY_CSV = "kg_sel_single_bridge_summary.csv"

# Multi Bridge Config
MULTI_OUTPUTS_DIR = "kg-sel-outputs-mul-bridge"
MULTI_ROW_CSV = "kg_sel_multi_bridge_row_results.csv"
MULTI_SUMMARY_CSV = "kg_sel_multi_bridge_summary.csv"

# Files to explicitly ignore
IGNORED_FILES = {
    "core_scholar_shallow.txt", 
    "enslaved_wiki.txt", 
    "gbo.txt", 
    "kwg_lite.txt"
}

# Mapping shorthand from filenames to full KG names expected in outputs
KG_FILENAME_MAPPING = {
    "kwg": "kwg",
    "currkg": "currkg",
    "enslaved": "enslaved",
    "gmo": "gmo",
    "core_scholar": "core_scholar_rich"
}

NO_KG_PHRASE = "no kg can be selected based on the provided schema contexts"

def normalize_text(text):
    text = text.lower().strip()
    return text.translate(str.maketrans('', '', string.punctuation))

def parse_kg_set(text):
    """Returns a SET of KGs for proper mathematical intersection/difference."""
    norm_text = normalize_text(text)
    if NO_KG_PHRASE in norm_text:
        return set() # Empty set represents "no_kg"
    
    parts = [p.strip() for p in text.lower().split(',')]
    parts = [p.translate(str.maketrans('', '', string.punctuation)) for p in parts if p]
    return set(parts)

# --- 1. Load All CQs ---
print(f"Loading questions from '{CQS_DIR}'...")
question_to_meta = {}

for filepath in glob.glob(os.path.join(CQS_DIR, '*.txt')):
    filename = os.path.basename(filepath)
    if filename in IGNORED_FILES:
        continue
    
    if "3bridge" in filename:
        num_bridges = 3
    elif "2bridge" in filename:
        num_bridges = 2
    else:
        num_bridges = 1

    if num_bridges > 1:
        found_kgs = [full_kg for shorthand, full_kg in KG_FILENAME_MAPPING.items() if shorthand in filename]
        expected_raw = ", ".join(found_kgs) if found_kgs else filename.replace(".txt", "")
    else:
        expected_raw = filename.replace(".txt", "")
        
    expected_set = parse_kg_set(expected_raw)
    expected_str = "consolidated_kgs" if expected_raw == "consolidated_kgs" else ", ".join(sorted(expected_set))

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        for idx, question in enumerate(lines, start=1):
            q_norm = normalize_text(question)
            
            complexity = "unknown"
            if num_bridges == 1:
                if idx <= 5: complexity = "simple"
                elif idx <= 10: complexity = "moderate"
                else: complexity = "complex"

            question_to_meta[q_norm] = {
                "expected_set": expected_set,
                "expected_str": expected_str,
                "num_bridges": num_bridges,
                "cq_complexity": complexity
            }

# --- 2. Process Single Bridge ---
def process_single_bridge():
    print(f"\nProcessing SINGLE bridge files in '{SINGLE_OUTPUTS_DIR}'...")
    rows = []
    
    for filepath in glob.glob(os.path.join(SINGLE_OUTPUTS_DIR, '*.jsonl')):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip(): continue
                data = json.loads(line)
                
                if "candidates" in data.get("response", {}):
                    llm_response_raw = data["response"]["candidates"][0]["content"]["parts"][0]["text"].strip()
                    got_set = parse_kg_set(llm_response_raw)
                    got_str = ", ".join(sorted(got_set)) if got_set else "no_kg"
                    
                    request_key = data.get("key", "")
                    match = re.search(r'-([^-]+)-temp([\d\.]+)-(.*)', request_key)
                    if match:
                        q_raw = match.group(3).strip()
                        q_norm = normalize_text(q_raw)
                        meta = question_to_meta.get(q_norm)
                        
                        if meta and meta["num_bridges"] == 1:
                            expected_str = meta["expected_str"]
                            
                            # Classification Logic
                            tp = fp = tn = fn = 0
                            
                            if got_str == expected_str:
                                tp = 1
                            elif got_str == "no_kg" and expected_str == "consolidated_kgs":
                                tn = 1
                            else:
                                if got_str != "no_kg": fp = 1 
                                if expected_str != "consolidated_kgs": fn = 1
                            
                            rows.append({
                                "source_file": os.path.basename(filepath),
                                "line": line_num,
                                "question": q_raw,
                                "kg": expected_str,
                                "representation": request_key.split('-')[1] if len(request_key.split('-')) > 1 else "Unknown",
                                "prompt_type": match.group(1).strip(),
                                "temperature_label": f"temp{match.group(2).strip()}",
                                "cq_complexity": meta["cq_complexity"],
                                "expected": expected_str,
                                "got": got_str,
                                "TP": tp, "FP": fp, "TN": tn, "FN": fn
                            })

    # Output Single Row-Level Results
    if rows:
        row_fields = ["source_file", "line", "question", "kg", "representation", "prompt_type", "temperature_label", "cq_complexity", "expected", "got", "TP", "FP", "TN", "FN"]
        with open(SINGLE_ROW_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row_fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved Single-Bridge Row Results to {SINGLE_ROW_CSV}")
                            
    # Aggregate Single Summaries
    dimensions_list = ["kg", "representation", "prompt_type", "temperature_label", "cq_complexity"]
    levels = [("__".join(dims), list(dims)) for size in range(1, len(dimensions_list) + 1) for dims in combinations(dimensions_list, size)]
    
    summaries = []
    for level_name, dims in levels:
        grouped = defaultdict(list)
        for r in rows: grouped[tuple(r.get(d, "") for d in dims)].append(r)
            
        for group_key, group_rows in grouped.items():
            t_tp, t_fp, t_tn, t_fn = sum(r["TP"] for r in group_rows), sum(r["FP"] for r in group_rows), sum(r["TN"] for r in group_rows), sum(r["FN"] for r in group_rows)
            acc = (t_tp + t_tn) / len(group_rows) if len(group_rows) > 0 else 0
            prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
            rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            summary = {
                "level": level_name, "dimensions": "|".join(dims), "n_records": len(group_rows), 
                "TP": t_tp, "FP": t_fp, "TN": t_tn, "FN": t_fn, 
                "Accuracy": round(acc, 4), "Precision": round(prec, 4), "Recall": round(rec, 4), "F1": round(f1, 4)
            }
            summary.update(dict(zip(dims, group_key)))
            summaries.append(summary)

    if summaries:
        summary_fields = ["level", "dimensions", "kg", "representation", "prompt_type", "temperature_label", "cq_complexity", "n_records", "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1"]
        with open(SINGLE_SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Saved Single-Bridge Summary to {SINGLE_SUMMARY_CSV}")

# --- 3. Process Multi Bridge ---
def process_multi_bridge():
    print(f"\nProcessing MULTI bridge files in '{MULTI_OUTPUTS_DIR}'...")
    rows = []
    
    for filepath in glob.glob(os.path.join(MULTI_OUTPUTS_DIR, '*.jsonl')):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip(): continue
                data = json.loads(line)
                
                if "candidates" in data.get("response", {}):
                    llm_response_raw = data["response"]["candidates"][0]["content"]["parts"][0]["text"].strip()
                    got_set = parse_kg_set(llm_response_raw)
                    
                    request_key = data.get("key", "")
                    match = re.search(r'-([^-]+)-temp([\d\.]+)-(.*)', request_key)
                    if match:
                        q_raw = match.group(3).strip()
                        q_norm = normalize_text(q_raw)
                        meta = question_to_meta.get(q_norm)
                        
                        if meta and meta["num_bridges"] > 1:
                            exp_set = meta["expected_set"]
                            
                            # Multi-Label Set Math
                            matches_set = got_set.intersection(exp_set)
                            misses_set = exp_set.difference(got_set)
                            spurious_set = got_set.difference(exp_set)
                            
                            exact_match = 1 if (got_set == exp_set) else 0
                            complete_mismatch = 1 if (len(matches_set) == 0) else 0
                            partial_match = 1 if (len(matches_set) > 0 and not exact_match) else 0
                            
                            precision = len(matches_set) / len(got_set) if got_set else 0.0
                            fdr = len(spurious_set) / len(got_set) if got_set else 0.0
                            recall = len(matches_set) / len(exp_set) if exp_set else 0.0
                            miss_rate = len(misses_set) / len(exp_set) if exp_set else 0.0
                            
                            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                            
                            rows.append({
                                "source_file": os.path.basename(filepath),
                                "line": line_num,
                                "question": q_raw,
                                "kg": meta["expected_str"],
                                "representation": request_key.split('-')[1] if len(request_key.split('-')) > 1 else "Unknown",
                                "prompt_type": match.group(1).strip(),
                                "temperature_label": f"temp{match.group(2).strip()}",
                                "num_bridges": meta["num_bridges"],
                                "expected": meta["expected_str"],
                                "got": ", ".join(sorted(got_set)) if got_set else "no_kg",
                                "exact_match": exact_match,
                                "partial_match": partial_match,
                                "complete_mismatch": complete_mismatch,
                                "precision_retrieval_rate": round(precision, 4),
                                "incorrect_fdr_rate": round(fdr, 4),
                                "recall_hit_rate": round(recall, 4),
                                "miss_rate": round(miss_rate, 4),
                                "f1_score": round(f1_score, 4)
                            })

    # Output Multi Row-Level Results
    if rows:
        row_fields = ["source_file", "line", "question", "kg", "representation", "prompt_type", "temperature_label", "num_bridges", "expected", "got", "exact_match", "partial_match", "complete_mismatch", "precision_retrieval_rate", "incorrect_fdr_rate", "recall_hit_rate", "miss_rate", "f1_score"]
        with open(MULTI_ROW_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row_fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved Multi-Bridge Row Results to {MULTI_ROW_CSV}")
                            
    # Aggregate Multi Summaries
    dimensions_list = ["kg", "representation", "prompt_type", "temperature_label", "num_bridges"]
    levels = [("__".join(dims), list(dims)) for size in range(1, len(dimensions_list) + 1) for dims in combinations(dimensions_list, size)]
    
    summaries = []
    for level_name, dims in levels:
        grouped = defaultdict(list)
        for r in rows: grouped[tuple(r.get(d, "") for d in dims)].append(r)
            
        for group_key, group_rows in grouped.items():
            total = len(group_rows)
            avg_exact = sum(r["exact_match"] for r in group_rows) / total if total else 0
            avg_partial = sum(r["partial_match"] for r in group_rows) / total if total else 0
            avg_mismatch = sum(r["complete_mismatch"] for r in group_rows) / total if total else 0
            
            avg_precision = sum(r["precision_retrieval_rate"] for r in group_rows) / total if total else 0
            avg_recall = sum(r["recall_hit_rate"] for r in group_rows) / total if total else 0
            avg_f1 = sum(r["f1_score"] for r in group_rows) / total if total else 0
            
            summary = {
                "level": level_name, "dimensions": "|".join(dims), "n_records": total, 
                "Exact_Match_Rate": round(avg_exact, 4), 
                "Partial_Match_Rate": round(avg_partial, 4), 
                "Complete_Mismatch_Rate": round(avg_mismatch, 4),
                "Avg_Precision (Successful Retrieval)": round(avg_precision, 4),
                "Avg_Recall (Target Hit Rate)": round(avg_recall, 4),
                "Avg_F1_Score": round(avg_f1, 4)
            }
            summary.update(dict(zip(dims, group_key)))
            summaries.append(summary)

    if summaries:
        summary_fields = ["level", "dimensions", "kg", "representation", "prompt_type", "temperature_label", "num_bridges", "n_records", "Exact_Match_Rate", "Partial_Match_Rate", "Complete_Mismatch_Rate", "Avg_Precision (Successful Retrieval)", "Avg_Recall (Target Hit Rate)", "Avg_F1_Score"]
        with open(MULTI_SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Saved Multi-Bridge Summary to {MULTI_SUMMARY_CSV}")

# --- Execution ---
process_single_bridge()
process_multi_bridge()
print("\nEvaluation Complete.")