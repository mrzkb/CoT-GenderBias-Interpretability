from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm 
# NOTE: tqdm.write() prints above a progress bar so as not to interupt the visualization
import argparse
import os
import ast
import re 
import csv, os, ast
import os.path as op

import torch.nn.functional as F  # you already import F later, keep it once

### LLAMA
#   /network/weights/llama.var/llama_3/Meta-Llama-3-8B-Instruct
#   /network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B-Instruct
#   /network/weights/llama.var/llama_3.3/Meta-Llama-3.3-70B-Instruct

def ensure_answer_cue(s: str) -> str:
    # Make sure the prompt ends with exactly "Answer: "
    s = s.rstrip("\n")  # DO NOT strip spaces!
    if re.search(r"Answer:\s*$", s, flags=re.IGNORECASE):
        return re.sub(r"(Answer:)\s*$", r"\1 ", s)
    return s + "\nAnswer: "

### Original log likelihood 
def compute_log_likelihood(prompt, answer, model, tokenizer):
    try:
        # Tokenize prompt
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        # Tokenize prompt + possible answer
        full_input = prompt + "\nAnswer: " + answer
        full_input_ids = tokenizer(full_input, return_tensors="pt").input_ids.to(model.device)

        # Create labels in order to mask the prompt tokens, only compute loss on the answer
        prompt_len = prompt_ids.shape[-1]
        labels = full_input_ids.clone()
        labels[:, :prompt_len] = -100 

        # Compute loss (negative log-likelihood)
        with torch.no_grad():
            loss = model(full_input_ids, labels=labels).loss.item() # pytorch returns average token loss by default

        # Approximate log-likelihood 
        log_likelihood = -loss * (full_input_ids.shape[-1] - prompt_len)

        return log_likelihood # returns single float

    except Exception as e:
        tqdm.write(f'Error: {str(e)}')
        return float('-inf') # Return very low likelihood on error

### Original get_answer_from likelihoods, no change
def get_answer_from_likelihoods(prompt, model, tokenizer, answer_choices=['0', '1', '2']):
    try:
        log_likelihoods = {} 
        for answer in answer_choices:
            log_likelihoods[answer] = compute_log_likelihood(prompt, answer, model, tokenizer)

        # Determine the predicted answer based on the highest probability
        predicted_answer = max(log_likelihoods, key=log_likelihoods.get) # compares the dictionary values, returns the corresponding key

        return {
            'predicted_answer': predicted_answer, # single string
            'max_log_likelihood': log_likelihoods[predicted_answer], # single float value
            'log_likelihoods': log_likelihoods # dictionary with string keys of MCQ index, float values of logli
        }
        
    except Exception as e:
        return {
            'predicted_answer': 'ERROR',
            'max_log_likelihood': float('-inf'),
            'log_likelihoods': {answer: float('-inf') for answer in answer_choices},
            'error': str(e)
        }


### Modified get_answers_for_df, not from HERO, but even older...
# sas_writer=None, max_len=768
def get_answers_for_df(df, model, tokenizer, answer_choices=['0', '1', '2'], sample_size=None, random_state=22, sas_writer=None, max_len=768):
    # Optional subsample
    if sample_size is not None:
        original_size = len(df)
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state).reset_index(drop=True)
        print(f"Sampling {len(df)} rows from {original_size} total rows")

    results_data = []

    start_time = time.time()

    # We need the full row (not just the prompt) to read bias_label / unknown_label / answer_texts
    for i, row in enumerate(tqdm(df.itertuples(index=False), desc="Processing prompts", total=len(df))):
        rd = row._asdict()
        prompt_text = rd["prompt"]
        example_id = rd["example_id"]

        # iter_start = time.time()

        # 1) Get likelihoods for each answer
        results = get_answer_from_likelihoods(prompt_text, model, tokenizer, answer_choices)
        pred_digit = results['predicted_answer']

        # Save results attached to example_id to ensure no indexing errors
        result_row = {
            'example_id': example_id,
            'predicted_answer': pred_digit, # string '0', '1', or '2'
            'max_log_likelihood': results['max_log_likelihood'] # float
        }
        for answer in answer_choices: # Add individual answer probabilities
            result_row[f"prob_{answer}"] = results['log_likelihoods'].get(answer, float('-inf'))
        results_data.append(result_row)


        # 2) Locate S/A token columns strictly inside the Answer Options block (prompt side)
        S_idx, A_idx, S_text, A_text = sa_from_indices(rd)
        S_cols, A_cols, prompt_len = locate_SA_in_options_block(tokenizer, prompt_text, S_idx, S_text, A_idx, A_text, max_len=max_len)

        if i < 3:
            tqdm.write(f"[{rd.get('example_id', i)}] S={S_idx}:{S_text}  A={A_idx}:{A_text}")
            tqdm.write(f"S_cols[:8]={S_cols[:8]}  A_cols[:8]={A_cols[:8]}  prompt_len={prompt_len}")


        # 3) Build full input with the chosen digit answer to get attentions of the actual decision pass
        ctx = ensure_answer_cue(prompt_text)           # ensures trailing "Answer: " (with one space)
        full_text = ctx + pred_digit 
  
        enc_prompt = tokenizer(ctx, return_tensors="pt",
                       add_special_tokens=True, truncation=True, max_length=max_len)
        enc_full   = tokenizer(full_text, return_tensors="pt",
                       add_special_tokens=True, truncation=True, max_length=max_len)

        input_ids_prompt = enc_prompt["input_ids"].to(model.device)
        inputs_full = {k: v.to(model.device) for k, v in enc_full.items()}

        labels = inputs_full["input_ids"].clone()
        labels[:, :input_ids_prompt.shape[-1]] = -100  # score only the answer segment

        # use THIS prompt_len for "i >= prompt_len" queries in SAS
        prompt_len_ctx = input_ids_prompt.shape[-1]
        with torch.no_grad():
            out = model(**inputs_full, labels=labels, output_attentions=True, use_cache=False, return_dict=True)
            sas_lh = sas_from_attn(out.attentions, prompt_len_ctx, S_cols, A_cols)

        # 4) Optionally write per-(layer, head) rows now (so you can aggregate later)
        if sas_writer is not None:
            # ex_id = rd.get("example_id", f"row_{i}")
            
            # Handle tuple return (when S_cols or A_cols is empty)
            if isinstance(sas_lh, tuple):
                tqdm.write(f"WARNING: Failed to locate S/A tokens for example_id={example_id}")
                sas_lh = sas_lh[0]  # Extract just the tensor from the tuple

            L, H = sas_lh.shape
            for l in range(L):
                for h in range(H):
                    sas_writer.writerow([example_id, l, h, float(sas_lh[l, h])])

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.2f} minutes ({total_time/len(df):.2f}s per prompt)")

    # Create results dataframe and merge on example_id
    out = df.copy()
    results_df = pd.DataFrame(results_data)
    out = out.merge(results_df, on='example_id', how='left')

    return out

def process_dataset(input_filename, model, tokenizer, model_name, sample_size):
    dataset_name = op.basename(input_filename).replace('prompts_', '').replace('.csv','').replace('_',' ').title()

    # Always write outputs into ./data
    os.makedirs('data', exist_ok=True)
    input_path = f'data/{input_filename}'

    if not op.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Generate timestamp in format: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    output_filename = input_filename.replace('prompts_', f'{model_name}_responses_{timestamp}_')
    output_path = f'data/{output_filename}'

    print("\n" + "=" * 60)
    print(f"Processing: {dataset_name}")
    print("=" * 60)

    df = pd.read_csv(input_path)

    # Ensure answer_texts are proper lists
    if isinstance(df.iloc[0]["answer_texts"], str):
        df["answer_texts"] = df["answer_texts"].apply(lambda s: ast.literal_eval(s))

    # Per-(layer, head) SAS rows for this dataset
    sas_rows_path = op.join("data", f"sas_rows_{model_name}_{op.splitext(input_filename)[0]}.csv")
    write_header = not op.exists(sas_rows_path)
    with open(sas_rows_path, "a", newline="") as sas_f:
        sas_writer = csv.writer(sas_f)
        if write_header:
            sas_writer.writerow(["example_id", "layer", "head", "sas"])

        responses = get_answers_for_df(
            df, model, tokenizer, sample_size=sample_size, sas_writer=sas_writer
        )

    responses.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}\n")


#BELOW are functions for attention 
def sa_from_indices(row):
    """
    Returns:
      S_idx: index of stereotypical option
      A_idx: index of anti-stereotypical option
      S_text, A_text: their texts
    """
    ans_texts = row['answer_texts']
    if isinstance(ans_texts, str):
        import ast
        ans_texts = ast.literal_eval(ans_texts)

    S_idx = int(row['bias_label'])
    U_idx = int(row['unknown_label'])
    A_idx = ({0, 1, 2} - {S_idx, U_idx}).pop()

    S_text, A_text = ans_texts[S_idx], ans_texts[A_idx]
    return S_idx, A_idx, S_text, A_text


def _char_span_to_token_idxs(char_start, char_end, offsets):
    idxs = []
    for ti, (s, e) in enumerate(offsets):
        # skip specials ie (0,0)
        if s is None or e is None or (s == 0 and e == 0):
            continue
        if max(s, char_start) < min(e, char_end):  # any overlap
            idxs.append(ti)
    return idxs

def locate_SA_in_options_block(tokenizer, prompt_text, S_idx, S_text, A_idx, A_text, max_len=768):
    """
    Returns: (S_token_indices, A_token_indices, prompt_token_length)
    Searches strictly between 'Answer Options:' and the next 'Answer:'.
    Tokenization here MUST match the forward pass (add_special_tokens, truncation, max_length).
    """
    m = re.search(r"answer\s*options:\s*(?P<body>.+?)\n\s*answer:",
                  prompt_text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        m = re.search(r"answer\s*options:\s*(?P<body>.+)$",
                      prompt_text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return [], [], 0

    block_start, block_end = m.start("body"), m.end("body")

    def opt_line(n, t): return f"{n}) {t}"
    S_line = opt_line(S_idx, S_text)
    A_line = opt_line(A_idx, A_text)

    def find_in_block(needle):
        mm = re.search(re.escape(needle), prompt_text[block_start:block_end])
        if not mm:
            # spacing fallback option 
            num, tail = needle.split(") ", 1)
            pat = rf"{re.escape(num)}\)\s+{re.escape(tail)}"
            mm = re.search(pat, prompt_text[block_start:block_end], flags=re.IGNORECASE)
            if not mm:
                return None
        return (block_start + mm.start(), block_start + mm.end())

    S_char = find_in_block(S_line)
    A_char = find_in_block(A_line)
    if not S_char or not A_char:
        return [], [], 0

    # IMPORTANT: same settings as encodings used for the forward pass
    enc = tokenizer(
        prompt_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len,
    )
    offsets = enc["offset_mapping"]
    prompt_len = len(enc["input_ids"])

    S_cols = _char_span_to_token_idxs(S_char[0], S_char[1], offsets)
    A_cols = _char_span_to_token_idxs(A_char[0], A_char[1], offsets)
    return S_cols, A_cols, prompt_len



EPS = 1e-30
def sas_from_attn(attentions, prompt_len, S_cols, A_cols):
    """
    attentions: list over layers, each [B,H,T,S] (self-attention weights)
    Uses answer rows (i >= prompt_len) as queries, sums attention to Stereotypical and Antistereotypical columns.
    Returns:
      sas_lh: [L,H] tensor (per-layer, per-head SAS for this prompt)
    """
    if not S_cols or not A_cols:
        L, H = len(attentions), attentions[0].shape[1]
        return torch.zeros(L, H, dtype=torch.float64), 0.0

    per_layer = []
    for layer_attn in attentions:                   # [B,H,T,S]
        A = layer_attn[0]                           # [H,T,S]
        ans_rows = A[:, prompt_len:, :]             # queries = answer tokens

        # Sum attention to stereotypical and anti-stereotypical columns
        A_stereo = ans_rows[:, :, S_cols].sum(-1)   # [H, T_ans] # Why -1?
        A_anti = ans_rows[:, :, A_cols].sum(-1)     # [H, T_ans]

        # SAS formula: (A_stereo + A_anti) * log(A_stereo / A_anti)
        term = (A_stereo + A_anti) * torch.log((A_stereo + EPS)/(A_anti + EPS))

        per_layer.append(term.sum(-1).double())     # Sum over token range, [H]
    
    sas = torch.stack(per_layer, dim=0) # [L,H]
    return sas  

def main():
    parser = argparse.ArgumentParser(description='Process prompts with LLM and compute log-likelihoods')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the model (local path or HuggingFace model ID)')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Short name for the model (used in output filenames)')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Test on a smaller subset (optional, uses full dataset if not provided)')                   

    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    print(f"Model name: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    dtype="auto",   # Was torch.float16 before           
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2"  
    )

    # Attention 02
    model.eval()
    model.config.return_dict = True
    # Do NOT set model.config.output_attentions globally, already passed per call.

    print("Model successfully accessed from cluster.")

    # Define all datasets to process
    datasets = [
        'prompts_gender_ambig_no_cot.csv', # Mira put prompts/ in front for each
        'prompts_gender_ambig_cot.csv',
        'prompts_gender_disambig_no_cot.csv',
        'prompts_gender_disambig_cot.csv'
    ]
    
    for dataset in datasets:
        process_dataset(
            input_filename=dataset,
            model=model,
            tokenizer=tokenizer,
            model_name=args.model_name,
            sample_size=args.sample_size
        )

    print("\nAll datasets processed, cheers!")
    return

if __name__ == '__main__':
    main()
