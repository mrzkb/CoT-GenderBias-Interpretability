from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import time
from tqdm import tqdm 
# NOTE: tqdm.write() prints above a progress bar so as not to interupt the visualization
import argparse
import os
import ast
import re 
import csv, os, ast
import os.path as op

### LLAMA
#   /network/weights/llama.var/llama_3/Meta-Llama-3-8B-Instruct
#   /network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B-Instruct
#   /network/weights/llama.var/llama_3.3/Meta-Llama-3.3-70B-Instruct

'''
def compute_log_likelihood(prompt, answer, model, tokenizer, max_len=768):
    prompt2 = prompt 
    enc_prompt = tokenizer(prompt2, return_tensors="pt",
                           add_special_tokens=True, truncation=True, max_length=max_len)
    enc_answer = tokenizer(answer, return_tensors="pt",
                           add_special_tokens=False)  # important: no specials here

    input_ids = torch.cat([enc_prompt.input_ids, enc_answer.input_ids], dim=1).to(model.device)
    prompt_len = enc_prompt.input_ids.shape[-1]
    ans_len = enc_answer.input_ids.shape[-1]

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100             # mask prompt and "Answer: "
    # (the last ans_len tokens remain unmasked)

    with torch.no_grad():
        out = model(input_ids, labels=labels)
        loss = out.loss.item()

    # avg loss * number of answer tokens (only ans_len contribute)
    return -loss * ans_len
'''

import re
import torch.nn.functional as F  # you already import F later, keep it once

def ensure_answer_cue(s: str) -> str:
    # Make sure the prompt ends with exactly "Answer: "
    s = s.rstrip("\n")  # DO NOT strip spaces!
    if re.search(r"Answer:\s*$", s, flags=re.IGNORECASE):
        return re.sub(r"(Answer:)\s*$", r"\1 ", s)
    return s + "\nAnswer: "

import torch
import torch.nn.functional as F

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
'''
@torch.no_grad()
@torch.no_grad()
def compute_log_likelihood(prompt, answer, model, tokenizer, max_len=768, debug=True, _printer=print):
    """
    Constrained next-token scorer:
      - Builds the context with a normalized 'Answer: ' cue.
      - Runs a single forward on the context.
      - Maps the candidate `answer` (e.g., '0','1','2') to the *context-derived* next-token id.
      - Returns the log-prob of that single next token (sum_logprob for 1 token).

    Returns: float (log-prob), or -inf if truncated.
    """
    device = model.device
    cand = str(answer)

    # 1) Normalize the cue and encode ONLY the context
    ctx = ensure_answer_cue(prompt)  # ensures it ends with exactly 'Answer: '
    enc_ctx = tokenizer(ctx, return_tensors="pt",
                        add_special_tokens=True, truncation=True, max_length=max_len)
    ids_ctx = enc_ctx.input_ids.to(device)              # [1, T_ctx]

    # 2) Forward pass on the context; get next-position logits
    out = model(ids_ctx)
    logits_next = out.logits[0, -1, :]                  # [V]
    logprobs_next = torch.log_softmax(logits_next, dim=-1)

    # 3) Find the *context-derived* token id for this candidate
    #    (encode ctx+cand and take the first new token after ctx)
    enc_full = tokenizer(ctx + cand, return_tensors="pt",
                         add_special_tokens=True, truncation=True, max_length=max_len)
    ids_full = enc_full.input_ids[0]                    # [T_full]
    T_ctx = ids_ctx.shape[-1]
    if ids_full.shape[-1] <= T_ctx:
        if debug:
            _printer(f"[dbg][cand={cand}][WARN] Truncated while mapping candidate; increase max_len.")
        return float("-inf")

    cand_token_id = ids_full[T_ctx].item()              # the *first* new token id for `cand`
    lp = float(logprobs_next[cand_token_id].item())     # log-prob for that token

    if not debug:
        return lp

    # ---- Debug prints (single-token classification) ----
    tok_str = tokenizer.convert_ids_to_tokens([cand_token_id])[0]
    prob = float(torch.exp(torch.tensor(lp)).item())

    _printer(f"[dbg][cand={cand}] prompt_len={T_ctx} total_T_ctx={ids_ctx.shape[-1]}")
    _printer(f"[dbg][cand={cand}] next-token id: {cand_token_id}")
    _printer(f"[dbg][cand={cand}] next-token str: {tok_str!r}")
    _printer(f"[dbg][cand={cand}] logprob: {lp:.6f}  prob: {prob:.6f}")
    _printer("****************************************************")

    return lp
    '''


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

def get_answers_for_df(
    df,
    model,
    tokenizer,
    answer_choices=['0', '1', '2'],
    sample_size=None,
    random_state=22,
    nas_writer=None,          
    max_len=768
    ):
    # Optional subsample
    if sample_size is not None:
        original_size = len(df)
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state).reset_index(drop=True)
        print(f"Sampling {len(df)} rows from {original_size} total rows")

    predicted_answers = []
    max_log_likelihoods = []
    nas_scalars = []          
    all_answer_logli = {a: [] for a in answer_choices}

    start_time = time.time()

    # We need the full row (not just the prompt) to read bias_label / unknown_label / answer_texts
    for i, row in enumerate(tqdm(df.itertuples(index=False), desc="Processing prompts", total=len(df))):
        rd = row._asdict()
        prompt_text = rd["prompt"]

        # 1) Likelihoods over your digit choices (keep your existing behavior)
        results = get_answer_from_likelihoods(prompt_text, model, tokenizer, answer_choices)
        pred_digit = results['predicted_answer']

        predicted_answers.append(pred_digit)
        max_log_likelihoods.append(results['max_log_likelihood'])
        for a in answer_choices:
            all_answer_logli[a].append(results['log_likelihoods'].get(a, float('-inf')))

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

        # use THIS prompt_len for "i >= prompt_len" queries in NAS
        prompt_len_ctx = input_ids_prompt.shape[-1]
        with torch.no_grad():
            out = model(**inputs_full, labels=labels, output_attentions=True, use_cache=False, return_dict=True)
            nas_lh, nas_scalar = nas_from_attn(out.attentions, prompt_len_ctx, S_cols, A_cols)

        nas_scalars.append(nas_scalar)

        # 4) Optionally write per-(layer, head) rows now (so you can aggregate later)
        if nas_writer is not None:
            ex_id = rd.get("example_id", f"row_{i}")
            L, H = nas_lh.shape
            for l in range(L):
                for h in range(H):
                    nas_writer.writerow([ex_id, l, h, float(nas_lh[l, h])])

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.2f} minutes ({total_time/len(df):.2f}s per prompt)")

    # Add outputs to the dataframe 
    out = df.copy()
    out["predicted_answer"] = predicted_answers
    out["max_log_likelihood"] = max_log_likelihoods
    out["nas_stereo_scalar"] = nas_scalars   
    for a in answer_choices:
        out[f"prob_{a}"] = all_answer_logli[a]
    return out

def process_dataset(input_filename, model, tokenizer, model_name, sample_size):
    dataset_name = op.basename(input_filename).replace('prompts_', '').replace('.csv','').replace('_',' ').title()

    # Always write outputs into ./data
    os.makedirs('data', exist_ok=True)

    # Use the input path exactly as provided 
    input_path = input_filename
    if not op.exists(input_path):
        # optional fallback 
        alt = op.join('data', input_filename)
        if op.exists(alt):
            input_path = alt
        else:
            raise FileNotFoundError(f"Input file not found: {input_filename} (also tried {alt})")

    # Build output name in ./data, stripping any directories from the input
    base_in = op.basename(input_filename)                      # e.g., "prompts_gender_ambig_no_cot_withAnswer.csv"
    output_filename = base_in.replace('prompts_', f'{model_name}_responses_')
    output_path = op.join('data', output_filename)             # e.g., "data/llama8b_responses_gender_ambig_no_cot_withAnswer.csv"

    print("\n" + "=" * 60)
    print(f"Processing: {dataset_name}")
    print("=" * 60)

    df = pd.read_csv(input_path)

    # Ensure answer_texts are proper lists
    if isinstance(df.iloc[0]["answer_texts"], str):
        df["answer_texts"] = df["answer_texts"].apply(lambda s: ast.literal_eval(s))

    # Per-(layer, head) NAS rows for this dataset
    nas_rows_path = op.join("data", f"nas_rows_{model_name}_{op.splitext(base_in)[0]}.csv")
    write_header = not op.exists(nas_rows_path)
    with open(nas_rows_path, "a", newline="") as nas_f:
        nas_writer = csv.writer(nas_f)
        if write_header:
            nas_writer.writerow(["example_id", "layer", "head", "nas"])

        responses = get_answers_for_df(
            df, model, tokenizer, sample_size=sample_size, nas_writer=nas_writer
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
def nas_from_attn(attentions, prompt_len, S_cols, A_cols):
    """
    attentions: list over layers, each [B,H,T,S] (self-attention weights)
    Uses answer rows (i >= prompt_len) as queries, sums attention to Stereotypical and Antistereotypical columns.
    Returns:
      nas_lh: [L,H] tensor (per-layer, per-head NAS for this prompt)
      nas_scalar: float (sum over L,H), this is not as important as nas_lh .. not sure if we neeed it 
    """
    if not S_cols or not A_cols:
        L, H = len(attentions), attentions[0].shape[1]
        return torch.zeros(L, H, dtype=torch.float64), 0.0

    per_layer = []
    for layer_attn in attentions:  # [B,H,T,S]
        A = layer_attn[0]               # [H,T,S]
        ans_rows = A[:, prompt_len:, :] # queries = answer tokens
        mS = ans_rows[:, :, S_cols].sum(-1)  # [H, T_ans]
        mA = ans_rows[:, :, A_cols].sum(-1)
        term = (mS + mA) * torch.log((mS + EPS)/(mA + EPS))
        per_layer.append(term.sum(-1).double())  # [H]
    nas = torch.stack(per_layer, dim=0)          # [L,H]

    return nas, float(nas.sum().item())


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

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    dtype="auto",              
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="eager"  
    )
    model.eval()
    model.config.return_dict = True
    # Do NOT set model.config.output_attentions globally, already passed per call.

    print("Model successfully accessed from cluster.")

    # Define all datasets to process
    datasets = [
        'prompts/prompts_gender_ambig_no_cot.csv',
        'prompts/prompts_gender_ambig_cot.csv',
        'prompts/prompts_gender_disambig_no_cot.csv',
        'prompts/prompts_gender_disambig_cot.csv'
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
