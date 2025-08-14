import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.config.output_attentions = True
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.chat_template = None

def find_sublist_indices(full_list, sublist):
    for i in range(len(full_list) - len(sublist) + 1):
        if full_list[i:i+len(sublist)] == sublist:
            return i, i + len(sublist)
    return None, None

def extract_attention_per_sensitive_token(correct_label, choices, tokens, attentions, input_ids, tokenizer):
    final_token_idx = input_ids.shape[-1] - 1
    input_id_list = input_ids[0].tolist()

    sensitive_words = []
    for idx, choice in enumerate(choices):
        if idx == correct_label:
            continue
        if ' ' in choice:
            _, word = choice.strip().split(' ', 1)
        else:
            word = choice.strip()
        last_word = word.strip().split()[-1]
        sensitive_words.append(last_word)

    if len(sensitive_words) != 2:
        raise ValueError(f"Expected 2 sensitive words, got {len(sensitive_words)}: {sensitive_words}")

    positions = []
    for word in sensitive_words:
        token_ids = tokenizer(" " + word, add_special_tokens=False).input_ids
        start, end = find_sublist_indices(input_id_list, token_ids)

        if start is None:
            print(f"[ERROR] Could not find word '{word}' in the prompt.")
            print(f"→ Target token IDs: {token_ids}")
            print(f"→ Prompt tokens: {tokenizer.convert_ids_to_tokens(input_id_list)}")
            print(f"→ Target tokens: {tokenizer.convert_ids_to_tokens(token_ids)}")
            raise ValueError(f"Couldn't find token span for word: {word}")

        positions.append(start)

    pos_1, pos_2 = positions
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    attn_matrix_1 = np.zeros((num_layers, num_heads))
    attn_matrix_2 = np.zeros((num_layers, num_heads))

    for l, layer_attn in enumerate(attentions):
        attn_from_final = layer_attn[0, :, final_token_idx, :]
        attn_matrix_1[l] = attn_from_final[:, pos_1].cpu().numpy()
        attn_matrix_2[l] = attn_from_final[:, pos_2].cpu().numpy()

    return attn_matrix_1, attn_matrix_2, sensitive_words, [pos_1, pos_2]

def compute_log_likelihood(prompt, answer, model, tokenizer):
    full_input = prompt + "\nAnswer: " + answer
    input_ids = tokenizer(full_input, return_tensors="pt").input_ids.to(model.device)
    prompt_len = tokenizer(prompt + "\nAnswer: ", return_tensors="pt").input_ids.shape[-1]

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels, output_attentions=True)
        loss = outputs.loss.item()
        attentions = outputs.attentions  # all layers

    return -loss * (input_ids.shape[-1] - prompt_len), attentions, input_ids

df = pd.read_csv("gender_bbq/prompts/bbqPrompts_ambig_cot.csv") #no chain of thought

attention_results = []

for i, row in df.iterrows():
    prompt = row["formatted_prompt"]
    true_label = row["correct_label"]
    bias_label = row['answer_choices'][row['bias_label']]
    choices = eval(row["answer_choices"]) if isinstance(row["answer_choices"], str) else row["answer_choices"]

    scores = []
    all_attentions = []
    input_ids_list = []
    tokens_list = []

    for j, choice in enumerate(choices):
        full_answer = f"{j} {choice}"
        score, attentions, input_ids = compute_log_likelihood(prompt, full_answer, model, tokenizer)
        scores.append(score)
        all_attentions.append(attentions)
        input_ids_list.append(input_ids)
        tokens_list.append(tokenizer.convert_ids_to_tokens(input_ids[0]))

    pred = int(torch.tensor(scores).argmax().item())

    is_correct = (pred == true_label)
    is_bias_aligned = (pred == row["bias_label"])

    if is_correct:
        group = "correct"
    elif is_bias_aligned:
        group = "bias_wrong"
    else:
        group = "anti_bias_wrong"


    attentions = all_attentions[pred]
    input_ids = input_ids_list[pred]
    tokens = tokens_list[pred]

    attn_matrix_1, attn_matrix_2, sensitive_words, positions = extract_attention_per_sensitive_token(
        true_label, choices, tokens, attentions, input_ids, tokenizer
    )

    _, bias_word_raw = bias_label.strip().split(' ', 1) if ' ' in bias_label else ("", bias_label)
    bias_word = bias_word_raw.strip().split()[-1]

    if bias_word == sensitive_words[0]:
        bias_matrix = attn_matrix_1
        anti_bias_matrix = attn_matrix_2
        bias_pos = positions[0]
        anti_bias_pos = positions[1]
    else:
        bias_matrix = attn_matrix_2
        anti_bias_matrix = attn_matrix_1
        bias_pos = positions[1]
        anti_bias_pos = positions[0]

    attention_results.append({
        "prompt_index": i,
        "true_label": true_label,
        "pred_label": pred,
        "bias_label_idx": row["bias_label"],
        "bias_word": bias_word,
        "sensitive_word_1": sensitive_words[0],
        "sensitive_word_2": sensitive_words[1],
        "bias_pos": bias_pos,
        "anti_bias_pos": anti_bias_pos,
        "attn_matrix_shape": f"{bias_matrix.shape[0]}x{bias_matrix.shape[1]}",
        "bias_matrix_flat": bias_matrix.flatten(),
        "anti_bias_matrix_flat": anti_bias_matrix.flatten(),
        "prompt_group": group
    })

    if i % 10 == 0:
        print(f"Processed {i} examples")

rows = []
for entry in attention_results:
    row = {
        "prompt_index": entry["prompt_index"],
        "true_label": entry["true_label"],
        "pred_label": entry["pred_label"],
        "bias_label_idx": entry["bias_label_idx"],
        "bias_word": entry["bias_word"],
        "sensitive_word_1": entry["sensitive_word_1"],
        "sensitive_word_2": entry["sensitive_word_2"],
        "bias_pos": entry["bias_pos"],
        "anti_bias_pos": entry["anti_bias_pos"],
        "attn_matrix_shape": entry["attn_matrix_shape"],
        "bias_matrix_flat": ",".join(map(str, entry["bias_matrix_flat"])),
        "anti_bias_matrix_flat": ",".join(map(str, entry["anti_bias_matrix_flat"])),
        'prompt_group': entry["prompt_group"]
    }
    rows.append(row)

df_out = pd.DataFrame(rows)
df_out.to_csv("gender_bbq/results/attention_results/attention_per_prompt_COT.csv", index=False)