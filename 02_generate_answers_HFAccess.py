import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
 
import time
from tqdm import tqdm

def get_model_responses(df):
    responses = []
    start_time = time.time()

    # optional, test on small subset
    # df = df.head(5)

    for i, prompt in enumerate(tqdm(df['prompt'], desc="Processing prompts")):
        try:
            iter_start = time.time()

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=1.0)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)

            iter_time = time.time() - iter_start
            elapsed = time.time() - start_time
            avg_time = elapsed / (i+1)
            remaining = avg_time * (len(df) - i - 1)
            tqdm.write(f"✓ Row {i+1}/{len(df)} | Time: {iter_time:.2f}s | Avg: {avg_time:.2f}s | ETA: {remaining/60:.1f}m")
            
        except Exception as e:
            tqdm.write(f"✗ Error on row {i}: {e}")
            responses.append(f"ERROR: {str(e)}")

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.2f} minutes ({total_time/len(df):.2f}s per prompt)")
    
    # Add the responses as a new column
    df["model_response"] = responses
    return df

### Our Standard models
### LLAMA
#   'meta-llama/Meta-Llama-3.1-8B' (on the cluster /network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B)
#   'meta-llama/Meta-Llama-3.1-8B-Instruct' (on the cluster /network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B-Instruct)
#   'meta-llama/Llama-3.1-70B' (on the cluster /network/weights/llama.var/llama_3.1/Meta-Llama-3.1-70B)
#   'meta-llama/Llama-3.1-70B-Instruct' (on the cluster /network/weights/llama.var/llama_3.1/Meta-Llama-3.1-70B-Instruct)
#   'meta-llama/Llama-3.3-70B-Instruct' (on the cluster /network/weights/llama.var/llama_3.3/Meta-Llama-3.3-70B-Instruct)
#   NOTE: The llama models stored on the cluster seem to be in the same hugging face format with .safetensor files, etc.

### Qwen
#   'Qwen/Qwen-7B'
#   'Qwen/Qwen-7B-Chat'

### Mistral
#   'mistralai/Mistral-7B-v0.3'
#   'mistralai/Mistral-7B-Instruct-v0.3'

### Our Reasoning models
### Qwen
#   'Qwen/QwQ-32B'

### GPT
#   'openai/gpt-oss-20b'
#   'openai/gpt-oss-120b'

# Load model locally (downloads to your machine)
HF_token = 'its a secret' # replace with your own
model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    token=HF_token,
    dtype=torch.float16,
    device_map="auto"       # You should try and understand how this part works edie!
)

print("=" * 60)
print("Processing: Gender Ambig No CoT")
print("=" * 60)
A_NCOT_df = pd.read_csv('data/prompts_gender_ambig_no_cot.csv')
A_NCOT_responses = get_model_responses(A_NCOT_df)
A_NCOT_responses.to_csv('data/responses_gender_ambig_no_cot.csv', index=False)

print("\n" + "=" * 60)
print("Processing: Gender Ambig CoT")
print("=" * 60)
A_COT_df = pd.read_csv('data/prompts_gender_ambig_cot.csv')
A_COT_responses = get_model_responses(A_COT_df)
A_COT_responses.to_csv('data/responses_gender_ambig_cot.csv', index=False)

print("\n" + "=" * 60)
print("Processing: Gender Disambig No CoT")
print("=" * 60)
D_NCOT_df = pd.read_csv('data/prompts_gender_disambig_no_cot.csv')
D_NCOT_responses = get_model_responses(D_NCOT_df)
D_NCOT_responses.to_csv('data/responses_gender_disambig_no_cot.csv', index=False)

print("\n" + "=" * 60)
print("Processing: Gender Disambig CoT")
print("=" * 60)
D_COT_df = pd.read_csv('data/prompts_gender_disambig_cot.csv')
D_COT_responses = get_model_responses(D_COT_df)
D_COT_responses.to_csv('data/responses_gender_disambig_cot.csv', index=False)

print("\nAll datasets processed!")