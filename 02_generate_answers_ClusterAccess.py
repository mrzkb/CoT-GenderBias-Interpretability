import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import time
from tqdm import tqdm 
# NOTE: tqdm.write() prints above a progress bar so as not to interupt the visualization

def hey_model(prompt, model, tokenizer, max_new_tokens=200, temperature=1.0):
    try:
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 
            
        # Generate Response
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, # Hard coding these for now, will make variable later.
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
                ) # Other arguments include do_sample, and pad_token_id. Should we use these?

        # Decode generated response from the model
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
        
    except Exception as e:
        return f"ERROR: {str(e)}"

def generate_responses_for_df(df, model, tokenizer, max_new_tokens=200, temperature=1.0, sample_size=None, random_state=22):
    # Allow for testing on smaller subset of the df
    if sample_size is not None:
        original_size = len(df)
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state).reset_index(drop=True) # Be careful about the index here Edie, make sure that doesn't throw any errors down the line
        print(f"Sampling {len(df)} rows from {original_size} total rows")

    responses = []
    start_time = time.time()

    for i, prompt in enumerate(tqdm(df['prompt'], desc="Processing prompts")):
        iter_start = time.time()

        # Prompt model
        response = hey_model(
            prompt,
            model,
            tokenizer,
            max_new_tokens,
            temperature
        )
        
        # Save response
        responses.append(response)

        # Timing information
        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        avg_time = elapsed / (i+1)
        remaining = avg_time * (len(df) - i - 1)

        # Progress Update
        status = "✗" if response.startswith("ERROR:") else "✓"
        tqdm.write(f"{status} Row {i+1}/{len(df)} | Time: {iter_time:.2f}s | Avg: {avg_time:.2f}s | ETA: {remaining/60:.1f}m")

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.2f} minutes ({total_time/len(df):.2f}s per prompt)")
    
    # Save the responses as a new column
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

# Load model from cluster
model_path = '/network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B' # Test this!!!!
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    dtype=torch.float16,    # I don't know what this does! You should try and understand how this part works edie!
    device_map="auto"       # Automatically use available GPUs. I don't understand this part either!
)

print("Model successfully accessed from cluster.")

print("=" * 60)
print("Processing: Gender Ambig No CoT")
print("=" * 60)
A_NCOT_df = pd.read_csv('data/prompts_gender_ambig_no_cot.csv')
A_NCOT_responses = generate_responses_for_df(A_NCOT_df, model, tokenizer, sample_size=100)
A_NCOT_responses.to_csv('data/responses_gender_ambig_no_cot.csv', index=False)

print("\n" + "=" * 60)
print("Processing: Gender Ambig CoT")
print("=" * 60)
A_COT_df = pd.read_csv('data/prompts_gender_ambig_cot.csv')
A_COT_responses = generate_responses_for_df(A_COT_df, model, tokenizer, sample_size=100)
A_COT_responses.to_csv('data/responses_gender_ambig_cot.csv', index=False)

print("\n" + "=" * 60)
print("Processing: Gender Disambig No CoT")
print("=" * 60)
D_NCOT_df = pd.read_csv('data/prompts_gender_disambig_no_cot.csv')
D_NCOT_responses = generate_responses_for_df(D_NCOT_df, model, tokenizer, sample_size=100)
D_NCOT_responses.to_csv('data/responses_gender_disambig_no_cot.csv', index=False)

print("\n" + "=" * 60)
print("Processing: Gender Disambig CoT")
print("=" * 60)
D_COT_df = pd.read_csv('data/prompts_gender_disambig_cot.csv')
D_COT_responses = generate_responses_for_df(D_COT_df, model, tokenizer, sample_size=100)
D_COT_responses.to_csv('data/responses_gender_disambig_cot.csv', index=False)

print("\nAll datasets processed, cheers!")