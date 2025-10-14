import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import time
from tqdm import tqdm 
# NOTE: tqdm.write() prints above a progress bar so as not to interupt the visualization

def compute_log_likelihood(prompt, answer, model, tokenizer):
    try:
        # Tokenize prompt
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        # Tokenize prompt + possible answer
        full_input = prompt + "\nAnswer: " + answer
        full_input_ids = tokenizer(full_input, return_tensors="pt").input_ids.to(model.device)

        # Create labels in order to mask the prompt tokens, only compute loss on the answer
        prompt_len = prompt_ids.shape[-1]
        labels = full_input_ids.clone() # What does label mean in this context? What does mask mean?
        labels[:, :prompt_len] = -100 # -100 is ignored in the loss calculation

        # Compute loss (negative log-likelihood)
        with torch.no_grad():
            loss = model(full_input_ids, labels=labels).loss.item()

        # Approximate log-likelihood 
        log_likelihood = -loss * (full_input_ids.shape[-1] - prompt_len)

        return log_likelihood

    except Exception as e:
        # tqdm.write(f'Error: {str(e)}')
        return float('-inf') # Return very low likelihood on error


def get_answer_from_likelihoods(prompt, model, tokenizer, answer_choices=['0', '1', '2']): # I think this can be significantly simplified.
    try:
        log_likelihoods = {}

        for answer in answer_choices:
            log_lik = compute_log_likelihood(prompt, answer, model, tokenizer)
            log_likelihoods[answer] = log_lik

        # Determine the predicted answer based on the highest probability
        predicted_answer = max(log_likelihoods, key=log_likelihoods.get)
        tqdm.write(f'Predicted Answer: {predicted_answer}')

        return {
            'predicted_answer': predicted_answer,
            'log_likelihoods': log_likelihoods,
            'max_log_likelihood': log_likelihoods[predicted_answer]
        }
        
    except Exception as e:
        return {
            'predicted_answer': 'ERROR',
            'log_likelihoods': {answer: float('-inf') for answer in answer_choices},
            'max_log_likelihood': float('-inf'),
            'error': str(e)
        }

def validate_likelihood_vs_generation(prompt, model, tokenizer, answer_choices=['0', '1', '2']):
    """Compare log-likelihood prediction with actual generation"""
    
    # Method 1: Log-likelihood
    log_liks = {}
    for answer in answer_choices:
        log_liks[answer] = compute_log_likelihood(prompt, answer, model, tokenizer)
    predicted_by_likelihood = max(log_liks, key=log_liks.get)
    
    # Method 2: Actual generation
    inputs = tokenizer(prompt + '\nAnswer: ', return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    generated_answer = generated_text.split("Answer:")[-1].strip()[0]  # Get first char
    # generated_answer = generated_text[0]
    
    # Compare
    match = "✓" if predicted_by_likelihood == generated_answer else "✗"
    print(f"{match} Likelihood predicted: {predicted_by_likelihood}, Generated: {generated_answer}")
    print(f"   Log-likelihoods: {log_liks}")
    
    return predicted_by_likelihood == generated_answer

# Load model from cluster
model_path = '/network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    dtype=torch.float16,    # I don't know what this does! You should try and understand how this part works edie!
    device_map="auto"       # Automatically use available GPUs. I don't understand this part either!
)

print("Model successfully accessed from cluster.")

# Test on a few examples
test_df = pd.read_csv('data/prompts_gender_ambig_no_cot.csv').head(10)
matches = 0
for i, prompt in enumerate(test_df['prompt']):
    print(f"\n--- Example {i+1} ---")
    if validate_likelihood_vs_generation(prompt, model, tokenizer):
        matches += 1

print(f"\n{matches}/{len(test_df)} predictions matched generation")

# This worked, 10/10 match between the loglikelihood answers the fully generated ones.