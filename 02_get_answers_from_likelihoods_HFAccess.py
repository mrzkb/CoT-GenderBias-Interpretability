from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import time
from tqdm import tqdm 
# NOTE: tqdm.write() prints above a progress bar so as not to interupt the visualization
import argparse
import os

### Mistral
#   'mistralai/Mistral-7B-Instruct-v0.3'

### GPT
#   'openai/gpt-oss-20b' (reasoning model)
#   'openai/gpt-oss-120b' (reasoning model)

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
        tqdm.write(f'Error: {str(e)}')
        return float('-inf') # Return very low likelihood on error


def get_answer_from_likelihoods(prompt, model, tokenizer, answer_choices=['0', '1', '2']):
    try:
        log_likelihoods = {}

        for answer in answer_choices:
            log_likelihoods[answer] = compute_log_likelihood(prompt, answer, model, tokenizer)

        # Determine the predicted answer based on the highest probability
        predicted_answer = max(log_likelihoods, key=log_likelihoods.get)

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

def get_answers_for_df(df, model, tokenizer, answer_choices=['0', '1', '2'], sample_size=None, random_state=22):
    # Allow for testing on smaller subset of the df
    if sample_size is not None:
        original_size = len(df)
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state).reset_index(drop=True) # Be careful about the index here Edie, make sure that doesn't throw any errors down the line
        print(f"Sampling {len(df)} rows from {original_size} total rows")

    predicted_answers = []
    max_log_likelihoods = []
    all_answer_logli = {answer: [] for answer in answer_choices}

    start_time = time.time()

    for i, prompt in enumerate(tqdm(df['prompt'], desc="Processing prompts")):
        iter_start = time.time()

        # Get likelihoods
        results = get_answer_from_likelihoods(prompt, model, tokenizer, answer_choices)
        
        # Save results
        predicted_answers.append(results['predicted_answer'])
        max_log_likelihoods.append(results['max_log_likelihood'])
        for answer in answer_choices:
            all_answer_logli[answer].append(results['log_likelihoods'].get(answer, float('-inf')))

        # Timing information
        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        avg_time = elapsed / (i+1)
        remaining = avg_time * (len(df) - i - 1)

        # Progress Update
        # status = "✗" if results['predicted_answer'].startswith("ERROR:") else "✓"
        # tqdm.write(f"{status} Row {i+1}/{len(df)} | Time: {iter_time:.2f}s | Avg: {avg_time:.2f}s | ETA: {remaining/60:.1f}m")

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.2f} minutes ({total_time/len(df):.2f}s per prompt)")
    
    # Add results as new columns
    df["predicted_answer"] = predicted_answers
    df["max_log_likelihood"] = max_log_likelihoods
    
    # Add individual choice probabilities
    for answer in answer_choices:
        df[f"prob_{answer}"] = all_answer_logli[answer]

    return df

def process_dataset(input_filename, model, tokenizer, model_name, sample_size):
    dataset_name = input_filename.replace('prompts_', '').replace('.csv', '').replace('_', ' ').title()
    
    os.makedirs('data', exist_ok=True)
    input_path = f'data/{input_filename}'
    output_filename = input_filename.replace('prompts_', f'{model_name}_responses_')
    output_path = f'data/{output_filename}'
    
    print("\n" + "=" * 60)
    print(f"Processing: {dataset_name}")
    print("=" * 60)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    responses = get_answers_for_df(df, model, tokenizer, sample_size=sample_size)
    responses.to_csv(output_path, index=False)
    print(f"✓ Saved to: {output_path}\n")  # Nice feedback!
    return

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

    # Load model from hugging face
    HF_token = 'its a secret'
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=HF_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        token=HF_token,
        dtype=torch.float16,    # I don't know what this does! You should try and understand how this part works edie!
        device_map="auto"       # Automatically use available GPUs. I don't understand this part either!
    )
    print("Model successfully accessed from hugging face.")

    # Define all datasets to process
    datasets = [
        'prompts_gender_ambig_no_cot.csv',
        'prompts_gender_ambig_cot.csv',
        'prompts_gender_disambig_no_cot.csv',
        'prompts_gender_disambig_cot.csv'
    ]
    
    # Process each dataset
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

if __name__=='__main__':
    main()