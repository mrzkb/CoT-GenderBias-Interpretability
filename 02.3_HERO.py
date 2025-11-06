from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import time
from tqdm import tqdm 
# NOTE: tqdm.write() prints above a progress bar so as not to interupt the visualization
import argparse
import os
import ast

#### This one is for crowspairs and stero?
### LLAMA
#   /network/weights/llama.var/llama_3/Meta-Llama-3-8B-Instruct
#   /network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B-Instruct
#   /network/weights/llama.var/llama_3.3/Meta-Llama-3.3-70B-Instruct

#   /network/weights/llama.var/llama_2/llama-2-7b-chat
#   /network/weights/llama.var/llama_2/Llama-2-7b-chat-hf

def compute_log_likelihood(prompt, answer, model, tokenizer):
    try:
        # Tokenize prompt
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        # Tokenize prompt + possible answer
        full_input = prompt + "\nAnswer: " + answer
        # full_input = prompt + "\n " + answer # Significant decrease in Disambiguous Accuracy
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


def get_answer_from_likelihoods(prompt, model, tokenizer, answer_choices=['0', '1', '2']):
    try:
        log_likelihoods = {} # dictionary, each key is an MCQ index '0', '1', '2'

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

def get_answers_for_df(df, model, tokenizer, answer_choices=['0', '1', '2'], sample_size=None, random_state=22):
    # Allow for testing on smaller subset of the df
    if sample_size is not None:
        original_size = len(df)
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state).reset_index(drop=True) # Be careful about the index here Edie, make sure that doesn't throw any errors down the line
        print(f"Sampling {len(df)} rows from {original_size} total rows")

    # Store results with example_id for safe merging
    results_data = []
    
    start_time = time.time()

    for i, row in enumerate(tqdm(df.itertuples(), desc="Processing prompts", total=len(df))):
        example_id = row.example_id
        prompt = row.prompt
        
        iter_start = time.time()

        # Get likelihoods
        results = get_answer_from_likelihoods(prompt, model, tokenizer, answer_choices)
        
        # Save results attached to example_id to ensure no indexing errors
        result_row = {
            'example_id': example_id,
            'predicted_answer': str(results['predicted_answer']), # string '0', '1', or '2'
            'max_log_likelihood': results['max_log_likelihood'] # float
        }

        # Add individual answer probabilities
        for answer in answer_choices:
            result_row[f"prob_{answer}"] = results['log_likelihoods'].get(answer, float('-inf'))
        
        results_data.append(result_row)
    
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
    
    # Create results dataframe and merge on example_id
    results_df = pd.DataFrame(results_data)
    df = df.merge(results_df, on='example_id', how='left')

    return df

def process_dataset(input_filename, model, tokenizer, model_name, sample_size):
    dataset_name = input_filename.replace('prompts_', '').replace('.csv', '').replace('_', ' ').title()
    
    os.makedirs('data', exist_ok=True)
    input_path = f'data/{input_filename}'

    output_filename = input_filename.replace('prompts_stereoset_', f'{model_name}_responses_gender_')
    output_path = f'data/stereo/{output_filename}'

    # output_filename = input_filename.replace('prompts_crowspairs_', f'{model_name}_responses_gender_')
    # output_path = f'data/crows/{output_filename}'
    
    print("\n" + "=" * 60)
    print(f"Processing: {dataset_name}")
    print("=" * 60)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)

    df = df[df[bias_type] == 'gender']
    
    # Check data types
    # print("Check Data Types")
    # print(df.dtypes)

    responses = get_answers_for_df(df, model, tokenizer, sample_size=sample_size)
    responses.to_csv(output_path, index=False) # when saving to csv, predicted_answer gets saved as int by default
    print(f"✓ Saved to: {output_path}\n")
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

    # Load model from cluster
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        dtype=torch.float16,    # I don't know what this does! You should try and understand how this part works edie!
        device_map="auto"       # Automatically use available GPUs. I don't understand this part either!
    )

    print("Model successfully accessed from cluster.")

    datasets = [
        'prompts_stereoset_cot.csv',
        'prompts_stereoset_no_cot.csv',
    ]

    # datasets = [
    #     'prompts_crowspairs_cot.csv',
    #     'prompts_crowspairs_no_cot.csv',
    # ]
    
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

if __name__ == '__main__':
    main()