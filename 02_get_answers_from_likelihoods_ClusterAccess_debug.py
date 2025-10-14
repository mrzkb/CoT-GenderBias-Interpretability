from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import time
from tqdm import tqdm 
import argparse
import os
import ast

### LLAMA
#   /network/weights/llama.var/llama_3/Meta-Llama-3-8B-Instruct
#   /network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B-Instruct
#   /network/weights/llama.var/llama_3.3/Meta-Llama-3.3-70B-Instruct

# salloc --gres=gpu:1 --cpus-per-task=6 --mem=32G --time=01:00:0 just so I don't lose this rn

def compute_log_likelihood(prompt, answer, model, tokenizer, debug=False):
    try:
        # Tokenize prompt
        prompt_with_answer_prefix = prompt + "\nAnswer: "
        prompt_ids = tokenizer(prompt_with_answer_prefix, return_tensors="pt").input_ids.to(model.device)
        prompt_len = prompt_ids.shape[-1]

        # Tokenize prompt + possible answer
        full_input = prompt + "\nAnswer: " + answer
        full_input_ids = tokenizer(full_input, return_tensors="pt").input_ids.to(model.device)

        if debug:
            tqdm.write("\n" + "="*80)
            tqdm.write("DEBUG: compute_log_likelihood")
            tqdm.write("="*80)
            tqdm.write(f"Prompt:\n{prompt}")
            tqdm.write(f"\nAnswer being evaluated: '{answer}'")
            tqdm.write(f"\nFull input:\n{full_input}")
            tqdm.write(f"\nPrompt length (tokens): {prompt_len}")
            tqdm.write(f"Full input length (tokens): {full_input_ids.shape[-1]}")
            tqdm.write(f"Answer length (tokens): {full_input_ids.shape[-1] - prompt_len}")
            
            # Show actual tokens
            # tqdm.write(f"\nPrompt tokens: {tokenizer.decode(prompt_ids[0])}")
            # tqdm.write(f"Full tokens: {tokenizer.decode(full_input_ids[0])}")

        # Create labels in order to mask the prompt tokens, only compute loss on the answer
        labels = full_input_ids.clone()
        labels[:, :prompt_len] = -100 # -100 is ignored in the loss calculation

        if debug:
            tqdm.write(f"\nLabels shape: {labels.shape}")
            tqdm.write(f"Number of masked tokens (prompt): {(labels == -100).sum().item()}")
            tqdm.write(f"Number of unmasked tokens (answer): {(labels != -100).sum().item()}")

        # Compute loss (negative log-likelihood)
        with torch.no_grad():
            outputs = model(full_input_ids, labels=labels)
            loss = outputs.loss.item()

        # Approximate log-likelihood 
        log_likelihood = -loss * (full_input_ids.shape[-1] - prompt_len)

        if debug:
            tqdm.write(f"\nLoss: {loss}")
            tqdm.write(f"Log-likelihood: {log_likelihood}")
            # tqdm.write(f"New LL: {-loss}")
            tqdm.write("="*80 + "\n")

        return log_likelihood

    except Exception as e:
        tqdm.write(f'Error in compute_log_likelihood: {str(e)}')
        import traceback
        tqdm.write(traceback.format_exc())
        return float('-inf')


def get_answer_from_likelihoods(prompt, model, tokenizer, answer_texts, debug=False):
    try:
        log_likelihoods = {}

        if debug:
            tqdm.write("\n" + "#"*80)
            tqdm.write("DEBUG: get_answer_from_likelihoods")
            tqdm.write("#"*80)
            tqdm.write(f"Number of answer choices: {len(answer_texts)}")
            tqdm.write(f"Answer texts: {answer_texts}")

        for i, answer_text in enumerate(answer_texts):
            full_answer = f"{i} {answer_text}"
            if debug:
                tqdm.write(f"\n--- Evaluating choice {i}: '{full_answer}' ---")
            
            log_likelihoods[str(i)] = compute_log_likelihood(
                prompt, full_answer, model, tokenizer, debug=debug
            )

        # Determine the predicted answer based on the highest probability
        predicted_answer = max(log_likelihoods, key=log_likelihoods.get)

        if debug:
            tqdm.write("\n" + "-"*80)
            tqdm.write("FINAL RESULTS:")
            for i, ll in log_likelihoods.items():
                marker = " <-- PREDICTED" if i == predicted_answer else ""
                tqdm.write(f"Choice {i}: log_likelihood = {ll:.4f}{marker}")
            tqdm.write("#"*80 + "\n")

        return {
            'predicted_answer': predicted_answer,
            'max_log_likelihood': log_likelihoods[predicted_answer],
            'log_likelihoods': log_likelihoods
        }
        
    except Exception as e:
        tqdm.write(f'Error in get_answer_from_likelihoods: {str(e)}')
        import traceback
        tqdm.write(traceback.format_exc())
        return {
            'predicted_answer': 'ERROR',
            'max_log_likelihood': float('-inf'),
            'log_likelihoods': {str(i): float('-inf') for i in range(len(answer_texts))},
            'error': str(e)
        }

def get_answers_for_df(df, model, tokenizer, sample_size=None, random_state=22, debug_rows=None):
    """
    debug_rows: list of row indices to debug, e.g., [0, 1, 5] or None to debug none
    """
    # Allow for testing on smaller subset of the df
    if sample_size is not None:
        original_size = len(df)
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state).reset_index(drop=True)
        print(f"Sampling {len(df)} rows from {original_size} total rows")

    predicted_answers = []
    max_log_likelihoods = []
    all_answer_logli = {answer: [] for answer in ['0', '1', '2']}

    start_time = time.time()

    for i, (prompt, answer_texts, correct_answer) in enumerate(tqdm(
        zip(df['prompt'], df['answer_texts'], df['answer_label']), 
        desc="Processing prompts", 
        total=len(df)
    )):
        iter_start = time.time()

        # Debug specific rows if requested
        debug_this_row = debug_rows is not None and i in debug_rows
        
        if debug_this_row:
            tqdm.write(f"\n{'*'*80}")
            tqdm.write(f"DEBUGGING ROW {i}")
            tqdm.write(f"Correct answer should be: {correct_answer}")
            tqdm.write(f"{'*'*80}")

        # Get likelihoods
        results = get_answer_from_likelihoods(
            prompt, model, tokenizer, answer_texts, debug=debug_this_row
        )
        
        # Save results
        predicted_answers.append(results['predicted_answer'])
        max_log_likelihoods.append(results['max_log_likelihood'])
        for answer in results['log_likelihoods']:
            all_answer_logli[answer].append(results['log_likelihoods'].get(answer, float('-inf')))
    
        # Show correctness for debugging rows
        if debug_this_row:
            is_correct = results['predicted_answer'] == str(correct_answer)
            tqdm.write(f"\nPredicted: {results['predicted_answer']} | Correct: {correct_answer} | Match: {is_correct}")
            tqdm.write(f"{'*'*80}\n")

        # Timing information
        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        avg_time = elapsed / (i+1)
        remaining = avg_time * (len(df) - i - 1)

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.2f} minutes ({total_time/len(df):.2f}s per prompt)")
    
    # Add results as new columns
    df["predicted_answer"] = predicted_answers
    df["max_log_likelihood"] = max_log_likelihoods
    
    # Add individual choice probabilities
    for i in ['0', '1', '2']:
        df[f"prob_{i}"] = all_answer_logli[i]

    # Calculate and display accuracy
    df['predicted_answer_int'] = df['predicted_answer'].apply(lambda x: int(x) if x != 'ERROR' else -1)
    accuracy = (df['predicted_answer_int'] == df['answer_label']).mean()
    print(f"\nAccuracy: {accuracy:.2%} ({(df['predicted_answer_int'] == df['answer_label']).sum()}/{len(df)})")
    
    # Show confusion matrix if useful
    print("\nPrediction distribution:")
    print(df['predicted_answer'].value_counts().sort_index())
    print("\nCorrect answer distribution:")
    print(df['answer_label'].value_counts().sort_index())

    return df

def process_dataset(input_filename, model, tokenizer, model_name, sample_size, debug_rows=None):
    dataset_name = input_filename.replace('prompts_', '').replace('.csv', '').replace('_', ' ').title()
    
    os.makedirs('data', exist_ok=True)
    input_path = f'data/{input_filename}'
    output_filename = input_filename.replace('prompts_', f'{model_name}_responses_')
    output_path = f'data/{output_filename}'
    
    tqdm.write(input_filename)
    tqdm.write(input_path)
    tqdm.write(output_filename)
    tqdm.write(output_path)

    print("\n" + "=" * 60)
    print(f"Processing: {dataset_name}")
    print("=" * 60)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    
    # Verify required columns
    print(f"Columns in dataset: {df.columns.tolist()}")
    print(f"Dataset size: {len(df)} rows")
    
    # Check first few rows
    # print("\nFirst row preview:")
    # print(f"Prompt preview: {df['prompt'].iloc[0][:200]}...")
    # print(f"Answer texts: {df['answer_texts'].iloc[0]}")
    # print(f"Correct answer: {df['answer_label'].iloc[0]}")
    
    df['answer_texts'] = df['answer_texts'].apply(ast.literal_eval)
    
    responses = get_answers_for_df(
        df, model, tokenizer, 
        sample_size=sample_size,
        debug_rows=debug_rows
    )
    
    responses.to_csv(output_path, index=False)
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
    parser.add_argument('--debug_rows', type=str, default=None,
                        help='Comma-separated list of row indices to debug, e.g., "0,1,5"')

    args = parser.parse_args()

    # Parse debug_rows
    debug_rows = None
    if args.debug_rows:
        debug_rows = [int(x.strip()) for x in args.debug_rows.split(',')]
        print(f"Will debug rows: {debug_rows}")

    print(f"Loading model from: {args.model_path}")
    print(f"Model name: {args.model_name}")

    # Load model from cluster
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Model successfully accessed from cluster.")

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
            sample_size=args.sample_size,
            debug_rows=debug_rows
        )

    print("\nAll datasets processed, cheers!")

    return

if __name__ == '__main__':
    main()