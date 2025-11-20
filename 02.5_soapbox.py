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

### LLAMA
#   /network/weights/llama.var/llama_3/Meta-Llama-3-8B-Instruct
#   /network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B-Instruct
#   /network/weights/llama.var/llama_3.3/Meta-Llama-3.3-70B-Instruct

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
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                ) # Other arguments include do_sample

        # Decode only the newly generated tokens (exclude the input)
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],  # Slice from end of input onwards
            skip_special_tokens=True
        )
        return generated_text
        
    except Exception as e:
        return f"ERROR: {str(e)}"

def count_answer_options(text):
    # This is messy, could flag falsely....
    # Count occurrences of the option patterns
    has_option_0 = '0)' in text
    has_option_1 = '1)' in text
    has_option_2 = '2)' in text
    
    # Count how many options are present
    option_count = sum([has_option_0, has_option_1, has_option_2])
    
    return option_count

def extract_cot_reasoning_and_answer(prompt, model, tokenizer):
    """
    Two-step Zero-shot-CoT extraction:
    1. First prompt: Extract reasoning with "Let's think step by step"
    2. Second prompt: Extract final answer from reasoning
    """
    
    reasoning_response = hey_model(
        prompt, 
        model, 
        tokenizer,
    )
    
    if reasoning_response.startswith("ERROR:"):
        return reasoning_response, None
    
    # Step 2: Answer Extraction
    # Define answer trigger based on format
    if count_answer_options(prompt) == 3:
        answer_trigger = "Therefore, among 0 through 2, the multiple choice answer is "
    else: 
        answer_trigger = "Therefore, among 0 through 1, the multiple choice answer is "
    
    # Construct second prompt: [X'] [Z] [A]
    answer_extraction_prompt = f"{reasoning_response} {answer_trigger}"
    
    answer_response = hey_model(
        answer_extraction_prompt,
        model,
        tokenizer,
        max_new_tokens=10,  # Shorter for answer extraction
    )
    
    if answer_response.startswith("ERROR:"):
        return reasoning_response, answer_response
    
    return reasoning_response, answer_response

def generate_responses_for_df(df, model, use_cot, tokenizer, sample_size=None, random_state=22):
    # Allow for testing on smaller subset of the df
    if sample_size is not None:
        original_size = len(df)
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state).reset_index(drop=True) # Be careful about the index here Edie, make sure that doesn't throw any errors down the line
        print(f"Sampling {len(df)} rows from {original_size} total rows")

    # Store results with example_id for safe merging
    responses = []
    start_time = time.time()

    for i, row in enumerate(tqdm(df.itertuples(), desc="Processing prompts", total=len(df))):
        example_id = row.example_id
        prompt = row.prompt
        
        iter_start = time.time()

        if use_cot:
            # Two-step CoT extraction
            reasoning, answer = extract_cot_reasoning_and_answer(
                prompt,
                model,
                tokenizer,
            )
            
            response_row = {
                'example_id': example_id,
                'reasoning': reasoning,
                'answer': answer,
            }

            status = "✗" if reasoning.startswith("ERROR:") or (answer and answer.startswith("ERROR:")) else "✓"
            
        else:
            # Standard single-step prompting
            response = hey_model(
                prompt,
                model,
                tokenizer,

            )
            
            response_row = {
                'example_id': example_id,
                'answer': response,
            }
            
            status = "✗" if response.startswith("ERROR:") else "✓"

        responses.append(response_row)
    
        # Timing information
        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        avg_time = elapsed / (i+1)
        remaining = avg_time * (len(df) - i - 1)

        # Progress Update
        tqdm.write(f"{status} Row {i+1}/{len(df)} | Time: {iter_time:.2f}s | Avg: {avg_time:.2f}s | ETA: {remaining/60:.1f}m")

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time/60:.2f} minutes ({total_time/len(df):.2f}s per prompt)")
    
    # Create results dataframe and merge on example_id
    responses_df = pd.DataFrame(responses)
    df = df.merge(responses_df, on='example_id', how='left')

    return df

def process_dataset(input_filename, model_name, model, tokenizer, sample_size=None):
    print("\n" + "=" * 60)
    print(f"Processing: {input_filename}")
    print("=" * 60)

    os.makedirs('data', exist_ok=True)
    if not os.path.exists(input_filename):
        raise FileNotFoundError(f"Input file not found: {input_filename}")
    
    if 'no_cot' in input_filename:
        use_cot=False
    else:
        use_cot=True

    df = pd.read_csv(input_filename)
    responses = generate_responses_for_df(df, model, use_cot, tokenizer, sample_size)
   
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base_name = input_filename.replace('prompts_', f'{model_name}_soapbox_').replace('.csv', '')
    output_filename = f'{base_name}_{timestamp}.csv'

    responses.to_csv(output_filename, index=False) 
    print(f"✓ Saved to: {output_filename}\n")
    return

def configure_model(model='llama8b'):
    if model == 'llama8b':
        model_name = 'Meta-Llama-3.1-8B-Instruct'
        model_path = '/network/weights/llama.var/llama_3.1/Meta-Llama-3.1-8B-Instruct'
    elif model == 'llama70b':
        model_name = 'Meta-Llama-3.3-70B-Instruct'
        model_path = '/network/weights/llama.var/llama_3.3/Meta-Llama-3.3-70B-Instruct'

    print(f"Model name: {model_name}")
    print(f"Loading model from: {model_path}")

    # Load model from cluster
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype=torch.float16,    
        device_map="auto"   
    )

    return model, tokenizer

def configure_data(source='bbq', cot=False, unknown=True, instruction_version=None):
    # Base paths for each source
    source_configs = {
        'bbq': {
            'base_dir': 'data/bbq/',
            'variants': ['ambig', 'disambig']
        },
        'stereo': {
            'base_dir': 'data/stereo/',
            'variants': [None]
        },
        'crows': {
            'base_dir': 'data/crows/',
            'variants': [None]
        },
        'mina': {
            'base_dir': 'data/mina/',
            'variants': [None]
        }
    }
    
    config = source_configs[source]
    
    # Build filename components
    cot_str = 'cot' if cot else 'no_cot'
    unknown_str = 'unknown' if unknown else 'no_unknown'
    
    # Add instruction version suffix only for unknown cases
    version_suffix = ''
    if unknown and instruction_version:
        version_suffix = f'_{instruction_version}'
    
    # Generate dataset list
    datasets = []
    for variant in config['variants']:
        if variant is None:
            filename = f"{config['base_dir']}prompts_{cot_str}_{unknown_str}{version_suffix}.csv"
        else:
            filename = f"{config['base_dir']}prompts_{variant}_{cot_str}_{unknown_str}{version_suffix}.csv"
        datasets.append(filename)
    
    return datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama8b', 
                       choices=['llama8b', 'llama70b'])
    parser.add_argument('--source', type=str, default='bbq',
                       choices=['bbq', 'stereo', 'crows', 'mina'])
    parser.add_argument('--cot', action='store_true',
                       help='Use chain-of-thought prompts')
    parser.add_argument('--unknown', action='store_true',
                       help='Include unknown option')
    parser.add_argument('--instruction_version', type=str, default=None,
                       help='Instruction version suffix (e.g., IV2). Only applies when --unknown is set.')
    parser.add_argument('--sample_size', type=int, default=None)
    args = parser.parse_args()
    
    # Configure model
    model, tokenizer = configure_model(model=args.model)
    print("Model successfully accessed from cluster.")

    # Configure data
    datasets = configure_data(
        source=args.source, 
        cot=args.cot, 
        unknown=args.unknown,
        instruction_version=args.instruction_version)
    
    print(f"\nProcessing {len(datasets)} datasets:")
    for d in datasets:
        print(f"  - {d}")
    
    # Process each dataset
    for dataset in datasets:
        process_dataset(
            input_filename=dataset,
            model_name=args.model,
            model=model,
            tokenizer=tokenizer,
            sample_size=args.sample_size
        )
    print("\nAll datasets processed, cheers!")

if __name__ == '__main__':
    main()