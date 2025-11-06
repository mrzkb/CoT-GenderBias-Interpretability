from datasets import load_dataset
import random
import pandas as pd

# Import BBQ prompts
bbq = pd.read_csv('data/prompts/gender_ambig_cot.csv')
    
# Import stereoset promptts
stereo = pd.read_csv('data/prompts/stereoset_cot.csv')
    
# Import crows prompts
crows = pd.read_csv('data/prompts/crowspairs_cot.csv')

# Reformat prompt 
def reformat_prompt(row):
    prompt = row['prompt']
    answer_texts = eval(row['answer_texts'])  # Convert string tuple to actual tuple
    unknown_label = row['unknown_label']
    
    # Get the two non-unknown answers
    non_unknown_answers = [ans for i, ans in enumerate(answer_texts) if i != unknown_label]
    
    # Remove everything from "Answer Options:" onwards
    prompt_without_options = re.split(r'Answer Options:', prompt)[0]
    
    # Create new answer options section with only two options
    new_answer_section = "Answer Options:\n"
    for i, answer in enumerate(non_unknown_answers):
        new_answer_section += f"            {i}) {answer}\n"
    
    # Combine and add the "Let's think step by step" part
    new_prompt = prompt_without_options + new_answer_section + "\nLet's think step by step"
    
    return new_prompt

# Usage example:
df = pd.read_csv('your_file.csv')
df['reformatted_prompt'] = df.apply(reformat_prompt, axis=1)

# To save the result
df.to_csv('reformatted_output.csv', index=False)
    