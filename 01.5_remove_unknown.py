from datasets import load_dataset
import random
import pandas as pd

# Import BBQ prompts
bbq = pd.read_csv('data/prompts/gender_ambig_cot.csv')
    
# Import stereoset promptts
stereo = pd.read_csv('data/prompts/stereoset_cot.csv')
    
# Import crows prompts
crows = pd.read_csv('data/prompts/crowspairs_cot.csv')

def reformat_prompt(row):
    prompt = row['prompt']
    answer_texts = eval(row['answer_texts'])  # Convert string tuple to actual tuple
    unknown_label = row['unknown_label']
    
    # Get the two non-unknown answers
    non_unknown_answers = [ans for i, ans in enumerate(answer_texts) if i != unknown_label]
    
    # Find where "Answer Options:" starts
    match = re.search(r'Answer Options:', prompt)
    if not match:
        return prompt  # If no answer options found, return original
    
    prompt_before = prompt[:match.start()]
    prompt_remaining = prompt[match.end():]
    
    # Find where the answer options end (after the last numbered option)
    # Look for the pattern of numbered options and capture everything after
    options_match = re.match(r'(.*?)\n\s*\d+\).*?(?=\n\n|\n(?!\s*\d+\))|$)', prompt_remaining, flags=re.DOTALL)
    
    # Find everything after the last answer option
    # Match all the answer option lines, then capture what comes after
    after_options = ''
    lines = prompt_remaining.split('\n')
    last_option_idx = -1
    
    for i, line in enumerate(lines):
        if re.match(r'\s*\d+\)', line.strip()):
            last_option_idx = i
    
    if last_option_idx >= 0 and last_option_idx < len(lines) - 1:
        # There's content after the last option
        after_options = '\n' + '\n'.join(lines[last_option_idx + 1:])
    
    # Create new answer options section with only two options
    new_answer_section = "Answer Options:\n"
    for i, answer in enumerate(non_unknown_answers):
        new_answer_section += f"            {i}) {answer}\n"
    
    # Combine all parts
    new_prompt = prompt_before + new_answer_section + after_options
    
    return new_prompt

# Usage example:
df = pd.read_csv('your_file.csv')
df['reformatted_prompt'] = df.apply(reformat_prompt, axis=1)

# To save the result
df.to_csv('reformatted_output.csv', index=False)
    