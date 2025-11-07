import pandas as pd

prompts = pd.read_csv('data/prompts_gender_ambig_cot_nounk.csv')
responses = pd.read_csv('data/Meta-Llama-3.1-8B-Instruct_responses_gender_ambig_cot_nounk.csv')

# print(prompts.columns.tolist())
# print(responses.columns.tolist())

print(len(prompts))
print(len(responses))

print(prompts['example_id'].nunique())
print(responses['example_id'].nunique())
print(len(prompts.groupby(['example_id', 'instruction_type'])))