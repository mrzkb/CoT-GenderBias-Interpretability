# import statements
import pandas as pd

def clean_text(text):
    # Handle non-string types and None
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    
    # Convert to string and strip whitespace
    text_str = str(text).strip()
    
    # Check if empty after stripping
    if not text_str:
        return None
    
    # Check first character
    if text_str[0] in ['0', '1', '2']:
        return int(text_str[0])
    else:
        return int(-1)
    
    return None

# From each dataset:
#   Mina
#   CrowS
#   Stereo
#   BBQ

########## MINA
mina = pd.read_csv('data/mina/prompts_no_cot_unknown.csv')
mina['source'] = 'mina'
mina.rename(columns={'context': 'bias_type'}, inplace=True)
mina = mina[['example_id', 'source', 'bias_type', 'prompt', 'bias_label', 'unknown_label']]

# Let's add with and without CoT model answers
mina_no_cot = pd.read_csv('data/mina/llama8b_soapbox_no_cot_unknown_1113_1835.csv')
mina_cot = pd.read_csv('data/mina/llama8b_soapbox_cot_unknown_1113_1903.csv')

# Clean answer column to get predicted_answer
mina_no_cot['predicted_answer'] = mina_no_cot['answer'].apply(clean_text)
mina_cot['predicted_answer'] = mina_cot['answer'].apply(clean_text)

# Merge 
mina = mina.merge(
    mina_no_cot[['example_id', 'predicted_answer']], 
    on='example_id'
).rename(columns={'predicted_answer': 'predicted_answer_no_cot'})

mina = mina.merge(
    mina_cot[['example_id', 'predicted_answer']], 
    on='example_id'
).rename(columns={'predicted_answer': 'predicted_answer_cot'})

########## CROWS
crows = pd.read_csv('data/crows/prompts_no_cot_unknown.csv')
crows['source'] = 'crows'
crows = crows[['example_id', 'source', 'bias_type', 'prompt', 'bias_label', 'unknown_label']]

# Let's add with and without CoT model answers
crows_no_cot = pd.read_csv('data/crows/llama8b_soapbox_no_cot_unknown_1113_1704.csv')
crows_cot = pd.read_csv('data/crows/llama8b_soapbox_cot_unknown_1113_1715.csv')

# Clean answer column to get predicted_answer
crows_no_cot['predicted_answer'] = crows_no_cot['answer'].apply(clean_text)
crows_cot['predicted_answer'] = crows_cot['answer'].apply(clean_text)

# Merge 
crows = crows.merge(
    crows_no_cot[['example_id', 'predicted_answer']], 
    on='example_id'
).rename(columns={'predicted_answer': 'predicted_answer_no_cot'})

crows = crows.merge(
    crows_cot[['example_id', 'predicted_answer']], 
    on='example_id'
).rename(columns={'predicted_answer': 'predicted_answer_cot'})

########## STEREO
stereo = pd.read_csv('data/stereo/prompts_no_cot_unknown.csv')
stereo['source'] = 'stereo'
stereo = stereo[['example_id', 'source', 'bias_type', 'prompt', 'bias_label', 'unknown_label']]

# Let's add with and without CoT model answers
stereo_no_cot = pd.read_csv('data/stereo/llama8b_soapbox_no_cot_unknown_1113_1945.csv')
stereo_cot = pd.read_csv('data/stereo/llama8b_soapbox_cot_unknown_1113_2020.csv')

# Clean answer column to get predicted_answer
stereo_no_cot['predicted_answer'] = stereo_no_cot['answer'].apply(clean_text)
stereo_cot['predicted_answer'] = stereo_cot['answer'].apply(clean_text)

# Merge 
stereo = stereo.merge(
    stereo_no_cot[['example_id', 'predicted_answer']], 
    on='example_id'
).rename(columns={'predicted_answer': 'predicted_answer_no_cot'})

stereo = stereo.merge(
    stereo_cot[['example_id', 'predicted_answer']], 
    on='example_id'
).rename(columns={'predicted_answer': 'predicted_answer_cot'})

########## BBQ AMBIG
bbq_a = pd.read_csv('data/bbq/prompts_ambig_no_cot_unknown.csv')
bbq_a['source'] = 'bbq_ambig'
bbq_a['bias_type'] = 'gender'
bbq_a = bbq_a[['example_id', 'source', 'bias_type', 'prompt', 'bias_label', 'unknown_label']]

# Let's add with and without CoT model answers
bbq_a_no_cot = pd.read_csv('data/bbq/llama8b_soapbox_ambig_no_cot_unknown_1113_0944.csv')
bbq_a_cot = pd.read_csv('data/bbq/llama8b_soapbox_ambig_cot_unknown_1113_1034.csv')

# Clean answer column to get predicted_answer
bbq_a_no_cot['predicted_answer'] = bbq_a_no_cot['answer'].apply(clean_text)
bbq_a_cot['predicted_answer'] = bbq_a_cot['answer'].apply(clean_text)

# Merge 
bbq_a = bbq_a.merge(
    bbq_a_no_cot[['example_id', 'predicted_answer']], 
    on='example_id'
).rename(columns={'predicted_answer': 'predicted_answer_no_cot'})

bbq_a = bbq_a.merge(
    bbq_a_cot[['example_id', 'predicted_answer']], 
    on='example_id'
).rename(columns={'predicted_answer': 'predicted_answer_cot'})


########## BBQ DISAMBIG
bbq_b = pd.read_csv('data/bbq/prompts_disambig_no_cot_unknown.csv')
bbq_b['source'] = 'bbq_disambig'
bbq_b['bias_type'] = 'gender'
bbq_b = bbq_b[['example_id', 'source', 'bias_type', 'prompt', 'bias_label', 'unknown_label']]

# Let's add with and without CoT model answers
bbq_b_no_cot = pd.read_csv('data/bbq/llama8b_soapbox_disambig_no_cot_unknown_1113_1008.csv')
bbq_b_cot = pd.read_csv('data/bbq/llama8b_soapbox_disambig_cot_unknown_1113_1059.csv')

# Clean answer column to get predicted_answer
bbq_b_no_cot['predicted_answer'] = bbq_b_no_cot['answer'].apply(clean_text)
bbq_b_cot['predicted_answer'] = bbq_b_cot['answer'].apply(clean_text)

# Merge 
bbq_b = bbq_b.merge(
    bbq_b_no_cot[['example_id', 'predicted_answer']], 
    on='example_id'
).rename(columns={'predicted_answer': 'predicted_answer_no_cot'})

bbq_b = bbq_b.merge(
    bbq_b_cot[['example_id', 'predicted_answer']], 
    on='example_id'
).rename(columns={'predicted_answer': 'predicted_answer_cot'})

# Match samples from first pass
old_florian = pd.read_csv('data/florian_prompt_samples.csv')

source_dfs = {
    'mina': mina,
    'crows': crows,
    'stereo': stereo,
    'bbq_ambig': bbq_a,
    'bbq_disambig': bbq_b
}

# Process each source and concatenate
results = []
for source_name, source_df in source_dfs.items():
    filtered = old_florian[old_florian['source'] == source_name]
    merged = filtered.merge(
        source_df[['example_id', 'bias_label', 'unknown_label', 'predicted_answer_no_cot', 'predicted_answer_cot']], 
        on='example_id', 
        how='left'
    )
    results.append(merged)

old_florian_with_results = pd.concat(results, ignore_index=True)
old_florian_with_results[['example_id', 'source', 'bias_type', 'prompt', 'predicted_answer_no_cot', 'predicted_answer_cot']].to_csv('data/florian_prompt_samples_with_responses.csv')
old_florian_with_results.to_csv('data/florian_prompt_samples_key.csv')

# Random Sample of 10 prompts from each
# samples = []
# for dataset in [mina, crows, stereo, bbq_a, bbq_b]:
#     samples.append(dataset.sample(n=10, random_state=22))
# florian = pd.concat(samples, ignore_index=True)

# Save to CSV
# florian[['example_id', 'source', 'bias_type', 'prompt']].to_csv('data/florian_prompt_samples.csv')
# florian.to_csv('data/florian_prompt_samples_key.csv')


# I need to be more intentional with the samples, see the model response with and without CoT, when is it wrong, when is it right...
# Let's add with and without CoT model answers

# Let's add answer type