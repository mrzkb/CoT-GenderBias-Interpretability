import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def create_heatmap(sas_matrix, title, output_path, cbar_label):
    plt.figure(figsize=(12, 10))
    sns.heatmap(sas_matrix, cmap='RdBu_r', center=0, vmin=-5, vmax=5, cbar_kws={'label': cbar_label})
    # sns.heatmap(sas_matrix, cmap='RdBu_r', center=0, cbar_kws={'label': cbar_label})
    plt.xlabel('Head')
    plt.ylabel('Layer')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close figure to free memory
    print(f"Saved: {output_path}")
    return

# Create output directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

cases = [
    'data/sas_rows_Meta-Llama-3.1-8B-Instruct_prompts_gender_ambig_cot.csv',
    'data/sas_rows_Meta-Llama-3.1-8B-Instruct_prompts_gender_ambig_no_cot.csv',
    'data/sas_rows_Meta-Llama-3.1-8B-Instruct_prompts_gender_disambig_cot.csv',
    'data/sas_rows_Meta-Llama-3.1-8B-Instruct_prompts_gender_disambig_no_cot.csv'
]

# Corresponding predictions files
predictions_files = [
    'data/Meta-Llama-3.1-8B-Instruct_responses_gender_ambig_cot.csv',
    'data/Meta-Llama-3.1-8B-Instruct_responses_gender_ambig_no_cot.csv',
    'data/Meta-Llama-3.1-8B-Instruct_responses_gender_disambig_cot.csv',
    'data/Meta-Llama-3.1-8B-Instruct_responses_gender_disambig_no_cot.csv'
]

# Define readable names for each case
case_names = {
    'gender_ambig_cot': 'Gender Ambiguous with CoT',
    'gender_ambig_no_cot': 'Gender Ambiguous without CoT',
    'gender_disambig_cot': 'Gender Disambiguated with CoT',
    'gender_disambig_no_cot': 'Gender Disambiguated without CoT'
}

for case, predictions_file in zip(cases, predictions_files):
    # Extract case identifier from filename
    if 'gender_ambig_cot' in case:
        case_id = 'gender_ambig_cot'
    elif 'gender_ambig_no_cot' in case:
        case_id = 'gender_ambig_no_cot'
    elif 'gender_disambig_cot' in case:
        case_id = 'gender_disambig_cot'
    elif 'gender_disambig_no_cot' in case:
        case_id = 'gender_disambig_no_cot'

    # Check if this is a disambiguated case
    is_disambig = 'disambig' in case_id

    # Load the per-head SAS data
    sas_df = pd.read_csv(case)
    
    # Load the corresponding predictions dataframe
    predictions_df = pd.read_csv(predictions_file)

    print(f"\n{case_names[case_id]}:")

    if is_disambig:
        # For disambiguated cases: split by correctness AND bias
        # Assume there's a 'correct_answer' or 'answer_label' column in the predictions
        # Adjust the column name as needed
        
        # Unknown predictions
        unknown_ids = predictions_df[
            predictions_df['predicted_answer'].astype(int) == predictions_df['unknown_label'].astype(int)
        ]['example_id'].unique()
        
        # Bias-correct: predicted bias label AND it's correct
        bias_correct_ids = predictions_df[
            (predictions_df['predicted_answer'].astype(int) == predictions_df['bias_label'].astype(int)) &
            (predictions_df['predicted_answer'].astype(int) == predictions_df['answer_label'].astype(int))
        ]['example_id'].unique()
        
        # Bias-incorrect: predicted bias label AND it's incorrect
        bias_incorrect_ids = predictions_df[
            (predictions_df['predicted_answer'].astype(int) == predictions_df['bias_label'].astype(int)) &
            (predictions_df['predicted_answer'].astype(int) != predictions_df['answer_label'].astype(int))
        ]['example_id'].unique()
        
        # Anti-stereotypical-correct: not unknown, not bias, AND correct
        anti_correct_ids = predictions_df[
            (predictions_df['predicted_answer'].astype(int) != predictions_df['unknown_label'].astype(int)) &
            (predictions_df['predicted_answer'].astype(int) != predictions_df['bias_label'].astype(int)) &
            (predictions_df['predicted_answer'].astype(int) == predictions_df['answer_label'].astype(int))
        ]['example_id'].unique()
        
        # Anti-stereotypical-incorrect: not unknown, not bias, AND incorrect
        anti_incorrect_ids = predictions_df[
            (predictions_df['predicted_answer'].astype(int) != predictions_df['unknown_label'].astype(int)) &
            (predictions_df['predicted_answer'].astype(int) != predictions_df['bias_label'].astype(int)) &
            (predictions_df['predicted_answer'].astype(int) != predictions_df['answer_label'].astype(int))
        ]['example_id'].unique()
        
        print(f"  Unknown label predictions: {len(unknown_ids)} examples")
        print(f"  Bias-correct predictions: {len(bias_correct_ids)} examples")
        print(f"  Bias-incorrect predictions: {len(bias_incorrect_ids)} examples")
        print(f"  Anti-stereotypical-correct predictions: {len(anti_correct_ids)} examples")
        print(f"  Anti-stereotypical-incorrect predictions: {len(anti_incorrect_ids)} examples")
        
        # Define groups for disambiguated cases
        groups = {
            'unknown': (unknown_ids, 'Predicted Unknown Label'),
            'bias_correct': (bias_correct_ids, 'Predicted Bias Label (Correct)'),
            'bias_incorrect': (bias_incorrect_ids, 'Predicted Bias Label (Incorrect)'),
            'anti_correct': (anti_correct_ids, 'Predicted Anti-Stereotypical (Correct)'),
            'anti_incorrect': (anti_incorrect_ids, 'Predicted Anti-Stereotypical (Incorrect)')
        }
        
    else:
    
        # Identify three groups of example_ids based on prediction behavior
        unknown_ids = predictions_df[
            predictions_df['predicted_answer'] == predictions_df['unknown_label']
        ]['example_id'].unique()
    
        bias_ids = predictions_df[
            predictions_df['predicted_answer'] == predictions_df['bias_label']
        ]['example_id'].unique()
    
        # Anti-stereotypical: predicted label matches neither unknown nor bias
        anti_stereo_ids = predictions_df[
            (predictions_df['predicted_answer'] != predictions_df['unknown_label']) &
            (predictions_df['predicted_answer'] != predictions_df['bias_label'])
        ]['example_id'].unique()
    
        print(f"  Unknown label predictions: {len(unknown_ids)} examples")
        print(f"  Bias label predictions: {len(bias_ids)} examples")
        print(f"  Anti-stereotypical predictions: {len(anti_stereo_ids)} examples")

        # Create three heatmaps for each group
        groups = {
            'unknown': (unknown_ids, 'Predicted Unknown Label'),
            'bias': (bias_ids, 'Predicted Bias Label'),
            'anti_stereo': (anti_stereo_ids, 'Predicted Anti-Stereotypical Label')
        }
    
    for group_name, (example_ids, group_title) in groups.items():
        if len(example_ids) == 0:
            print(f"  Skipping {group_name} - no examples")
            continue
            
        # Filter SAS data to only these example_ids
        group_sas = sas_df[sas_df['example_id'].isin(example_ids)]
        
        # Create pivot table for heatmap (layers x heads matrix)
        sas_matrix = group_sas.groupby(['layer', 'head'])['nas'].mean().unstack()
        
        # Create descriptive title and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        title = f'{group_title} - {case_names[case_id]}'
        output_path = f'figures/sas_heatmap_{case_id}_{group_name}_{timestamp}.png'
        
        create_heatmap(sas_matrix, title, output_path, 'Average SAS')

# This worked, don't get rid of it yet....
# matrices = {}

# for case in cases:
#     # Extract case identifier from filename
#     if 'gender_ambig_cot' in case:
#         case_id = 'gender_ambig_cot'
#     elif 'gender_ambig_no_cot' in case:
#         case_id = 'gender_ambig_no_cot'
#     elif 'gender_disambig_cot' in case:
#         case_id = 'gender_disambig_cot'
#     elif 'gender_disambig_no_cot' in case:
#         case_id = 'gender_disambig_no_cot'
    
#     # Load the per-head SAS data
#     sas_df = pd.read_csv(case)

#     # Create pivot table for heatmap (layers x heads matrix)
#     sas_matrix = sas_df.groupby(['layer', 'head'])['nas'].mean().unstack() ##### COME BACK CHANGE TO 'sas'
#     matrices[case_id] = sas_matrix
    
#     # Create descriptive title and filename
#     title = f'Average SAS per Head - {case_names[case_id]}'
#     output_path = f'figures/sas_heatmap_{case_id}.png'
    
#     create_heatmap(sas_matrix, title, output_path, 'Average SAS')


#############################################################################################
### Top k




#############################################################################################






#############################################################################################
### Create difference heatmaps (CoT - No CoT)

# Ambiguous case difference
# ambig_diff = matrices['gender_ambig_cot'] - matrices['gender_ambig_no_cot']
# create_heatmap(
#     ambig_diff, 
#     'SAS Difference (CoT - No CoT) - Ambiguous',
#     'figures/sas_heatmap_gender_ambig_diff.png',
#     'SAS Difference (CoT - No CoT)'
# )

# # Disambiguated case difference
# disambig_diff = matrices['gender_disambig_cot'] - matrices['gender_disambig_no_cot']
# create_heatmap(
#     disambig_diff,
#     'SAS Difference (CoT - No CoT) - Disambiguous',
#     'figures/sas_heatmap_gender_disambig_diff.png',
#     'SAS Difference (CoT - No CoT)'
# )
#############################################################################################

# Separate Prompts by Case
# Should the cases be changes in answer with the addition of CoT?
# OR should the cases be just biased vs. non-biased...

# Heatmap for Cases.