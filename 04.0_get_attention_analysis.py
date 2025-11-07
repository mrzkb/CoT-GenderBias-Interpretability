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

matrices = {}

for case in cases:
    # Extract case identifier from filename
    if 'gender_ambig_cot' in case:
        case_id = 'gender_ambig_cot'
    elif 'gender_ambig_no_cot' in case:
        case_id = 'gender_ambig_no_cot'
    elif 'gender_disambig_cot' in case:
        case_id = 'gender_disambig_cot'
    elif 'gender_disambig_no_cot' in case:
        case_id = 'gender_disambig_no_cot'
    
    # Load the per-head SAS data
    sas_df = pd.read_csv(case)

    # Create pivot table for heatmap (layers x heads matrix)
    sas_matrix = sas_df.groupby(['layer', 'head'])['nas'].mean().unstack() ##### COME BACK CHANGE TO 'sas'
    matrices[case_id] = sas_matrix
    
    # Create descriptive title and filename
    title = f'Average SAS per Head - {case_names[case_id]}'
    output_path = f'figures/sas_heatmap_{case_id}.png'
    
    create_heatmap(sas_matrix, title, output_path, 'Average SAS')



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